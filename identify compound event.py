import os
import math
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from dask.diagnostics import ProgressBar
from joblib import Parallel, delayed
import dask

warnings.filterwarnings("ignore", category=FutureWarning)

# =========================
# User settings (EDIT)
# =========================
INPUT_XLSX = r"G:\LHY\[new]\IBTrACS\IBTrACS.since1980.v04r01__landfall_eq0_only__wind_1min_ms__with_vmax_lifetime.xlsx"
LAND_SHEET_PREFIX = "landfall_eq"   # exporter sheet name like landfall_eq0

# ERA5 daily Tmax files
ERA5_DAILY_TMAX_TPL = r"D:/data/ERA5-2m daily max temperature/daily_max_{}.nc"
ERA5_VAR = "t2m_max"  # Kelvin

YEARS = np.arange(1980, 2025)  # adjust if needed

# ★ NEW: Baseline years for P90 climatology
BASELINE_START = 1980
BASELINE_END   = 1999  # inclusive

# Output
OUT_CSV = r"G:\LHY\[new]\TC-HW compound\TC-HW\compound_events_post30d_hourly_mincols.csv"
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

# Window definition: post 30 days starting at landfall time (hourly boundary)
POST_DAYS = 30
FORCE_FLOOR_TO_HOUR = True  # ensure "hour precision" (e.g., 12:34 -> 12:00)

# HW definition (strict)
DOY_PAD = 7
HEMIS_SEASON = dict(N=[5, 6, 7, 8, 9], S=[11, 12, 1, 2, 3])
END_ON_COLD = 1             # any 1 cold day ends event (no bridging)
MIN_CONSEC_HOT = 3          # >=3 consecutive hot days to start event

# Performance
N_WORKERS    = max(1, os.cpu_count() - 1)
BATCH_POINTS = 2000
TIME_CHUNK   = 365 * 8
dask.config.set(scheduler="threads", num_workers=N_WORKERS)

# Optional numba
try:
    from numba import njit
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False

# ★关键：不要用 numba 的 O(T^2) 分位数实现（会导致跑十几个小时）
USE_NUMBA_QUANTILE = False


# =========================
# Helpers
# =========================
def _find_first_existing_era5_file(years):
    for y in years:
        fp = ERA5_DAILY_TMAX_TPL.format(int(y))
        if os.path.exists(fp):
            return fp
    return None

def _preprocess_era5(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    if "latitude" in ds.coords or "latitude" in ds.dims:
        rename["latitude"] = "lat"
    if "longitude" in ds.coords or "longitude" in ds.dims:
        rename["longitude"] = "lon"
    if rename:
        ds = ds.rename(rename)

    if "expver" in ds.dims:
        ds = ds.isel(expver=0).drop_vars(["expver"], errors="ignore")
    if "number" in ds.dims:
        ds = ds.isel(number=0).drop_vars(["number"], errors="ignore")
    else:
        ds = ds.drop_vars(["number"], errors="ignore")

    keep = [ERA5_VAR, "lat", "lon", "time"]
    drop_these = [v for v in ds.variables if v not in keep and v not in ("lat", "lon", "time")]
    ds = ds.drop_vars(drop_these, errors="ignore")
    return ds

def wrap_lon_to_domain(lon_vals, era5_lon_vals):
    lon_vals = np.array(lon_vals, dtype=float)
    lon_min = float(np.nanmin(era5_lon_vals))
    if lon_min >= 0.0:  # ERA5 lon in [0, 360]
        return np.where(lon_vals < 0, lon_vals + 360.0, lon_vals)
    else:  # ERA5 lon in [-180, 180]
        return ((lon_vals + 180.0) % 360.0) - 180.0

def nearest_index(arr, value):
    arr = np.asarray(arr, dtype=float)
    return int(np.nanargmin(np.abs(arr - float(value))))

def build_summer_mask(lat, time_index: pd.DatetimeIndex):
    months = time_index.month.values
    mask = np.isin(months, HEMIS_SEASON["N"] if lat >= 0 else HEMIS_SEASON["S"])
    return mask.astype(np.bool_)

def floor_hour(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(ts).floor("H")

def floor_day(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(ts).floor("D")

def intervals_overlap(a_start, a_end, b_start, b_end) -> bool:
    
    return (a_start < b_end) and (a_end > b_start)

def _pick_vmax_lifetime_column(df: pd.DataFrame):

    cols = list(df.columns)
    cols_low = [str(c).strip().lower() for c in cols]

    candidates = [
        "vmax_lifetime_1min",
        "vmax_lifetime",
        "vmax_lifetime_ms",
        "vmax_lifetime_mps",
        "lifetime_vmax",
        "lifetime_vmax_1min",
        "lifetime_vmax_ms",
        "lifetime_max_wind",
        "max_wind_lifetime",
        "vmax_life",
        "vmax_life_1min",
    ]
    for cand in candidates:
        if cand in cols_low:
            return cols[cols_low.index(cand)]

    for i, c in enumerate(cols_low):
        if ("lifetime" in c or "life" in c) and ("vmax" in c or ("max" in c and "wind" in c)):
            return cols[i]
    return None


# -------------------------
# HW quantile + detection
# -------------------------
if NUMBA_OK:
    @njit(fastmath=True)
    def _percentile(a, q):
        vals = a[~np.isnan(a)]
        if vals.size == 0:
            return np.nan
        vals.sort()
        k = (q / 100.0) * (vals.size - 1)
        f = int(np.floor(k))
        c = int(np.ceil(k))
        if f == c:
            return vals[f]
        return vals[f] + (vals[c] - vals[f]) * (k - f)

    @njit(fastmath=True)
    def compute_clim_quantiles_point_numba(T_c, doy, pad, max_doy):
        # 保留原函数不删，但默认不启用（USE_NUMBA_QUANTILE=False）
        p90 = np.empty(T_c.shape[0], dtype=np.float32)
        p75 = np.empty(T_c.shape[0], dtype=np.float32)
        p25 = np.empty(T_c.shape[0], dtype=np.float32)

        for i in range(T_c.shape[0]):
            d = doy[i]
            lo = d - pad
            hi = d + pad
            acc = []
            for dd in range(lo, hi + 1):
                dd_wrap = dd
                while dd_wrap <= 0:
                    dd_wrap += max_doy
                while dd_wrap > max_doy:
                    dd_wrap -= max_doy
                for j in range(T_c.shape[0]):
                    if doy[j] == dd_wrap:
                        acc.append(T_c[j])
            arr = np.array(acc, dtype=np.float32)
            mask = ~np.isnan(arr)
            arr = arr[mask]
            if arr.size == 0:
                p90[i] = np.nan
                p75[i] = np.nan
                p25[i] = np.nan
            else:
                p90[i] = _percentile(arr, 90.0)
                p75[i] = _percentile(arr, 75.0)
                p25[i] = _percentile(arr, 25.0)

        return p90, np.empty(T_c.shape[0], dtype=np.float32) * np.nan

    @njit(fastmath=True)
    def detect_hw_events_numba(is_hot, min_consec_hot, end_on_cold):
        T = is_hot.shape[0]
        starts = np.empty(T, dtype=np.int32)
        ends   = np.empty(T, dtype=np.int32)
        n_ev = 0

        in_event = False
        above = 0
        cold = 0
        start_i = -1

        for i in range(T):
            if is_hot[i]:
                if in_event:
                    cold = 0
                else:
                    above += 1
                    if above >= min_consec_hot:
                        start_i = i - above + 1
                        in_event = True
                        cold = 0
            else:
                if in_event:
                    cold += 1
                    above = 0
                    if cold == end_on_cold:
                        end_i = i - 1
                        starts[n_ev] = start_i
                        ends[n_ev] = end_i
                        n_ev += 1
                        in_event = False
                        cold = 0
                else:
                    above = 0

        if in_event:
            starts[n_ev] = start_i
            ends[n_ev] = T - 1
            n_ev += 1

        out = np.empty((n_ev, 2), dtype=np.int32)
        for k in range(n_ev):
            out[k, 0] = starts[k]
            out[k, 1] = ends[k]
        return out

# Python fallback / DOY-aggregate quantiles (FAST)
def compute_clim_quantiles_point_py(
    T_c,
    t_idx: pd.DatetimeIndex,
    summer_mask: np.ndarray,
    baseline_mask: np.ndarray,
    pad=DOY_PAD
):
    
    doy = t_idx.dayofyear.values  # 1..365/366
    max_doy = 366
    p90_by_doy = np.full(max_doy + 1, np.nan, dtype=np.float32)

    # Map doy -> indices, but only keep (summer & baseline) indices
    doy_to_idx = defaultdict(list)
    for i, d in enumerate(doy):
        if bool(summer_mask[i]) and bool(baseline_mask[i]):
            doy_to_idx[int(d)].append(i)

    for d in range(1, max_doy + 1):
        lo, hi = d - pad, d + pad
        window = ((np.arange(lo, hi + 1) - 1) % max_doy) + 1

        gather_idx = []
        for wd in window:
            gather_idx.extend(doy_to_idx.get(int(wd), []))
        if not gather_idx:
            continue

        vals = T_c[gather_idx]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue

        p90_by_doy[d] = np.percentile(vals, 90.0)

    p90_arr = p90_by_doy[doy]
    return p90_arr.astype(np.float32)

def detect_hw_events_py(is_hot: np.ndarray, min_consec_hot=MIN_CONSEC_HOT, end_on_cold=END_ON_COLD):
    T = len(is_hot)
    events = []
    in_event = False
    above = 0
    cold = 0
    start_i = None

    for i in range(T):
        if bool(is_hot[i]):
            if in_event:
                cold = 0
            else:
                above += 1
                if above >= min_consec_hot:
                    start_i = i - above + 1
                    in_event = True
                    cold = 0
        else:
            if in_event:
                cold += 1
                above = 0
                if cold == end_on_cold:
                    end_i = i - 1
                    events.append((int(start_i), int(end_i)))
                    in_event = False
                    cold = 0
            else:
                above = 0

    if in_event:
        events.append((int(start_i), T - 1))
    return np.array(events, dtype=np.int32)


# =========================
# Step A: Read landfall Excel
# =========================
if not os.path.exists(INPUT_XLSX):
    raise FileNotFoundError(f"Excel not found: {INPUT_XLSX}")

print(f"[A] Reading landfall Excel: {INPUT_XLSX}")
sheets = pd.read_excel(INPUT_XLSX, sheet_name=None, engine="openpyxl")
land_sheet = None
for sname in sheets:
    if str(sname).startswith(LAND_SHEET_PREFIX):
        land_sheet = sname
        break
if land_sheet is None:
    raise RuntimeError(f"No sheet starts with '{LAND_SHEET_PREFIX}'. Found: {list(sheets.keys())}")

df_lf = sheets[land_sheet].copy()
required_cols = ["sid", "time_utc", "lat", "lon"]
miss = [c for c in required_cols if c not in df_lf.columns]
if miss:
    raise KeyError(f"Missing required columns in sheet '{land_sheet}': {miss}. Available: {list(df_lf.columns)[:50]} ...")

df_lf["sid"] = df_lf["sid"].astype(str)
df_lf["landfall_time_utc"] = pd.to_datetime(df_lf["time_utc"], errors="coerce")
df_lf["landfall_lat"] = pd.to_numeric(df_lf["lat"], errors="coerce")
df_lf["landfall_lon_raw"] = pd.to_numeric(df_lf["lon"], errors="coerce")

df_lf = df_lf.dropna(subset=["landfall_time_utc", "landfall_lat", "landfall_lon_raw"]).reset_index(drop=True)

if FORCE_FLOOR_TO_HOUR:
    df_lf["landfall_time_utc"] = df_lf["landfall_time_utc"].dt.floor("H")

print(f"    Landfall rows (valid, pre-dedup): {len(df_lf)}")

# -------- lifetime vmax column (per SID) --------
vmax_life_col = _pick_vmax_lifetime_column(df_lf)
if vmax_life_col is None:
    print("    [WARN] No lifetime vmax column detected in Excel. Output vmax_lifetime_1min will be NaN.")
    df_lf["vmax_lifetime_1min"] = np.nan
else:
    df_lf["_vmax_life_row"] = pd.to_numeric(df_lf[vmax_life_col], errors="coerce")
    df_lf["vmax_lifetime_1min"] = df_lf.groupby("sid")["_vmax_life_row"].transform("max")
    print(f"    Detected lifetime vmax column: '{vmax_life_col}' -> output as vmax_lifetime_1min")

# Find wind columns in Excel (already 1-min and m/s after your Code2)
wind_cols = [c for c in df_lf.columns if str(c).lower().endswith("_wind")]
if not wind_cols:
    print("    [WARN] No '*_wind' columns found in landfall Excel. vmax_landfall_day_1min will be NaN.")
    df_lf["_vmax_row"] = np.nan
else:
    df_lf["_vmax_row"] = df_lf[wind_cols].apply(pd.to_numeric, errors="coerce").max(axis=1)

# per sid + landfall day max (UTC day)
df_lf["_lf_day"] = df_lf["landfall_time_utc"].dt.floor("D")
df_lf["vmax_landfall_day_1min"] = df_lf.groupby(["sid", "_lf_day"])["_vmax_row"].transform("max")


# =========================
# Step B: Probe ERA5 grid & map landfalls to nearest grid
# =========================
probe_fp = _find_first_existing_era5_file(YEARS)
if probe_fp is None:
    raise FileNotFoundError("No ERA5 daily_max_YYYY.nc found for given YEARS.")

print(f"[B] Probing ERA5 grid from: {probe_fp}")
with xr.open_dataset(probe_fp) as ds0:
    ds0 = _preprocess_era5(ds0)
    era5_lat_vals = ds0["lat"].values
    era5_lon_vals = ds0["lon"].values

# Wrap lon to ERA5 domain and map to nearest grid indices
df_lf["landfall_lon"] = wrap_lon_to_domain(df_lf["landfall_lon_raw"].values, era5_lon_vals)

ilat_list, ilon_list, grid_lat_list, grid_lon_list = [], [], [], []
for lat, lon in zip(df_lf["landfall_lat"].values, df_lf["landfall_lon"].values):
    ilat = nearest_index(era5_lat_vals, lat)
    ilon = nearest_index(era5_lon_vals, lon)
    ilat_list.append(ilat)
    ilon_list.append(ilon)
    grid_lat_list.append(float(era5_lat_vals[ilat]))
    grid_lon_list.append(float(era5_lon_vals[ilon]))

df_lf["ilat"] = np.array(ilat_list, dtype=int)
df_lf["ilon"] = np.array(ilon_list, dtype=int)
df_lf["grid_lat"] = np.array(grid_lat_list, dtype=float)
df_lf["grid_lon"] = np.array(grid_lon_list, dtype=float)

# =========================
# Deduplicate AFTER mapping to ERA5 grid
# same sid + same (ilat,ilon) -> keep FIRST enter time
# =========================
n_before = len(df_lf)
df_lf = (
    df_lf.sort_values("landfall_time_utc")
         .drop_duplicates(subset=["sid", "ilat", "ilon"], keep="first")
         .reset_index(drop=True)
)
n_after = len(df_lf)
print(f"    Dedup by (sid, grid): {n_before} -> {n_after} (keep earliest enter time)")

uniq_pairs = np.unique(df_lf[["ilat", "ilon"]].values, axis=0)
P = uniq_pairs.shape[0]
print(f"    Unique ERA5 grids (after dedup): {P}")

pair_to_point = {(int(i), int(j)): int(p) for p, (i, j) in enumerate(uniq_pairs)}
target_lats = era5_lat_vals[uniq_pairs[:, 0]]
target_lons = era5_lon_vals[uniq_pairs[:, 1]]


# =========================
# Step C: Load ERA5 daily Tmax time series for all target grids
# =========================
print("[C] Opening ERA5 multi-year daily Tmax ...")
files = [ERA5_DAILY_TMAX_TPL.format(int(y)) for y in YEARS if os.path.exists(ERA5_DAILY_TMAX_TPL.format(int(y)))]
if not files:
    raise FileNotFoundError("No ERA5 files found. Check ERA5_DAILY_TMAX_TPL and YEARS.")

ds_all = xr.open_mfdataset(
    files,
    combine="nested",
    concat_dim="time",
    preprocess=_preprocess_era5,
    engine="netcdf4",
    parallel=True,
    chunks={"time": TIME_CHUNK},
    data_vars="minimal",
    coords="minimal",
    compat="override",
)

lat_vals = ds_all["lat"].values
lon_vals = ds_all["lon"].values

lat_to_idx = {float(v): i for i, v in enumerate(lat_vals)}
lon_to_idx = {float(v): i for i, v in enumerate(lon_vals)}
lat_idx = np.array([lat_to_idx[float(v)] for v in target_lats], dtype=int)
lon_idx = np.array([lon_to_idx[float(v)] for v in target_lons], dtype=int)

da_pts = ds_all[ERA5_VAR].isel(
    lat=xr.DataArray(lat_idx, dims="point"),
    lon=xr.DataArray(lon_idx, dims="point")
).astype("float32")

t_idx_full = pd.to_datetime(da_pts["time"].values)
da_pts_c = da_pts - np.float32(273.15)  # Kelvin -> Celsius
T_days = da_pts_c.sizes["time"]

print(f"    Daily time length: {T_days}, points: {P}")
print("[C] Loading (time, point) into memory (batched) ...")

T_all = np.empty((T_days, P), dtype=np.float32)
n_batches = math.ceil(P / BATCH_POINTS)
for b in tqdm(range(n_batches), desc="    load batches", dynamic_ncols=True):
    s = b * BATCH_POINTS
    e = min((b + 1) * BATCH_POINTS, P)
    da_chunk = da_pts_c.isel(point=slice(s, e)).chunk({"time": -1, "point": -1})
    with ProgressBar():
        T_all[:, s:e] = da_chunk.load().values

ds_all.close()


# =========================
# Step D: Build cache per grid point (p90, events)
# =========================
print("[D] Precomputing HW cache per ERA5 grid point (summer-only P90; baseline 1980-1999) ...")
DOY_VEC = pd.DatetimeIndex(t_idx_full).dayofyear.values.astype(np.int32)
MAX_DOY = 366
point_lats = target_lats.astype(float)
point_lons = target_lons.astype(float)

# ★ NEW: baseline mask on the FULL time axis (used only for building p90_by_doy)
t_idx_dt_full = pd.DatetimeIndex(t_idx_full)
baseline_mask_full = (
    (t_idx_dt_full.year >= int(BASELINE_START)) &
    (t_idx_dt_full.year <= int(BASELINE_END))
).astype(np.bool_)

print(f"    Baseline years for P90: {BASELINE_START}-{BASELINE_END} "
      f"(days in baseline on time axis: {int(np.sum(baseline_mask_full))})")

def _build_cache_one_point(p: int):
    T_c = T_all[:, p].astype(np.float32)
    lat0 = float(point_lats[p])
    t_idx_dt = pd.DatetimeIndex(t_idx_full)

    summer_mask = build_summer_mask(lat0, t_idx_dt)

    # p90 threshold only — NOW: summer-only & baseline-only samples
    if NUMBA_OK and USE_NUMBA_QUANTILE:
        # 注意：numba 分位数版本未实现 baseline-only/summer-only 限制；默认 USE_NUMBA_QUANTILE=False 不走这里
        p90_arr, _dummy = compute_clim_quantiles_point_numba(T_c, DOY_VEC, DOY_PAD, MAX_DOY)
    else:
        p90_arr = compute_clim_quantiles_point_py(
            T_c,
            t_idx_dt,
            summer_mask=summer_mask,
            baseline_mask=baseline_mask_full,
            pad=DOY_PAD
        )

    is_hot = (T_c > p90_arr) & summer_mask & np.isfinite(T_c) & np.isfinite(p90_arr)

    # HW event detection (same strict rule)
    if NUMBA_OK:
        ev_idx = detect_hw_events_numba(is_hot.astype(np.bool_), MIN_CONSEC_HOT, END_ON_COLD)
        events_full = [(int(s), int(e)) for (s, e) in ev_idx]
    else:
        ev_idx = detect_hw_events_py(is_hot, MIN_CONSEC_HOT, END_ON_COLD)
        events_full = [(int(s), int(e)) for (s, e) in ev_idx]

    return dict(
        T=T_c,
        p90=p90_arr.astype(np.float32),
        events_full=events_full,
        lat=lat0,
        lon=float(point_lons[p]),
    )

cache_list = Parallel(n_jobs=N_WORKERS, prefer="threads", batch_size=1)(
    delayed(_build_cache_one_point)(p) for p in tqdm(range(P), desc="    cache points", dynamic_ncols=True)
)
grid_cache = {p: cache_list[p] for p in range(P)}


# =========================
# Step E: For each (deduped) landfall, intersect post-window with HW events
# (UPDATED: TC-HW only if HW START is within window; keep both FULL and CLIPPED features)
# =========================
print(f"[E] Intersecting post-{POST_DAYS}-day window with HW events (hour-precise boundaries) ...")
records = []

t_vals64 = pd.DatetimeIndex(t_idx_full).values  # numpy datetime64[ns]

def iso_hour(ts: pd.Timestamp) -> str:
    ts = pd.Timestamp(ts)
    if FORCE_FLOOR_TO_HOUR:
        ts = ts.floor("H")
    return ts.strftime("%Y-%m-%dT%H:%M:%S")

for _, row in tqdm(df_lf.iterrows(), total=len(df_lf), desc="    landfalls(deduped)", dynamic_ncols=True):
    sid = row["sid"]
    lf_time = pd.Timestamp(row["landfall_time_utc"])
    if FORCE_FLOOR_TO_HOUR:
        lf_time = floor_hour(lf_time)

    window_start = lf_time
    window_end = lf_time + pd.Timedelta(days=POST_DAYS)  # half-open [start, end)

    ilat = int(row["ilat"])
    ilon = int(row["ilon"])
    p = pair_to_point.get((ilat, ilon), None)
    if p is None:
        continue

    cache = grid_cache[p]
    T_c = cache["T"]
    p90_arr = cache["p90"]
    events_full = cache["events_full"]

    # KEEP ALL independent HW events whose START TIME is within the window
    for (s_full, e_full) in events_full:
        # FULL HW event time-interval (half-open): [day00(s_full), day00(e_full)+1day)
        ev_start = floor_day(t_idx_full[s_full])
        ev_end = floor_day(t_idx_full[e_full]) + pd.Timedelta(days=1)

        # --------------------------
        # (1) UPDATED TC-HW criterion:
        #     ONLY if HW START is within [window_start, window_end)
        # --------------------------
        if not (ev_start >= window_start and ev_start <= window_end):
            continue

        # --------------------------
        # (2) FULL HW features (on DAILY axis, full s_full..e_full)
        # --------------------------
        T_full = T_c[s_full:e_full + 1]
        p90_full = p90_arr[s_full:e_full + 1]
        hot_mask_full = (T_full > p90_full) & np.isfinite(T_full) & np.isfinite(p90_full)
        days_hot_full = int(np.nansum(hot_mask_full))
        if days_hot_full <= 0:
            # extremely unlikely under strict HW definition, but keep safe
            continue

        exceed_full = (T_full - p90_full).astype(np.float32)
        exceed_full_hot = np.where(hot_mask_full, exceed_full, 0.0).astype(np.float32)
        hw_intensity_full = float(np.nansum(np.where(np.isfinite(exceed_full_hot), exceed_full_hot, 0.0)))
        hw_duration_full_days = int(e_full - s_full + 1)

        # --------------------------
        # (3) CLIP to window (hour-precise boundaries), still half-open
        # --------------------------
        clip_start_dt = max(ev_start, window_start)
        clip_end_dt = min(ev_end, window_end)
        if clip_start_dt >= clip_end_dt:
            continue

        # Map clip interval to DAILY indices for duration/intensity:
        clip_start_day00 = floor_day(clip_start_dt)
        clip_last_day00 = floor_day(clip_end_dt - pd.Timedelta(nanoseconds=1))

        s_idx = int(np.searchsorted(t_vals64, np.datetime64(clip_start_day00)))
        e_idx = int(np.searchsorted(t_vals64, np.datetime64(clip_last_day00), side="right") - 1)

        # clamp within this HW event index range
        s_idx = max(s_idx, s_full)
        e_idx = min(e_idx, e_full)
        if s_idx > e_idx:
            continue

        T_seg = T_c[s_idx:e_idx + 1]
        p90_seg = p90_arr[s_idx:e_idx + 1]

        # hot days within this clipped segment
        hot_mask = (T_seg > p90_seg) & np.isfinite(T_seg) & np.isfinite(p90_seg)
        days_hot = int(np.nansum(hot_mask))
        if days_hot <= 0:
            continue

        # intensity: sum(Tmax - T90) over hot days only (CLIPPED)
        exceed = (T_seg - p90_seg).astype(np.float32)
        exceed_hot = np.where(hot_mask, exceed, 0.0).astype(np.float32)
        intensity = float(np.nansum(np.where(np.isfinite(exceed_hot), exceed_hot, 0.0)))

        duration_days = int(e_idx - s_idx + 1)

        records.append(dict(
            sid=sid,
            landfall_time_utc=iso_hour(lf_time),
            landfall_lat=float(row["landfall_lat"]),
            landfall_lon=float(row["landfall_lon"]),
            grid_lat=float(row["grid_lat"]),
            grid_lon=float(row["grid_lon"]),
            window_start=iso_hour(window_start),
            window_end=iso_hour(window_end),

            # FULL HW features
            hw_start=iso_hour(ev_start),
            hw_end=iso_hour(ev_end),
            hw_duration=hw_duration_full_days,
            hw_intensity=float(hw_intensity_full),

            # CLIPPED-to-window features (your old outputs)
            start_clip=iso_hour(clip_start_dt),
            end_clip=iso_hour(clip_end_dt),
            duration=duration_days,
            intensity=float(intensity),

            vmax_landfall_day_1min=(float(row["vmax_landfall_day_1min"]) if np.isfinite(row["vmax_landfall_day_1min"]) else np.nan),
            vmax_lifetime_1min=(float(row["vmax_lifetime_1min"]) if np.isfinite(row.get("vmax_lifetime_1min", np.nan)) else np.nan),
        ))

df_out = pd.DataFrame.from_records(records)

print(f"[F] Matched TC-HW records (HW start within window; keep FULL+CLIPPED): {len(df_out)}")

# Ensure column order exactly as requested (UPDATED with full HW feature fields)
cols_out = [
    "sid",
    "landfall_time_utc",
    "landfall_lat",
    "landfall_lon",
    "grid_lat",
    "grid_lon",
    "window_start",
    "window_end",

    # FULL HW features
    "hw_start",
    "hw_end",
    "hw_duration",
    "hw_intensity",

    # CLIPPED (window) features
    "start_clip",
    "end_clip",
    "duration",
    "intensity",

    "vmax_landfall_day_1min",
    "vmax_lifetime_1min",
]
for c in cols_out:
    if c not in df_out.columns:
        df_out[c] = np.nan
df_out = df_out[cols_out]

# Sort for readability
if not df_out.empty:
    df_out["_t"] = pd.to_datetime(df_out["landfall_time_utc"], errors="coerce")
    df_out = df_out.sort_values(["sid", "_t", "hw_start", "start_clip"]).drop(columns=["_t"]).reset_index(drop=True)

print(f"[F] Writing CSV -> {OUT_CSV}")
df_out.to_csv(OUT_CSV, index=False, encoding="utf-8")
print("Done.")
print(f"Saved:\n  {OUT_CSV}")
