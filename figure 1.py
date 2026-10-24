import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from cartopy.io import shapereader
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.prepared import prep
import pyproj
from shapely.ops import transform as shp_transform

from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# ============ optional scipy ============
_HAVE_SCIPY = True
try:
    from scipy import stats
except Exception:
    _HAVE_SCIPY = False

# Fonts
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Arial']

# ====================== Paths (EDIT) ======================
IBTRACS_NC  = r"G:\LHY\IBTrACS\IBTrACS.since1980.v04r01--original data.nc"

# total TC count from landfall xlsx
LANDFALL_XLSX = r"G:\LHY\[new]\IBTrACS\IBTrACS.since1980.v04r01__landfall_eq0_only__wind_1min_ms__with_vmax_lifetime.xlsx"
LAND_SHEET_PREFIX = "landfall_eq"  # pick first sheet startswith this

# ✅ TC-HW events (用于(a)(b)(c) 以及图(d)的 tchw grids)
TC_HW_CSV = r"G:\LHY\[new]\TC-HW compound\TC-HW\compound_events_post7d_from_post30d__no_small_islands.csv"

# ✅ AHW events (仅用于图(d)的“总HW事件”补充)
AHW_CSV   = r"G:\LHY\[new]\TC-HW compound\AHW\alone_hw_sameyear_nonTCHW_after30d__mincols_no_small_islands.csv"

# ====================== Analysis params ======================
YEAR_MIN, YEAR_MAX = 1980, 2024
NEARSHORE_DEG = 0.0
ISLAND_AREA_MIN_KM2 = 50000

EA_CRS = "EPSG:6933"
GEO_CRS = "EPSG:4326"

# ----------------- panel d text position -----------------
D_TEXT_BLUE_POS  = (0.98, 0.08)   # blue: right-bottom
D_TEXT_RIGHT_POS = (0.22, 0.92)   # black: left-top
D_TEXT_FS = 16

# ====================== Style params ======================
LEGEND_FS = 16
TICK_FS   = 16
LABEL_FS  = 18
SPINE_LW  = 1.4

INK_GREEN  = "#0B3D2E"
RATIO_COLOR = "#1f77b4"    # blue

# ====================== FIGURE LAYOUT ======================
FIG_W, FIG_H = 9, 10
LEFT, RIGHT = 0.07, 0.98
BOTTOM, TOP = 0.06, 0.98
ROW_GAP_1 = 0.02
ROW_GAP_2= 0.05
COL_GAP = 0.13
H_A  = 0.52
H_B  = 0.23
H_CD = 0.28

# ====================== Helpers ======================
def wrap_lon_to_180(lon):
    lon = np.asarray(lon, dtype=float)
    return ((lon + 180.0) % 360.0) - 180.0

def _decode_bytes(x):
    if isinstance(x, (bytes, bytearray, np.bytes_)):
        try:
            return x.decode("utf-8", errors="ignore")
        except Exception:
            return x.decode("latin1", errors="ignore")
    return x

def degrees_to_meters_at_equator(deg):
    return float(deg) * 111_000.0

def build_nearshore_mask(nearshore_deg=NEARSHORE_DEG,
                         min_land_area_km2=ISLAND_AREA_MIN_KM2,
                         land_res='50m'):
    shp = shapereader.natural_earth(resolution=land_res, category='physical', name='land')
    reader = shapereader.Reader(shp)
    geoms = list(reader.geometries())

    proj_to_ea   = pyproj.Transformer.from_crs(GEO_CRS, EA_CRS, always_xy=True).transform
    proj_to_geo  = pyproj.Transformer.from_crs(EA_CRS, GEO_CRS, always_xy=True).transform

    large_polys_ea = []
    for g in geoms:
        g_ea = shp_transform(proj_to_ea, g)
        area_km2 = g_ea.area / 1e6
        if area_km2 >= float(min_land_area_km2):
            large_polys_ea.append(g_ea)

    if not large_polys_ea:
        raise RuntimeError("No land polygons passed the area threshold; increase or check data.")

    continents_ea = unary_union(large_polys_ea)
    buf_m = degrees_to_meters_at_equator(nearshore_deg)
    nearshore_ea = continents_ea.buffer(buf_m)
    nearshore_geo = shp_transform(proj_to_geo, nearshore_ea)
    return prep(nearshore_geo), nearshore_geo

def split_at_dateline(lons, lats, thresh=180.0):
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    if lons.size < 2:
        return []
    segs = []
    start = 0
    for i in range(1, lons.size):
        if np.abs(lons[i] - lons[i-1]) > thresh:
            if i-1 >= start:
                segs.append((lons[start:i], lats[start:i]))
            start = i
    if lons.size-1 >= start:
        segs.append((lons[start:], lats[start:]))
    return [s for s in segs if s[0].size >= 2]

# ====================== NEW: grid snap (0.25°) ======================
GRID_STEP_DEG = 0.25

def snap_to_grid(val, step=GRID_STEP_DEG):
    val = np.asarray(val, dtype=float)
    return np.round(val / step) * step

def snap_latlon_to_grid(lat, lon):
    lat_g = snap_to_grid(lat, GRID_STEP_DEG)
    lon_g = snap_to_grid(wrap_lon_to_180(lon), GRID_STEP_DEG)
    return lat_g, lon_g

# ====================== P-value -> stars ======================
def p_to_stars(p):
    if p is None or (not np.isfinite(p)):
        return ""
    p = float(p)
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""

def slope_with_sig(slope, p, fmt=".3g"):
    stars = p_to_stars(p)
    if slope is None or (not np.isfinite(slope)):
        return "Slope=nan"
    return f"Slope={format(float(slope), fmt)}{stars}"

def _fit_line_ci(x, y, alpha=0.05):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < 3:
        return None

    if not _HAVE_SCIPY:
        a, b = np.polyfit(x, y, 1)
        yhat = a*x + b
        return {"x": x, "yhat": yhat, "lo": np.full_like(yhat, np.nan), "hi": np.full_like(yhat, np.nan),
                "slope": float(a), "p": np.nan}

    res = stats.linregress(x, y)
    a = float(res.slope); b = float(res.intercept); p = float(res.pvalue)
    yhat = a*x + b

    n = x.size
    xbar = x.mean()
    sxx = np.sum((x - xbar)**2)
    if sxx <= 0:
        return {"x": x, "yhat": yhat, "lo": np.full_like(yhat, np.nan), "hi": np.full_like(yhat, np.nan),
                "slope": a, "p": p}

    resid = y - yhat
    s_err = np.sqrt(np.sum(resid**2) / (n - 2))
    tcrit = stats.t.ppf(1 - alpha/2, df=n-2)
    se_mean = s_err * np.sqrt(1/n + (x - xbar)**2 / sxx)
    lo = yhat - tcrit * se_mean
    hi = yhat + tcrit * se_mean

    return {"x": x, "yhat": yhat, "lo": lo, "hi": hi, "slope": a, "p": p}

# ====================== SS category (1-min, m/s) ======================
def ss_category_from_vmax_ms(v):
    if not np.isfinite(v):
        return None
    v = float(v)
    if v < 33.0:
        return "TD+TS"
    if v < 43.0:
        return "Cat1"
    if v < 50.0:
        return "Cat2"
    if v < 58.0:
        return "Cat3"
    if v < 70.0:
        return "Cat4"
    return "Cat5"

SS_ORDER = ["TD+TS","Cat1","Cat2","Cat3","Cat4","Cat5"]
SS_COLORS = {
    "TD+TS": "#01665e",
    "Cat1":  "#35978f",
    "Cat2":  "#c7eae5",
    "Cat3":  "#dfc27d",
    "Cat4":  "#8c510a",
    "Cat5":  "#543005",
}

# ====================== Style helpers ======================
def add_clean_basemap(ax):
    # 只显示到南纬60度，去掉南极区域
    ax.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())
    ax.coastlines(resolution='50m', linewidth=0.6, zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.3, zorder=3)

def _kill_cartopy_outline(ax):
    if hasattr(ax, "outline_patch") and ax.outline_patch is not None:
        ax.outline_patch.set_visible(False)
    if hasattr(ax, "spines") and isinstance(ax.spines, dict) and ("geo" in ax.spines):
        try:
            ax.spines["geo"].set_visible(False)
        except Exception:
            pass

def _hide_all_spines(ax):
    if hasattr(ax, "spines"):
        for sp in ax.spines.values():
            try:
                sp.set_visible(False)
            except Exception:
                pass

def add_rect_frame(ax):
    rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                     fill=False, linewidth=SPINE_LW, edgecolor="k",
                     zorder=5000, clip_on=False)
    ax.add_patch(rect)

def legend_text_only(ax, text, loc="upper left"):
    dummy = Line2D([], [], linestyle="None", marker=None, linewidth=0, alpha=0)
    leg = ax.legend([dummy], [text], loc=loc, frameon=False,
                    fontsize=LEGEND_FS, handlelength=0, handletextpad=0.0, borderaxespad=0.6)
    return leg

def apply_axes_style(ax, *, remove_grid=False, is_cartopy=False):
    ax.set_title("")
    ax.tick_params(labelsize=TICK_FS, width=SPINE_LW)
    if remove_grid:
        ax.grid(False)

    _hide_all_spines(ax)
    if is_cartopy:
        _kill_cartopy_outline(ax)

    leg = ax.get_legend()
    if leg is not None:
        leg.set_frame_on(False)
        for t in leg.get_texts():
            t.set_fontsize(LEGEND_FS)

    add_rect_frame(ax)

def set_label_font(ax):
    ax.set_xlabel(ax.get_xlabel(), fontsize=LABEL_FS)
    ax.set_ylabel(ax.get_ylabel(), fontsize=LABEL_FS)

# ====================== Load landfall Excel ======================
def load_landfall_excel(xlsx_path, sheet_prefix="landfall_eq"):
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"Landfall Excel not found: {xlsx_path}")

    sheets = pd.read_excel(xlsx_path, sheet_name=None, engine="openpyxl")
    land_sheet = None
    for sname in sheets:
        if str(sname).startswith(sheet_prefix):
            land_sheet = sname
            break
    if land_sheet is None:
        raise RuntimeError(f"No sheet starts with '{sheet_prefix}'. Found: {list(sheets.keys())}")

    df = sheets[land_sheet].copy()
    need = ["sid", "time_utc", "lat", "lon"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise KeyError(f"Landfall sheet '{land_sheet}' missing columns: {miss}")

    df["sid"] = df["sid"].astype(str)
    df["time_utc"] = pd.to_datetime(df["time_utc"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["sid", "time_utc", "lat", "lon"]).reset_index(drop=True)

    df["lon_180"] = wrap_lon_to_180(df["lon"].values)
    df["year"] = df["time_utc"].dt.year.astype(int)
    df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)].copy()
    return df

def load_total_landfall_tc_counts_by_year_no_mask(df_landfall):
    
    if df_landfall.empty:
        return pd.Series(dtype=float), set(), pd.DataFrame()

    idx = df_landfall.groupby("sid")["time_utc"].idxmin()
    df_first = df_landfall.loc[idx].copy()
    counts = df_first.groupby("year")["sid"].nunique()
    sid_set = set(df_first["sid"].astype(str).tolist())
    return counts, sid_set, df_first

# ====================== Load TC-HW events (csv/xlsx robust) ======================
def load_tchw_events_mincols(path):
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"TC-HW file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path, engine="openpyxl")
    else:
        df = pd.read_csv(path)

    need = ["sid","landfall_time_utc","landfall_lat","landfall_lon","grid_lat","grid_lon",
            "start_clip","end_clip","duration","intensity"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise KeyError(f"TC-HW file missing columns: {miss}\nAvailable: {list(df.columns)}")

    df["sid"] = df["sid"].astype(str)
    df["landfall_time_utc"] = pd.to_datetime(df["landfall_time_utc"], errors="coerce")
    df["start_clip"] = pd.to_datetime(df["start_clip"], errors="coerce")
    df["end_clip"] = pd.to_datetime(df["end_clip"], errors="coerce")

    for c in ["landfall_lat","landfall_lon","grid_lat","grid_lon","duration","intensity"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["sid","landfall_time_utc","grid_lat","grid_lon","start_clip"]).copy()
    df["year"] = df["landfall_time_utc"].dt.year.astype(int)

    df["grid_lon"] = wrap_lon_to_180(df["grid_lon"].values)
    df["landfall_lon"] = wrap_lon_to_180(df["landfall_lon"].values)

    df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)].copy()

    vmax_candidates = ["vmax_lifetime_1min", "vmax_lifetime_1min_ms", "vmax_lifetime_1min_m_s", "vmax_life_1min"]
    vmax_col = None
    for c in vmax_candidates:
        if c in df.columns:
            vmax_col = c
            break
    if vmax_col is None:
        raise KeyError(
            "在 TC-HW 文件中找不到生命周期最大风速列（例如 vmax_lifetime_1min）。"
            f"请检查该文件是否包含以下任一列：{vmax_candidates}"
        )

    df["vmax_lifetime_1min"] = pd.to_numeric(df[vmax_col], errors="coerce")
    df["rel_to_landfall"] = "post"
    return df

# ====================== NEW: Load AHW mincols CSV for panel(d) ======================
def load_ahw_mincols_csv(path):
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"AHW file not found: {path}")

    df = pd.read_csv(path)

    # grid cols
    glat_candidates = ["grid_lat","g_lat","lat_grid","lat"]
    glon_candidates = ["grid_lon","g_lon","lon_grid","lon"]

    glat = None
    glon = None
    for c in glat_candidates:
        if c in df.columns:
            glat = c; break
    for c in glon_candidates:
        if c in df.columns:
            glon = c; break
    if glat is None or glon is None:
        raise KeyError(f"AHW file must contain grid lat/lon columns. Found columns: {list(df.columns)}")

    df["grid_lat"] = pd.to_numeric(df[glat], errors="coerce")
    df["grid_lon"] = wrap_lon_to_180(pd.to_numeric(df[glon], errors="coerce").values)

    # year
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    else:
        time_candidates = ["landfall_time_utc","start_clip","hw_start","start_time","time_utc","date","start"]
        tcol = None
        for c in time_candidates:
            if c in df.columns:
                tcol = c; break
        if tcol is None:
            raise KeyError(
                "AHW 文件既没有 year 列，也找不到可解析的时间列用于推断年份。"
                f"请在文件中提供 year 或时间列（如 start_clip/hw_start）。现有列：{list(df.columns)}"
            )
        tt = pd.to_datetime(df[tcol], errors="coerce")
        df["year"] = tt.dt.year.astype("Int64")

    df = df.dropna(subset=["year","grid_lat","grid_lon"]).copy()
    df["year"] = df["year"].astype(int)
    df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)].copy()

    # 对齐到0.25°网格，避免浮点误差
    df["grid_lat"], df["grid_lon"] = snap_latlon_to_grid(df["grid_lat"].values, df["grid_lon"].values)
    return df[["year","grid_lat","grid_lon"]].copy()

# ====================== Load tracks from ORIGINAL IBTrACS NC (robust) ======================
def _pick_var(ds, candidates):
    for c in candidates:
        if c in ds.variables:
            return c
    return None

def _get_time_1d_from_iso_char(iso_char_2d):
    arr = iso_char_2d
    if arr.dtype.kind in ["S", "U", "O"]:
        s = np.asarray(arr).astype(str)
        return pd.to_datetime(s, errors="coerce")
    a = np.asarray(arr)
    if a.ndim == 2:
        s = np.array([b"".join(row).decode("utf-8", errors="ignore").strip() for row in a], dtype=object)
        return pd.to_datetime(s, errors="coerce")
    if a.ndim == 3:
        raise ValueError("Expected per-storm iso_time slice to be 2D. You passed 3D.")
    return pd.to_datetime(a, errors="coerce")

def load_tc_tracks_from_original_ibtracs_nc(nc_path, allowed_sids=None):
    if not os.path.exists(nc_path):
        raise FileNotFoundError(f"IBTrACS NC not found: {nc_path}")

    ds = xr.open_dataset(nc_path, decode_times=True)

    sid_var = _pick_var(ds, ["sid", "SID"])
    if sid_var is None:
        ds.close()
        raise KeyError("Cannot find 'sid' in IBTrACS NC.")

    sid_vals = ds[sid_var].values
    sid_list = [str(_decode_bytes(x)) for x in sid_vals]
    sid_to_i = {s: i for i, s in enumerate(sid_list) if s not in {}}

    allowed = None
    if allowed_sids is not None:
        allowed = set(str(s) for s in allowed_sids)

    lat_var = _pick_var(ds, ["lat", "latitude", "usa_lat"])
    lon_var = _pick_var(ds, ["lon", "longitude", "usa_lon"])
    if lat_var is None or lon_var is None:
        ds.close()
        raise KeyError("Cannot find lat/lon variables in IBTrACS NC (expected 'lat' and 'lon').")

    time_var = _pick_var(ds, ["time"])
    iso_var  = _pick_var(ds, ["iso_time", "ISO_TIME"])

    tracks = []
    for sid, i in sid_to_i.items():
        if allowed is not None and sid not in allowed:
            continue

        lat = ds[lat_var].isel(storm=i).values
        lon = ds[lon_var].isel(storm=i).values
        lon = wrap_lon_to_180(lon)

        if time_var is not None and "storm" in ds[time_var].dims:
            t = ds[time_var].isel(storm=i).values
            t = pd.to_datetime(t, errors="coerce")
        elif iso_var is not None and "storm" in ds[iso_var].dims:
            iso_slice = ds[iso_var].isel(storm=i).values
            t = _get_time_1d_from_iso_char(iso_slice)
        else:
            t = pd.to_datetime(np.arange(np.asarray(lat).size), errors="coerce")

        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        t   = pd.DatetimeIndex(t)

        m = np.isfinite(lat) & np.isfinite(lon)
        if m.sum() < 2:
            continue

        tracks.append({
            "sid": sid,
            "time": t[m],
            "lat": lat[m],
            "lon": lon[m],
        })

    ds.close()
    return tracks

# ====================== Panels ======================
def plot_tracks_by_ss(tracks, sid2cat, ax, pie_xy=(0.15, 0.10), pie_size=0.50, draw_legend=True):
    add_clean_basemap(ax)

    used_cats = set()
    cat_counts = {k: 0 for k in SS_ORDER}

    for tr in tracks:
        sid = str(tr["sid"])
        cat = sid2cat.get(sid, None)
        if cat is None:
            continue

        cat_counts[cat] += 1
        used_cats.add(cat)

        col = SS_COLORS.get(cat, "#666666")
        segs = split_at_dateline(tr["lon"], tr["lat"], thresh=180.0)
        if not segs:
            ax.plot(tr["lon"], tr["lat"], color=col, linewidth=0.7, alpha=0.85,
                    transform=ccrs.PlateCarree(), zorder=2)
        else:
            for lon_s, lat_s in segs:
                ax.plot(lon_s, lat_s, color=col, linewidth=0.7, alpha=0.85,
                        transform=ccrs.PlateCarree(), zorder=2)

    if draw_legend:
        handles, labels = [], []
        for k in SS_ORDER:
            if (k in used_cats) and (cat_counts.get(k, 0) > 0):
                handles.append(Line2D([], [], color=SS_COLORS.get(k, "#666666"), lw=4.2))
                labels.append(k)
        ax.legend(handles, labels, loc="lower left", fontsize=LEGEND_FS, frameon=False,
                  ncol=len(labels), handlelength=1.2, handletextpad=0.5, columnspacing=1.2)

    total = sum(cat_counts.values())
    if total > 0:
        pie_labels = [k for k in SS_ORDER if cat_counts[k] > 0]
        pie_sizes  = [cat_counts[k] / total for k in pie_labels]
        pie_colors = [SS_COLORS[k] for k in pie_labels]

        left, bottom = pie_xy
        axins = ax.inset_axes([left, bottom, pie_size, pie_size], transform=ax.transAxes)
        axins.pie(
            pie_sizes,
            colors=pie_colors,
            startangle=90,
            counterclock=False,
            autopct=lambda p: f"{p:.0f}%" if p >= 3 else "",
            pctdistance=0.72,
            textprops={"fontsize": 16},
            wedgeprops={"edgecolor": "k", "linewidth": 0.3},
        )
        axins.set_aspect("equal")
        axins.set_xticks([])
        axins.set_yticks([])
        axins.set_facecolor("none")

    ax.set_title("")
    return used_cats, cat_counts

def plot_fig4_lifetime_ss_stacked(df_post, sid2cat, ax, total_tc_counts_yearly=None, draw_legend=True):
    if df_post.empty:
        return

    sid_year = df_post.groupby("sid")["year"].agg(
        lambda s: int(pd.Series(s).mode().iloc[0]) if len(pd.Series(s).mode()) else int(s.iloc[0])
    )

    years = np.arange(YEAR_MIN, YEAR_MAX + 1).astype(int)
    mat = pd.DataFrame(0, index=years, columns=SS_ORDER, dtype=int)

    for sid, yy in sid_year.items():
        cat = sid2cat.get(str(sid), None)
        if cat is None:
            continue
        if yy in mat.index and cat in mat.columns:
            mat.loc[int(yy), cat] += 1

    tc_hw_series = mat.sum(axis=1).values.astype(float)

    # ===================== ✅ NEW: Cat3-5 count trend text ONLY (no line/CI) =====================
    cats_c35 = ["Cat3", "Cat4", "Cat5"]
    c35_series = mat[cats_c35].sum(axis=1).values.astype(float)
    fit_c35 = _fit_line_ci(years.astype(float), c35_series.astype(float), alpha=0.05)
    if fit_c35 is not None:
        txt_c35 = "Cat3-5: " + slope_with_sig(fit_c35["slope"], fit_c35["p"], fmt=".2g")
        ax.text(
            0.15, 0.29, txt_c35,
            transform=ax.transAxes,
            ha="center", va="center",
           fontsize=16, color="k", zorder=7000
       )
    # =================================================================================================

    total_series = None
    if total_tc_counts_yearly is not None:
        ax2 = ax.twinx()
        total_series = np.array([float(total_tc_counts_yearly.get(int(yy), 0.0)) for yy in years], dtype=float)

        ax.set_facecolor("none")
        ax2.set_zorder(0)
        ax.set_zorder(1)

        ax2.step(years, total_series, where="mid", lw=1.2, color="0.25", alpha=0.65)
        ax2.set_ylabel("Number of landfall TC")
        ax2.set_xlim(YEAR_MIN - 1.5, YEAR_MAX + 1.5)
        ax2.set_ylim(0, 70)
        ax2.tick_params(labelsize=TICK_FS, width=SPINE_LW, colors="0.1")
        ax2.yaxis.label.set_color("0.1")
        ax2.grid(False)

    if (total_series is not None) and (np.any(total_series > 0)):
        ax3 = ax.twinx()
        ax3.spines["right"].set_position(("outward", 46))
        ax3.set_zorder(6)

        ratio = np.full_like(tc_hw_series, np.nan, dtype=float)
        ok = total_series > 0
        ratio[ok] = tc_hw_series[ok] / total_series[ok]

        ax3.plot(years, ratio, lw=1.6, color=RATIO_COLOR, alpha=1)

        rmax = np.nanmax(ratio) if np.any(np.isfinite(ratio)) else np.nan
        top = min(1.0, max(0.2, rmax * 1.2)) if (np.isfinite(rmax) and rmax > 0) else 1.0
        ax3.set_ylim(0, top)
        ax3.set_xlim(YEAR_MIN - 1.5, YEAR_MAX + 1.5)

        ax3.set_ylabel("Ratio (TC followed by HW / TC)")
        ax3.tick_params(labelsize=TICK_FS, width=SPINE_LW, colors=RATIO_COLOR)
        ax3.yaxis.label.set_color(RATIO_COLOR)
        ax3.spines["right"].set_color(RATIO_COLOR)
        ax3.grid(False)

        fit_r = _fit_line_ci(years.astype(float), ratio.astype(float), alpha=0.05)
        if fit_r is not None:
            txt = "Ratio: " + slope_with_sig(fit_r["slope"], fit_r["p"], fmt=".2g")
            ax.text(0.29, 0.49, txt,
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=16, color=RATIO_COLOR, zorder=7000)

    bottom = np.zeros(len(years), dtype=float)
    for cat in SS_ORDER:
        vals = mat[cat].values.astype(float)
        ax.bar(years, vals, bottom=bottom, width=0.85, label=cat,
               color=SS_COLORS.get(cat, None), zorder=5)
        bottom += vals

    ax.set_xlabel("")
    ax.set_ylabel("Number of TCs \nfollowed by HWs")
    ax.set_xlim(YEAR_MIN - 1.5, YEAR_MAX + 1.5)

    ax.set_ylim(0, 30)

    if draw_legend:
        labels = [cat for cat in SS_ORDER if mat[cat].sum() > 0]
        handles = [Line2D([], [], color=SS_COLORS[cat], lw=4.2) for cat in labels]
        ax.legend(handles, labels, ncol=len(labels), fontsize=LEGEND_FS, frameon=False, loc="upper left",
                  handlelength=1.2, handletextpad=0.5, columnspacing=1.2)

    ax.set_title("")

# ====================== ✅ UPDATED panel (c) style ONLY ======================
def plot_fig3_post_events_trend(df_post, ax):
    years = np.arange(YEAR_MIN, YEAR_MAX + 1).astype(int)
    y = df_post.groupby("year").size()
    y = np.array([float(y.get(yy, 0.0)) for yy in years], dtype=float)

    x = years.astype(float)
    fit = _fit_line_ci(x, y, alpha=0.05)

    # NEW: 2000–2024 fit
    mask_recent = (x >= 2000) & (x <= 2024) & np.isfinite(y)
    fit_recent = _fit_line_ci(x[mask_recent], y[mask_recent], alpha=0.05)

    LW_SERIES = 1.2
    LW_TREND  = 2.0
    MS        = 3.8

    FULL_COLOR = "k"
    RECENT_COLOR = "#a50f15"

    # original yearly series
    ax.plot(x, y, color=FULL_COLOR, linestyle="--",
            marker="s", markersize=MS, linewidth=LW_SERIES, zorder=3)

    # full-period trend + CI
    if fit is not None:
        order = np.argsort(fit["x"])
        ax.plot(fit["x"][order], fit["yhat"][order], color=FULL_COLOR, linestyle="-",
                lw=LW_TREND, zorder=4)
        if np.all(np.isfinite(fit["lo"])) and np.all(np.isfinite(fit["hi"])):
            ax.fill_between(fit["x"][order], fit["lo"][order], fit["hi"][order],
                            alpha=0.22, linewidth=0, color=FULL_COLOR, zorder=2)

        txt_full = slope_with_sig(fit["slope"], fit["p"], fmt=".3g")
        ax.text(0.85, 0.95, txt_full,
                transform=ax.transAxes, ha="right", va="top",
                fontsize=16, color=FULL_COLOR)

    # NEW: 2000–2024 trend + CI
    if fit_recent is not None:
        order_r = np.argsort(fit_recent["x"])
        ax.plot(fit_recent["x"][order_r], fit_recent["yhat"][order_r],
                color=RECENT_COLOR, linestyle="-", lw=LW_TREND, zorder=5)
        if np.all(np.isfinite(fit_recent["lo"])) and np.all(np.isfinite(fit_recent["hi"])):
            ax.fill_between(fit_recent["x"][order_r], fit_recent["lo"][order_r], fit_recent["hi"][order_r],
                            alpha=0.18, linewidth=0, color=RECENT_COLOR, zorder=1.5)

        txt_recent = "2000-2024: " + slope_with_sig(fit_recent["slope"], fit_recent["p"], fmt=".3g")
        ax.text(0.85, 0.85, txt_recent,
                transform=ax.transAxes, ha="right", va="top",
                fontsize=16, color=RECENT_COLOR)

    ax.set_xlabel("")
    ax.set_ylabel("Number of\nTC-HW events")
    ax.set_ylim(bottom=0)
    ax.set_title("")

# ====================== ✅ UPDATED panel (d) — NO HW_NC, use (TC-HW + AHW) CSVs ======================
def plot_fig5_compound_on_active_grids_no_hw_nc(df_landfall_allrows, df_tchw_all, df_ahw_all, ax, prepared_mask):
    
    # ---------- landfall -> active grids by year ----------
    df_lf = df_landfall_allrows.copy()

    # 对齐 landfall 点到 0.25° 网格
    lf_lat_g, lf_lon_g = snap_latlon_to_grid(df_lf["lat"].values, df_lf["lon_180"].values)
    df_lf["g_lat"] = lf_lat_g
    df_lf["g_lon"] = lf_lon_g

    # 保持你之前的“同一sid+同一grid只保留一次（最早）”
    df_lf = (df_lf.sort_values("time_utc")
                 .drop_duplicates(subset=["sid", "g_lat", "g_lon"], keep="first")
                 .reset_index(drop=True))

    active_grids_by_year = {y: set() for y in range(YEAR_MIN, YEAR_MAX + 1)}
    for r in df_lf.itertuples(index=False):
        y = int(r.year)
        if y < YEAR_MIN or y > YEAR_MAX:
            continue
        active_grids_by_year[y].add((float(r.g_lat), float(r.g_lon)))

    # ---------- TC-HW grids by year ----------
    df_t = df_tchw_all.copy()
    df_t = df_t[(df_t["year"] >= YEAR_MIN) & (df_t["year"] <= YEAR_MAX)].copy()
    # 对齐到0.25°，避免浮点误差
    df_t["g_lat"], df_t["g_lon"] = snap_latlon_to_grid(df_t["grid_lat"].values, df_t["grid_lon"].values)

    tchw_grids_by_year = {}
    for y, sub in df_t.groupby("year"):
        tchw_grids_by_year[int(y)] = set(zip(sub["g_lat"].astype(float).values, sub["g_lon"].astype(float).values))

    # ---------- total HW events (TC-HW + AHW) for BLUE numerator ----------
    # TC-HW events as HW events:
    df_hw1 = df_t[["year","g_lat","g_lon"]].copy()
    # AHW events:
    df_a = df_ahw_all.copy()
    df_a = df_a[(df_a["year"] >= YEAR_MIN) & (df_a["year"] <= YEAR_MAX)].copy()
    df_hw2 = df_a.rename(columns={"grid_lat":"g_lat","grid_lon":"g_lon"})[["year","g_lat","g_lon"]].copy()

    df_hw_all = pd.concat([df_hw1, df_hw2], axis=0, ignore_index=True)
    df_hw_all = df_hw_all.dropna(subset=["year","g_lat","g_lon"]).copy()
    df_hw_all["year"] = df_hw_all["year"].astype(int)
    df_hw_all["g_lat"] = df_hw_all["g_lat"].astype(float)
    df_hw_all["g_lon"] = df_hw_all["g_lon"].astype(float)

    # 预先做 (year, grid) -> count
    hw_count_map = df_hw_all.groupby(["year","g_lat","g_lon"]).size().to_dict()

    # ---------- compute y1 (blue) & y2 (black) ----------
    years = np.arange(YEAR_MIN, YEAR_MAX + 1, dtype=int)
    y1 = np.full(years.shape, np.nan, dtype=float)  # blue
    y2 = np.full(years.shape, np.nan, dtype=float)  # black

    for i, y in enumerate(years):
        active = active_grids_by_year.get(int(y), set())
        n_active = len(active)
        if n_active == 0:
            continue

        tchw_grids = tchw_grids_by_year.get(int(y), set())
        active_with_tchw = set(active) & set(tchw_grids)

        # black: fraction
        y2[i] = len(active_with_tchw) / float(n_active)

        # blue: mean HW events per (active grids that have TC-HW)
        denom_blue = len(active_with_tchw)
        if denom_blue == 0:
            y1[i] = np.nan
        else:
            total_hw_events_on_active_tchw = 0.0
            for (la, lo) in active_with_tchw:
                total_hw_events_on_active_tchw += float(hw_count_map.get((int(y), float(la), float(lo)), 0.0))
            y1[i] = total_hw_events_on_active_tchw / float(denom_blue)

    # ---------- plot (keep style unchanged) ----------
    x = years.astype(float)
    fit1 = _fit_line_ci(x, y1, alpha=0.05)
    fit2 = _fit_line_ci(x, y2, alpha=0.05)

    C1 = RATIO_COLOR
    C2 = "k"

    LW_SERIES = 1.2
    LW_TREND  = 2.0
    MS        = 3.8

    ax.plot(years, y1, color=C1, linestyle="--",
            marker="o", markersize=MS, linewidth=LW_SERIES, zorder=3)

    if fit1 is not None:
        order = np.argsort(fit1["x"])
        ax.plot(fit1["x"][order], fit1["yhat"][order], color=C1, linestyle="-",
                linewidth=LW_TREND, zorder=4)
        if np.all(np.isfinite(fit1["lo"])) and np.all(np.isfinite(fit1["hi"])):
            ax.fill_between(fit1["x"][order], fit1["lo"][order], fit1["hi"][order],
                            color=C1, alpha=0.22, linewidth=0, zorder=2)

    ax.set_xlabel("")
    ax.set_ylabel("Number of HWs\nper area with TC-HW", color=C1)
    ax.tick_params(axis="y", colors=C1)
    ax.yaxis.label.set_color(C1)

    ax2 = ax.twinx()
    ax2.plot(years, y2, color=C2, linestyle="--",
             marker="s", markersize=MS, linewidth=LW_SERIES, zorder=3)

    if fit2 is not None:
        order = np.argsort(fit2["x"])
        ax2.plot(fit2["x"][order], fit2["yhat"][order], color=C2, linestyle="-",
                 linewidth=LW_TREND, zorder=4)
        if np.all(np.isfinite(fit2["lo"])) and np.all(np.isfinite(fit2["hi"])):
            ax2.fill_between(fit2["x"][order], fit2["lo"][order], fit2["hi"][order],
                             color=C2, alpha=0.18, linewidth=0, zorder=2)

    ax2.set_ylabel("Fraction of areas\nwith TC-HW", color=C2)
    ax2.tick_params(axis="y", colors=C2)
    ax2.yaxis.label.set_color(C2)
    ax2.spines["right"].set_color(C2)

    ax.tick_params(axis="x", labelsize=TICK_FS, width=SPINE_LW)
    ax.tick_params(axis="y", labelsize=TICK_FS, width=SPINE_LW, colors=C1)
    ax2.tick_params(axis="y", labelsize=TICK_FS, width=SPINE_LW, colors=C2)

    ax.set_xlim(YEAR_MIN - 1.5, YEAR_MAX + 1.5)
    ax.set_title("")

    s1 = fit1["slope"] if fit1 is not None else np.nan
    p1 = fit1["p"]     if fit1 is not None else np.nan
    s2 = fit2["slope"] if fit2 is not None else np.nan
    p2 = fit2["p"]     if fit2 is not None else np.nan

    txt1 = slope_with_sig(s1, p1, fmt=".2g")
    txt2 = slope_with_sig(s2, p2, fmt=".2g")

    ax.text(D_TEXT_BLUE_POS[0], D_TEXT_BLUE_POS[1], txt1,
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=D_TEXT_FS, color=C1, zorder=7000)

    ax.text(D_TEXT_RIGHT_POS[0], D_TEXT_RIGHT_POS[1], txt2,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=D_TEXT_FS, color=C2, zorder=7000)

# ====================== Manual layout ======================
def compute_axes_positions():
    W = RIGHT - LEFT
    H_total = TOP - BOTTOM

    H_usable = H_total - 4 * ROW_GAP_2
    hsum = H_A + H_B + H_CD

    ha  = H_usable * (H_A  / hsum)
    hb  = H_usable * (H_B  / hsum)
    hcd = H_usable * (H_CD / hsum)

    y_cd = BOTTOM
    y_b  = y_cd + hcd + ROW_GAP_2
    y_a  = y_b  + hb  + ROW_GAP_1

    w_half = (W - COL_GAP) / 2.0
    x_c = LEFT
    x_d = LEFT + w_half + COL_GAP

    pos_a  = [LEFT, y_a, W, ha]
    pos_b  = [LEFT, y_b, W, hb]
    pos_c  = [x_c,  y_cd, w_half, hcd]
    pos_d  = [x_d,  y_cd, w_half, hcd]
    return pos_a, pos_b, pos_c, pos_d

def add_shared_ss_legend(fig, used_cats, pos_a, pos_b):
    gap_bottom = pos_b[1] + pos_b[3]
    gap_top    = pos_a[1]
    y_center   = (gap_bottom + gap_top) / 2.0
    x_center   = (pos_a[0] + pos_a[0] + pos_a[2]) / 2.0

    labels = [k for k in SS_ORDER if (k in used_cats)]
    handles = [Line2D([], [], color=SS_COLORS[k], lw=4.2) for k in labels]

    fig.legend(
        handles, labels,
        loc="center",
        bbox_to_anchor=(x_center, y_center),
        ncol=max(1, len(labels)),   # 改成一行
        frameon=False,
        fontsize=LEGEND_FS,
        handlelength=1.2,          # 线段更短
        columnspacing=1.0,
        handletextpad=0.5,
        borderaxespad=0.0
    )

# ====================== Main ======================
def main():
    plt.rcParams.update({
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": LABEL_FS,
        "xtick.labelsize": TICK_FS,
        "ytick.labelsize": TICK_FS,
        "legend.fontsize": LEGEND_FS,
    })

    print("Building land/nearshore mask (kept for compatibility; panel d still does NOT filter by mask) ...")
    prepared_mask, _ = build_nearshore_mask(
        nearshore_deg=NEARSHORE_DEG,
        min_land_area_km2=ISLAND_AREA_MIN_KM2,
        land_res="50m"
    )

    print("Loading landfall Excel (multi points) for total TC count ...")
    df_lf_all = load_landfall_excel(LANDFALL_XLSX, sheet_prefix=LAND_SHEET_PREFIX)

    print("Computing yearly total landfall TC count from landfall Excel (NO mask filtering; yearly unique sid) ...")
    total_tc_counts_yearly, _, _ = load_total_landfall_tc_counts_by_year_no_mask(df_lf_all)

    print("Loading TC-HW events (csv) + lifetime vmax ...")
    df_evt_all = load_tchw_events_mincols(TC_HW_CSV)
    df_post = df_evt_all[df_evt_all["rel_to_landfall"] == "post"].copy()

    post_sids = sorted(df_post["sid"].astype(str).unique())
    print(f"POST storms (unique sids) in TC-HW file: {len(post_sids)}")
    print(f"POST event rows in TC-HW file: {len(df_post)}")

    sid_vmax = df_post.groupby("sid")["vmax_lifetime_1min"].max()
    sid2cat = {str(sid): ss_category_from_vmax_ms(v) for sid, v in sid_vmax.items()}

    print("Loading tracks from ORIGINAL IBTrACS NC (allowed_sids = storms in df_post) ...")
    tracks = load_tc_tracks_from_original_ibtracs_nc(IBTRACS_NC, allowed_sids=post_sids)

    print("Loading AHW mincols CSV for panel(d) ...")
    df_ahw = load_ahw_mincols_csv(AHW_CSV)

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=800)

    pos_a, pos_b, pos_c, pos_d = compute_axes_positions()

    ax_a = fig.add_axes(pos_a, projection=ccrs.PlateCarree())
    ax_b = fig.add_axes(pos_b)
    ax_c = fig.add_axes(pos_c)
    ax_d = fig.add_axes(pos_d)

    used_cats, _ = plot_tracks_by_ss(
        tracks, sid2cat, ax=ax_a,
        pie_xy=(-0.13, 0.09), pie_size=0.52,
        draw_legend=False
    )

    plot_fig4_lifetime_ss_stacked(
        df_post, sid2cat, ax=ax_b,
        total_tc_counts_yearly=total_tc_counts_yearly,
        draw_legend=False
    )
    add_shared_ss_legend(fig, used_cats, pos_a, pos_b)

    plot_fig3_post_events_trend(df_post, ax=ax_c)

    # ✅ panel(d) updated: use (TC-HW + AHW) instead of HW_NC
    plot_fig5_compound_on_active_grids_no_hw_nc(
        df_lf_all, df_evt_all, df_ahw,
        ax=ax_d, prepared_mask=prepared_mask
    )

    ax_a.text(0.01, 0.97, "(a)", transform=ax_a.transAxes, ha="left", va="top",
              fontsize=20, fontweight="bold", zorder=6000)
    ax_b.text(0.01, 0.96, "(b)", transform=ax_b.transAxes, ha="left", va="top",
              fontsize=20, fontweight="bold", zorder=6000)
    ax_c.text(0.01, 0.96, "(c)", transform=ax_c.transAxes, ha="left", va="top",
              fontsize=20, fontweight="bold", zorder=6000)
    ax_d.text(0.01, 0.96, "(d)", transform=ax_d.transAxes, ha="left", va="top",
              fontsize=20, fontweight="bold", zorder=6000)

    apply_axes_style(ax_a, remove_grid=False, is_cartopy=True)
    apply_axes_style(ax_b, remove_grid=False, is_cartopy=False)
    apply_axes_style(ax_c, remove_grid=True,  is_cartopy=False)
    apply_axes_style(ax_d, remove_grid=True,  is_cartopy=False)

    set_label_font(ax_b)
    set_label_font(ax_c)
    set_label_font(ax_d)

    plt.show()

if __name__ == "__main__":
    main()