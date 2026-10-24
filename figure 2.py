import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, linregress, t

# Fonts
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Arial']

# =========================
# 0) User settings
# =========================
XLSX_PATH = r"G:\LHY\[new]\TC-HW compound\TC-HW\compound_events_post7d_from_post30d__no_small_islands__with_Pdam.xlsx"
ONLY_POST = True

PDAM_THR_LINE = 0.9   #Pdam>=0.9

YEAR_MIN, YEAR_MAX = 1980, 2024
CI_LEVEL = 0.95

DECADE_TICK = 10
HIDE_LAST_YEAR_LABEL = True

WINDOW_DAYS = 7
WINDOW_H = WINDOW_DAYS * 24.0

# For panels (e)(f)
PDAM_MIN_ECDF = 0.9   #Pdam>=0.9
EARLY_Y0, EARLY_Y1 = 1980, 1999
LATE_Y0,  LATE_Y1  = 2000, 2024

TAIL_MAX_H = 168.0
BOOTSTRAP_N = 1000
CI_ALPHA = 0.05
RNG_SEED = 12345

SID_CANDIDATES = ["sid", "SID", "storm_id", "event_id", "EventID"]
WIND80_COL_CANDIDATES = [
    "vmax_landfall_day_80m",
    "vmax_landfall_day_80m_ms",
    "vmax_lfday_80m",
    "vmax_lfday_80m_ms",
    "wind80_landfall_day",
]

# =========================
# Added controls
# =========================
EF_XTICK_INTERVAL = 24
ROW_HSPACE = 0.222

# =========================
# Style controls
# =========================
PANEL_LABEL_FONTSIZE = 22
ANNO_FONTSIZE = 16
FRAME_LW = 1.5

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

BLUE = "#2166ac"
ALPHA_CI_AB = 0.18
ALPHA_CI_F  = 0.12
ALPHA_ZERO  = 0.30

LW_TREND = 2.2
LW_PROC  = 1.6
LW_ECDF  = 2.2
LW_DELTA = 2.4

LABEL_POS = {
    "(a)": (0.03, 0.94),
    "(b)": (0.03, 0.94),
    "(c)": (0.88, 0.94),
    "(d)": (0.03, 0.94),
    "(e)": (0.03, 0.94),
    "(f)": (0.03, 0.94),
}

# =========================
# Helpers
# =========================
def _read_excel_first_sheet(xlsx_path: str) -> pd.DataFrame:
    sheets = pd.read_excel(xlsx_path, sheet_name=None, engine="openpyxl")
    if not sheets:
        raise RuntimeError(f"No sheets found in Excel: {xlsx_path}")
    first = list(sheets.keys())[0]
    return sheets[first].copy()

def _pick_first_existing(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

def _infer_year_series(df_: pd.DataFrame) -> pd.Series:
    if "year" in df_.columns:
        return pd.to_numeric(df_["year"], errors="coerce").astype("Int64")
    candidates = [
        "landfall_time_utc", "landfall_day", "landfall_date", "landfall_time",
        "start_date", "event_start", "hw_start", "tc_landfall_time",
        "start_clip", "start"
    ]
    for c in candidates:
        if c in df_.columns:
            dt = pd.to_datetime(df_[c], errors="coerce")
            if dt.notna().any():
                return dt.dt.year.astype("Int64")
    raise ValueError("Need a year column or a parsable datetime column (e.g., landfall_time_utc).")

def set_axes_frame_width(ax, lw=FRAME_LW):
    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_linewidth(lw)

def add_panel_label(ax, letter, fontsize=PANEL_LABEL_FONTSIZE):
    x, y = LABEL_POS.get(letter, (0.03, 0.94))
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold", va="top", ha="left")

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

def _annotate_slope(ax, slope, pval, unit_text="", xy=(0.43, 0.93)):
    star = p_to_stars(pval)
    if np.isfinite(slope):
        txt = f"Slope={slope:.2g}{unit_text} {star}"
    else:
        txt = "Slope=nan"
    ax.text(xy[0], xy[1], txt, transform=ax.transAxes,
            ha="left", va="top", fontsize=ANNO_FONTSIZE)

def _linreg_ci_band(x, y, x_grid, ci_level=0.95):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]

    if x.size < 3:
        return (np.full_like(x_grid, np.nan, float),
                np.full_like(x_grid, np.nan, float),
                np.full_like(x_grid, np.nan, float),
                np.nan, np.nan, np.nan)

    lr = linregress(x, y)
    b = float(lr.slope)
    a = float(lr.intercept)
    p = float(lr.pvalue)

    y_hat = a + b * x
    n = x.size
    dof = max(n - 2, 1)

    s_err = np.sqrt(np.sum((y - y_hat) ** 2) / dof)
    x_bar = np.mean(x)
    Sxx = np.sum((x - x_bar) ** 2)

    alpha = 1.0 - float(ci_level)
    tcrit = t.ppf(1.0 - alpha / 2.0, dof)

    xg = np.asarray(x_grid, float)
    y_hat_g = a + b * xg

    if Sxx <= 0:
        se_mean = s_err * np.sqrt(1.0 / n) * np.ones_like(xg)
    else:
        se_mean = s_err * np.sqrt(1.0 / n + (xg - x_bar) ** 2 / Sxx)

    ci_low = y_hat_g - tcrit * se_mean
    ci_high = y_hat_g + tcrit * se_mean
    return y_hat_g, ci_low, ci_high, b, a, p

def _yearly_box_mean(df_in: pd.DataFrame, year_col: str, value_col: str):
    years_sorted = np.sort(df_in[year_col].unique())
    data_by_year = [df_in.loc[df_in[year_col] == y, value_col].values.astype(float) for y in years_sorted]
    means_by_year = np.array([np.nanmean(v) if len(v) else np.nan for v in data_by_year], dtype=float)
    return years_sorted, data_by_year, means_by_year

def _set_decade_ticks_only(ax, years_sorted, tick_every=10, hide_last_year_label=True):
    years_sorted = np.asarray(years_sorted, dtype=int)
    pos = (np.arange(len(years_sorted)) + 1).astype(float)

    tick_years = [y for y in years_sorted if (y % tick_every == 0)]
    if hide_last_year_label and len(years_sorted) > 0:
        lasty = int(years_sorted[-1])
        tick_years = [y for y in tick_years if y != lasty]

    year2pos = {int(y): float(p) for y, p in zip(years_sorted, pos)}
    xticks = [year2pos[y] for y in tick_years if y in year2pos]
    xlabels = [str(int(y)) for y in tick_years if y in year2pos]

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=0)

    ax.minorticks_off()
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.tick_params(axis="x", which="major", length=5)

def classify_ss_from_ms(vmax_ms: float) -> float:
    if not np.isfinite(vmax_ms):
        return np.nan
    if vmax_ms < 33:
        return 0.0
    elif vmax_ms < 43:
        return 1.0
    elif vmax_ms < 50:
        return 2.0
    elif vmax_ms < 58:
        return 3.0
    elif vmax_ms < 70:
        return 4.0
    else:
        return 5.0

def ecdf_xy(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([]), np.array([])
    x = np.sort(x)
    y = np.arange(1, x.size + 1) / x.size
    return x, y

def ecdf_on_grid(samples, grid):
    s = np.asarray(samples, float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return np.full_like(grid, np.nan, dtype=float)
    s = np.sort(s)
    cnt = np.searchsorted(s, grid, side="right")
    return cnt / float(s.size)

def bootstrap_delta_ecdf(vE, vL, grid, B=1000, alpha=0.05, seed=12345):
    vE = np.asarray(vE, float); vE = vE[np.isfinite(vE)]
    vL = np.asarray(vL, float); vL = vL[np.isfinite(vL)]

    d_hat = ecdf_on_grid(vL, grid) - ecdf_on_grid(vE, grid)

    if vE.size < 5 or vL.size < 5:
        lo = np.full_like(grid, np.nan, dtype=float)
        hi = np.full_like(grid, np.nan, dtype=float)
        return d_hat, lo, hi

    rng = np.random.default_rng(seed)
    nE, nL = vE.size, vL.size

    d_boot = np.empty((B, grid.size), dtype=float)
    for b in range(B):
        sE = vE[rng.integers(0, nE, size=nE)]
        sL = vL[rng.integers(0, nL, size=nL)]
        d_boot[b, :] = ecdf_on_grid(sL, grid) - ecdf_on_grid(sE, grid)

    lo = np.nanpercentile(d_boot, 100.0 * (alpha / 2.0), axis=0)
    hi = np.nanpercentile(d_boot, 100.0 * (1.0 - alpha / 2.0), axis=0)
    return d_hat, lo, hi

def find_zero_crossings(x, y):
   
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    crosses = []
    if x.size < 2:
        return crosses

    for i in range(len(x) - 1):
        x0, x1 = x[i], x[i + 1]
        y0, y1 = y[i], y[i + 1]

        if y0 == 0:
            direction = "touch_zero"
            if i > 0:
                yp = y[i - 1]
                if yp > 0 and y1 < 0:
                    direction = "pos_to_neg"
                elif yp < 0 and y1 > 0:
                    direction = "neg_to_pos"
            crosses.append({"x_cross": float(x0), "direction": direction})
            continue

        if y0 * y1 < 0:
            x_cross = x0 - y0 * (x1 - x0) / (y1 - y0)

            if y0 > 0 and y1 < 0:
                direction = "pos_to_neg"
            else:
                direction = "neg_to_pos"

            crosses.append({"x_cross": float(x_cross), "direction": direction})

        elif y1 == 0:
            direction = "touch_zero"
            if y0 > 0:
                direction = "pos_to_neg"
            elif y0 < 0:
                direction = "neg_to_pos"
            crosses.append({"x_cross": float(x1), "direction": direction})

    return crosses

def build_interval_df_for_ecdf(df_in: pd.DataFrame) -> pd.DataFrame:
    
    if "landfall_time_utc" not in df_in.columns and "landfall_time" not in df_in.columns:
        raise KeyError("To plot ECDF panels, Excel must contain 'landfall_time_utc' or 'landfall_time'.")
    if "start_clip" not in df_in.columns and "start" not in df_in.columns:
        raise KeyError("To plot ECDF panels, Excel must contain 'start_clip' or 'start'.")

    d = df_in.copy()

    lf_col = "landfall_time_utc" if "landfall_time_utc" in d.columns else "landfall_time"
    st_col = "start_clip" if "start_clip" in d.columns else "start"

    d[lf_col] = pd.to_datetime(d[lf_col], errors="coerce")
    d[st_col] = pd.to_datetime(d[st_col], errors="coerce")
    d = d.dropna(subset=[lf_col, st_col]).copy()

    d["year"] = d[lf_col].dt.year.astype(int)
    d = d[(d["year"] >= YEAR_MIN) & (d["year"] <= YEAR_MAX)].copy()

    d["interval_h"] = (d[st_col] - d[lf_col]).dt.total_seconds() / 3600.0
    d["interval_h"] = pd.to_numeric(d["interval_h"], errors="coerce")
    d = d[np.isfinite(d["interval_h"].values.astype(float))].copy()
    d = d[(d["interval_h"] >= 0.0) & (d["interval_h"] <= float(WINDOW_H))].copy()

    d["Pdam"] = pd.to_numeric(d["Pdam"], errors="coerce")
    d = d[np.isfinite(d["Pdam"].values.astype(float)) & (d["Pdam"].values.astype(float) >= float(PDAM_MIN_ECDF))].copy()
    return d

def slope_with_sig(slope, p, fmt=".3g"):
    if slope is None or (not np.isfinite(slope)):
        return "Slope=nan"
    return f"Slope={format(float(slope), fmt)}{p_to_stars(p)}"

def _trend_slope_p(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if np.sum(m) < 3:
        return np.nan, np.nan, np.nan
    lr = linregress(x[m], y[m])
    return float(lr.slope), float(lr.intercept), float(lr.pvalue)

# =========================
# 1) Read + filter raw EVENT-level table
# =========================
df_evt = _read_excel_first_sheet(XLSX_PATH)

sid_col = _pick_first_existing(df_evt.columns, SID_CANDIDATES)
if sid_col is None:
    raise KeyError(
        "Excel must contain a SID column for deduplication.\n"
        f"Tried: {SID_CANDIDATES}\n"
        f"Available: {list(df_evt.columns)}"
    )

wind80_col = _pick_first_existing(df_evt.columns, WIND80_COL_CANDIDATES)
if wind80_col is None:
    raise KeyError(
        "Excel must contain landfall-day max wind at 80m.\n"
        f"Tried: {WIND80_COL_CANDIDATES}\n"
        f"Available: {list(df_evt.columns)}"
    )

if "Pdam" not in df_evt.columns:
    raise KeyError(f"Excel missing required column 'Pdam'. Available columns: {list(df_evt.columns)}")

df_evt[sid_col] = df_evt[sid_col].astype(str).str.strip()
df_evt[wind80_col] = pd.to_numeric(df_evt[wind80_col], errors="coerce")
df_evt["Pdam"] = pd.to_numeric(df_evt["Pdam"], errors="coerce")

if ONLY_POST and ("rel_to_landfall" in df_evt.columns):
    rel = df_evt["rel_to_landfall"].astype(str).str.lower().str.strip()
    df_evt = df_evt[rel == "post"].copy()

df_evt["year"] = _infer_year_series(df_evt)
df_evt = df_evt.dropna(subset=["year"]).copy()
df_evt["year"] = df_evt["year"].astype(int)
df_evt = df_evt[(df_evt["year"] >= YEAR_MIN) & (df_evt["year"] <= YEAR_MAX)].copy()

if df_evt.empty:
    raise RuntimeError("No valid EVENT-level rows after filtering.")

print("==== EVENT-level base (for e/f, NO SID dedup) ====")
print(f"Rows (TC-HW event records) = {len(df_evt)}")
print(f"Unique SID = {df_evt[sid_col].nunique(dropna=False)}")

# =========================
# 2) Build TC-level dedup table ONLY for (a)(b)(c)(d)
# =========================
df_tc = df_evt.copy()

sort_keys = []
if "landfall_time_utc" in df_tc.columns:
    df_tc["_sort_landfall"] = pd.to_datetime(df_tc["landfall_time_utc"], errors="coerce")
    sort_keys.append("_sort_landfall")
elif "landfall_time" in df_tc.columns:
    df_tc["_sort_landfall"] = pd.to_datetime(df_tc["landfall_time"], errors="coerce")
    sort_keys.append("_sort_landfall")

if "start_clip" in df_tc.columns:
    df_tc["_sort_start"] = pd.to_datetime(df_tc["start_clip"], errors="coerce")
    sort_keys.append("_sort_start")
elif "start" in df_tc.columns:
    df_tc["_sort_start"] = pd.to_datetime(df_tc["start"], errors="coerce")
    sort_keys.append("_sort_start")

if sort_keys:
    df_tc = df_tc.sort_values(sort_keys, ascending=True, na_position="last")

df_tc = df_tc.drop_duplicates(subset=[sid_col], keep="first").copy()

for c in ["_sort_landfall", "_sort_start"]:
    if c in df_tc.columns:
        df_tc.drop(columns=[c], inplace=True)

print("\n==== TC-level base (for a/b/c/d, SID dedup) ====")
print(f"Rows = {len(df_tc)} | Unique SID = {df_tc[sid_col].nunique(dropna=False)}")

# =========================
# 3) Panels (a)(b): yearly summaries + trends, using df_tc
# =========================
df_wind = df_tc[np.isfinite(df_tc[wind80_col].values.astype(float))].copy()
df_pdam = df_tc[np.isfinite(df_tc["Pdam"].values.astype(float))].copy()

if df_wind.empty:
    raise RuntimeError(f"No valid wind rows for column {wind80_col} in dedup table.")
if df_pdam.empty:
    raise RuntimeError("No valid Pdam rows in dedup table.")

yearsA, dataA, meanA = _yearly_box_mean(df_wind, "year", wind80_col)
yearsB, dataB, meanB = _yearly_box_mean(df_pdam, "year", "Pdam")

q1A = np.array([np.nanpercentile(v, 25) if len(v) else np.nan for v in dataA], dtype=float)
q3A = np.array([np.nanpercentile(v, 75) if len(v) else np.nan for v in dataA], dtype=float)

mA = np.isfinite(meanA)
xA = yearsA[mA].astype(float)
yA = meanA[mA].astype(float)
xgA = np.linspace(yearsA.min(), yearsA.max(), 300)
yhatA, loA, hiA, slopeA, icptA, pA = _linreg_ci_band(xA, yA, xgA, ci_level=CI_LEVEL)

mB = np.isfinite(meanB)
xB = yearsB[mB].astype(float)
yB = meanB[mB].astype(float)
xgB = np.linspace(yearsB.min(), yearsB.max(), 300)
yhatB, loB, hiB, slopeB, icptB, pB = _linreg_ci_band(xB, yB, xgB, ci_level=CI_LEVEL)

posA = (np.arange(len(yearsA)) + 1).astype(float)
posB = (np.arange(len(yearsB)) + 1).astype(float)
posgA = np.interp(xgA, yearsA.astype(float), posA)
posgB = np.interp(xgB, yearsB.astype(float), posB)

# =========================
# 4) Panels (c)(d), using df_tc
# =========================
col_cat = _pick_first_existing(df_tc.columns, ["tc_cat_from_lifetime_v10", "tc_cat", "ss_cat", "tc_category"])
col_life = _pick_first_existing(df_tc.columns, ["vmax_lifetime_1min", "vmax_lifetime_1min_ms", "vmax_lifetime", "life_vmax_1min"])

if col_cat is not None:
    tc_cat = pd.to_numeric(df_tc[col_cat], errors="coerce").to_numpy(dtype=float)
else:
    if col_life is None:
        raise KeyError("Excel must contain tc_cat_from_lifetime_v10 (or similar) OR vmax_lifetime_1min to compute category.")
    df_tc[col_life] = pd.to_numeric(df_tc[col_life], errors="coerce")
    tc_cat = np.array([classify_ss_from_ms(v) for v in df_tc[col_life].to_numpy(dtype=float)], dtype=float)

pdam_tc = df_tc["Pdam"].to_numpy(dtype=float)
valid = np.isfinite(pdam_tc) & np.isfinite(tc_cat)
vals_all = pdam_tc[valid]
cat_all = tc_cat[valid].astype(int)

sp_r, sp_p = spearmanr(cat_all, vals_all)
zero_frac = float(np.mean(vals_all == 0.0)) if vals_all.size else np.nan
high_frac = float(np.mean(vals_all >= PDAM_THR_LINE)) if vals_all.size else np.nan

tmp = pd.DataFrame({
    "year": pd.to_numeric(df_tc.loc[valid, "year"].to_numpy(), errors="coerce"),
    "pdam": vals_all
}).dropna(subset=["year"])
tmp["year"] = tmp["year"].astype(int)

yearly_gt0 = tmp.groupby("year").apply(
    lambda g: np.sum(np.isfinite(g["pdam"].to_numpy()) & (g["pdam"].to_numpy() > 0))
).rename("N_pdam_gt0")

yearly_ge09 = tmp.groupby("year").apply(
    lambda g: np.sum(np.isfinite(g["pdam"].to_numpy()) & (g["pdam"].to_numpy() >= PDAM_THR_LINE))
).rename("N_pdam_ge09")

yearly = pd.concat([yearly_gt0, yearly_ge09], axis=1).fillna(0.0).reset_index().sort_values("year")
x_year = yearly["year"].to_numpy(dtype=float)
y_gt0 = yearly["N_pdam_gt0"].to_numpy(dtype=float)
y_ge09 = yearly["N_pdam_ge09"].to_numpy(dtype=float)

slope_gt0, icpt_gt0, p_gt0 = _trend_slope_p(x_year, y_gt0)
slope_ge09, icpt_ge09, p_ge09 = _trend_slope_p(x_year, y_ge09)

# =========================
# 5) Panels (e)(f), using df_evt (NO SID dedup)
# =========================
df_ecdf = build_interval_df_for_ecdf(df_evt)

tmax = float(min(TAIL_MAX_H, WINDOW_H))
grid = np.linspace(0.0, tmax, 300)

vE = df_ecdf.loc[(df_ecdf["year"] >= EARLY_Y0) & (df_ecdf["year"] <= EARLY_Y1), "interval_h"].values.astype(float)
vL = df_ecdf.loc[(df_ecdf["year"] >= LATE_Y0)  & (df_ecdf["year"] <= LATE_Y1),  "interval_h"].values.astype(float)
vE = vE[np.isfinite(vE)]
vL = vL[np.isfinite(vL)]

xE, yE = ecdf_xy(vE)
xL, yL = ecdf_xy(vL)
d_hat, lo, hi = bootstrap_delta_ecdf(vE, vL, grid, B=BOOTSTRAP_N, alpha=CI_ALPHA, seed=RNG_SEED)

zero_crossings = find_zero_crossings(grid, d_hat)
zero_crossings_pos_to_neg = [c for c in zero_crossings if c["direction"] == "pos_to_neg"]
first_pos_to_neg = zero_crossings_pos_to_neg[0] if len(zero_crossings_pos_to_neg) > 0 else None

print("\n==== Zero crossings of ΔECDF ====")
if len(zero_crossings) == 0:
    print("No zero crossing found.")
else:
    for i, c in enumerate(zero_crossings, start=1):
        print(f"{i}. x = {c['x_cross']:.2f} h, direction = {c['direction']}")

if first_pos_to_neg is not None:
    print(f"First pos_to_neg crossing: {first_pos_to_neg['x_cross']:.2f} h")
else:
    print("No pos_to_neg crossing found.")

print("\n==== e/f EVENT-level sample sizes (NO SID dedup) ====")
print(f"Early ({EARLY_Y0}-{EARLY_Y1}) events = {len(vE)}")
print(f"Late  ({LATE_Y0}-{LATE_Y1}) events = {len(vL)}")

# =========================
# 6) Plot
# =========================
fig = plt.figure(figsize=(16, 10), dpi=800)
gs = fig.add_gridspec(2, 3, wspace=0.28, hspace=ROW_HSPACE)

axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[0, 2])
axD = fig.add_subplot(gs[1, 0])
axE = fig.add_subplot(gs[1, 1])
axF = fig.add_subplot(gs[1, 2])

# -------------------------
# Panel (a) : TC-level
# -------------------------
m_iqr = np.isfinite(q1A) & np.isfinite(q3A)
axA.vlines(posA[m_iqr], q1A[m_iqr], q3A[m_iqr], colors="black", linewidth=1.3, zorder=2)
axA.scatter(posA, meanA, marker="D", s=25, color=BLUE, edgecolor=BLUE, zorder=3)
axA.plot(posgA, yhatA, linewidth=LW_TREND, color=BLUE)
if np.any(np.isfinite(loA)) and np.any(np.isfinite(hiA)):
    axA.fill_between(posgA, loA, hiA, alpha=ALPHA_CI_AB, color=BLUE, linewidth=0)

_set_decade_ticks_only(axA, yearsA, tick_every=DECADE_TICK, hide_last_year_label=HIDE_LAST_YEAR_LABEL)
axA.set_xlabel("Year")
axA.set_ylabel(r"Landfall max_wind speed (m s$^{-1}$)")
axA.grid(False)
_annotate_slope(axA, slopeA, pA, unit_text="", xy=(0.60, 0.93))

# -------------------------
# Panel (b) : TC-level
# -------------------------
axB.plot(posB, meanB, linestyle="--", linewidth=LW_PROC, color=BLUE, zorder=2)
axB.scatter(posB, meanB, marker="s", s=28, color=BLUE, edgecolor=BLUE, zorder=3)
axB.plot(posgB, yhatB, linewidth=LW_TREND, color=BLUE, zorder=2)
if np.any(np.isfinite(loB)) and np.any(np.isfinite(hiB)):
    axB.fill_between(posgB, loB, hiB, alpha=ALPHA_CI_AB, color=BLUE, linewidth=0, zorder=1)

_set_decade_ticks_only(axB, yearsB, tick_every=DECADE_TICK, hide_last_year_label=HIDE_LAST_YEAR_LABEL)
axB.set_xlabel("Year")
axB.set_ylabel("Damage probability (Pdam)")
axB.set_ylim(-0.02, 1.02)
axB.grid(False)
_annotate_slope(axB, slopeB, pB, unit_text="", xy=(0.50, 0.93))

# -------------------------
# Panel (c) : TC-level
# -------------------------
bin_edges = np.linspace(0.0, 1.0, 31)
axC.hist(vals_all, bins=bin_edges, edgecolor="k", linewidth=0.5, color=BLUE)
axC.set_xlabel("Pdam")
axC.set_ylabel("Number of TCs followd by HWs")
axC.grid(False)
axC.text(0.10, 0.25, f"Pdam=0: {zero_frac*100:.0f}%",
         transform=axC.transAxes, va="top", ha="left")
axC.text(0.52, 0.17, f"Pdam≥{PDAM_THR_LINE:g}: {high_frac*100:.0f}%",
         transform=axC.transAxes, va="top", ha="left", color="#b2182b")

axC.yaxis.set_label_position("left")
axC.yaxis.tick_left()

vals_nz = vals_all[vals_all > 0]
axins = None
if vals_nz.size >= 5:
    axins = axC.inset_axes([0.33, 0.45, 0.43, 0.43])
    counts, edges = np.histogram(vals_nz, bins=bin_edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = edges[1:] - edges[:-1]
    axins.bar(centers, counts, width=widths, align="center",
              edgecolor="k", linewidth=0.4, color=BLUE)
    axins.set_yscale("log")
    axins.set_xlim(-0.02, 1.02)
    axins.set_title("Pdam >0 (log count)", fontsize=14)
    axins.grid(True, axis="y", linestyle="--", alpha=ALPHA_ZERO)
    pos = counts[counts > 0]
    if pos.size > 0:
        axins.set_ylim(0.8, pos.max() * 1.3)

# -------------------------
# Panel (d) : TC-level
# -------------------------
years_plot = yearly["year"].to_numpy(dtype=float)
n_gt0 = yearly["N_pdam_gt0"].to_numpy(dtype=float)
n_ge09 = yearly["N_pdam_ge09"].to_numpy(dtype=float)

x = years_plot.astype(float)
edges = np.r_[x - 0.5, x[-1] + 0.5]

lab_gt0  = f"N(Pdam>0)\n{slope_with_sig(slope_gt0, p_gt0, fmt='.2g')}"
lab_ge09 = f"N(Pdam≥{PDAM_THR_LINE:g})\n{slope_with_sig(slope_ge09, p_ge09, fmt='.2g')}"

axD.step(edges, np.r_[n_gt0, n_gt0[-1]], where="post",
         linewidth=1.2, color="#2166ac")
axD.plot(years_plot, n_ge09, linewidth=2.0, color="#b2182b")

axD.set_xlabel("Year")
axD.set_ylabel("Number of TCs followd by HWs")
axD.grid(False)
axD.text(0.03, 0.72, lab_gt0, transform=axD.transAxes, ha="left", va="center", color="#2166ac")
axD.text(0.03, 0.52, lab_ge09, transform=axD.transAxes, ha="left", va="center", color="#b2182b")

# -------------------------
# Panel (e) : EVENT-level (NO SID dedup)
# -------------------------
if xE.size > 0:
    mE = xE <= tmax
    axE.step(xE[mE], yE[mE], where="post", linewidth=LW_ECDF, color="#2166ac")
if xL.size > 0:
    mL = xL <= tmax
    axE.step(xL[mL], yL[mL], where="post", linewidth=LW_ECDF, color="#b2182b")

axE.set_xlim(0, tmax)
axE.set_ylim(0, 1.0)
axE.set_xticks(np.arange(0, tmax + 0.1, EF_XTICK_INTERVAL))
axE.grid(False)
axE.set_xlabel("TC-HW time interval (hours)")
axE.set_ylabel("ECDF")
axE.set_title("")
axE.text(0.16, 0.90, "Pdam≥0.9", transform=axE.transAxes, ha="left", va="center")

axE.text(0.53, 0.15, f"Early ({EARLY_Y0}-{EARLY_Y1})", transform=axE.transAxes,
         ha="left", va="bottom", color="#2166ac")
axE.text(0.53, 0.07, f"Late ({LATE_Y0}-{LATE_Y1})", transform=axE.transAxes,
         ha="left", va="bottom", color="#b2182b")

# -------------------------
# Panel (f) : EVENT-level (NO SID dedup)
# -------------------------
axF.axhline(0.0, linewidth=1.0, color="black")
axF.plot(grid, d_hat, linewidth=LW_DELTA, color=BLUE)
if np.any(np.isfinite(lo)) and np.any(np.isfinite(hi)):
    axF.fill_between(grid, lo, hi, alpha=ALPHA_CI_F, color=BLUE, linewidth=0)

if first_pos_to_neg is not None:
    x_cross = first_pos_to_neg["x_cross"]

    axF.scatter([x_cross], [0.0], s=55, color="black", zorder=6)

    axF.annotate(f"{x_cross:.1f} h",
                 xy=(x_cross, 0.0),
                 xytext=(x_cross + 6, 0.06),
                 textcoords="data",
                 ha="left", va="bottom",
                 fontsize=16, color="black",
                 arrowprops=dict(arrowstyle="-", color="black", lw=0.8))

axF.set_xlim(0, tmax)
axF.set_ylim(-0.6, 0.6)
axF.set_xticks(np.arange(0, tmax + 0.1, EF_XTICK_INTERVAL))
axF.grid(False)
axF.set_xlabel("TC-HW time interval (hours)")
axF.set_ylabel("ΔECDF=ECDF(late)-ECDF(early)")
axF.set_title("")

TEXT_POS_AXFRAC = {
    24: (0.03, 0.77),
    48: (0.30, 0.86),
    72: (0.63, 0.88),
    96: (0.18, 0.35),
    120: (0.40, 0.20),
    144: (0.70, 0.08),
    168: (0.70, 0.50),
}
for tt in [24, 48, 72, 96, 120, 144]:
    if tt <= tmax:
        val = float(np.interp(tt, grid, d_hat))
        axF.scatter([tt], [val], s=40, color="#b2182b", edgecolor="#b2182b", zorder=5)
        if np.any(np.isfinite(lo)) and np.any(np.isfinite(hi)):
            lo_t = float(np.interp(tt, grid, lo))
            hi_t = float(np.interp(tt, grid, hi))
            txt = f"{tt}h:{val:+.2f}\n[{lo_t:+.2f},{hi_t:+.2f}]"
        else:
            txt = f"{tt}h:{val:+.2f}"
        xaf, yaf = TEXT_POS_AXFRAC.get(tt, (0.05, 0.90))
        axF.text(xaf, yaf, txt, transform=axF.transAxes, ha="left", va="center")

# -------------------------
# Panel labels + frame widths
# -------------------------
add_panel_label(axA, "(a)")
add_panel_label(axB, "(b)")
add_panel_label(axC, "(c)")
add_panel_label(axD, "(d)")
add_panel_label(axE, "(e)")
add_panel_label(axF, "(f)")

for ax in [axA, axB, axC, axD, axE, axF]:
    set_axes_frame_width(ax, lw=FRAME_LW)
if axins is not None:
    set_axes_frame_width(axins, lw=1.0)

fig.subplots_adjust(left=0.05, right=0.97, top=0.93, bottom=0.10)
plt.show()

print("\nDONE.")
print("Panels (a)(b)(c)(d): TC-level, SID dedup.")
print("Panels (e)(f): EVENT-level, NO SID dedup.")