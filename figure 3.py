import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties


# ============================================================
# 0) Fonts
# ============================================================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.default": "regular",
    "font.size": 11,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})

arial_font = FontProperties(family="Arial", size=11)
arial_font_small = FontProperties(family="Arial", size=9.8)
arial_font_bold = FontProperties(family="Arial", size=11, weight="bold")


# ============================================================
# 1) User settings
# ============================================================

IN_DAILY_CSV = r"G:\LHY\[new]\TC-HW compound\TC-HW\Tmax&T90\all_TC_post7d_daily_Tmax_T90_distribution.csv"

IN_TCHW_EVENT = r"G:\LHY\[new]\TC-HW compound\TC-HW\compound_events_post7d_from_post30d__no_small_islands__with_Pdam.xlsx"
EVENT_SHEET = 0

LAGS_TO_USE = [0, 1, 2, 3, 4, 5, 6]
HOUR_TICKS = [24, 48, 72, 96, 120, 144, 168]
HOUR_TICKS_3D = [24, 48, 72, 96, 120]

# Early/Late comparison
PERIOD_DEFS = {
    "Early": (1980, 1999),
    "Late": (2000, 2024),
}

PERIOD_LABELS = {
    "Early": "Early (1980-1999)",
    "Late": "Late (2000-2024)",
}

PERIOD_LABELS_FULL = {
    "Early": "Early (1980–1999)",
    "Late": "Late (2000–2024)",
}

# Transition-to-hot settings
INITIAL_GROUP_HOUR = 24
PANEL_STATES = ["E≤-2", "-2<E≤-1", "-1<E≤0"]

BOOTSTRAP_N = 1000
CI_ALPHA = 0.05
RNG_SEED = 2026

# Colors
COLOR_ALL = "#2166ac"
COLOR_TCHW = "#b2182b"

COLOR_EARLY = "#2166ac"
COLOR_LATE = "#b2182b"

PERIOD_COLORS = {
    "Early": COLOR_EARLY,
    "Late": COLOR_LATE,
}

CAT_COLORS = {
    "E≤-2": "#08306b",
    "-2<E≤-1": "#6baed6",
    "-1<E≤0": "#d9d9d9",
    "0<E≤1": "#fcae91",
    "E>1": "#cb181d",
}

CATEGORY_ORDER = ["E≤-2", "-2<E≤-1", "-1<E≤0", "0<E≤1", "E>1"]

PANEL_STATE_COLORS = CAT_COLORS.copy()
PANEL_STATE_COLORS["-1<E≤0"] = "#8c8987"


# ============================================================
# 2) Helper functions
# ============================================================

def lag_to_hour(lag_day):
    return (int(lag_day) + 1) * 24


def assign_period(year):
    for pname, (y0, y1) in PERIOD_DEFS.items():
        if y0 <= year <= y1:
            return pname
    return np.nan


def classify_E_zone(E):
    if not np.isfinite(E):
        return np.nan
    if E <= -2:
        return "E≤-2"
    elif E <= -1:
        return "-2<E≤-1"
    elif E <= 0:
        return "-1<E≤0"
    elif E <= 1:
        return "0<E≤1"
    else:
        return "E>1"


def stat_near(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return np.nan

    return float(np.mean((vals > -1.0) & (vals <= 0.0)))


def stat_hot(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return np.nan

    return float(np.mean(vals > 0.0))


def bootstrap_ci(vals, stat_func, B=1000, alpha=0.05, seed=2026):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]

    if vals.size < 5:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    n = vals.size
    out = np.empty(B, dtype=float)

    for b in range(B):
        sample = vals[rng.integers(0, n, size=n)]
        out[b] = stat_func(sample)

    lo = np.nanpercentile(out, 100 * alpha / 2)
    hi = np.nanpercentile(out, 100 * (1 - alpha / 2))

    return float(lo), float(hi)


def bootstrap_ci_binary(vals, B=1000, alpha=0.05, seed=2026):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]

    if vals.size < 5:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    n = vals.size
    out = np.empty(B, dtype=float)

    for b in range(B):
        sample = vals[rng.integers(0, n, size=n)]
        out[b] = np.mean(sample)

    lo = np.nanpercentile(out, 100 * alpha / 2)
    hi = np.nanpercentile(out, 100 * (1 - alpha / 2))

    return float(lo), float(hi)


def add_panel_label(ax, label):
    ax.text(
        0.02, 0.96, label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=20,
        fontweight="bold",
        fontfamily="Arial"
    )


def set_frame(ax, lw=1.1):
    for spine in ax.spines.values():
        spine.set_linewidth(lw)


def set_axis_color(ax, color, side="left"):
    ax.tick_params(axis="y", colors=color)
    ax.yaxis.label.set_color(color)

    if side == "left":
        ax.spines["left"].set_color(color)
    elif side == "right":
        ax.spines["right"].set_color(color)


def line_with_ci(ax, d, x_col, y_col, lo_col, hi_col, color, label, yscale=1.0):
    x = d[x_col].values
    y = d[y_col].values * yscale
    lo = d[lo_col].values * yscale
    hi = d[hi_col].values * yscale

    yerr_low = np.maximum(0, y - lo)
    yerr_high = np.maximum(0, hi - y)

    h = ax.errorbar(
        x,
        y,
        yerr=[yerr_low, yerr_high],
        marker="o",
        linewidth=2.0,
        markersize=4.8,
        capsize=3,
        color=color,
        label=label
    )

    return h


def build_transition_to_hot_curve(
    dfin,
    group_col,
    group_name,
    initial_group_hour,
    hours_to_use,
    states_to_use
):
   
    pivot_E = dfin.pivot_table(
        index="tc_uid",
        columns="lag_hour",
        values="E",
        aggfunc="first"
    )

    pivot_state = dfin.pivot_table(
        index="tc_uid",
        columns="lag_hour",
        values="E_zone",
        aggfunc="first"
    )

    available_hours = [h for h in hours_to_use if h in pivot_E.columns]

    if initial_group_hour not in pivot_state.columns:
        return pd.DataFrame(
            columns=[group_col, "initial_state", "hour", "share_ever_hot_pct", "n_state"]
        )

    initial_state = pivot_state[initial_group_hour].copy()
    valid_uid = initial_state.dropna().index

    pivot_E = pivot_E.loc[valid_uid, available_hours].copy()
    initial_state = initial_state.loc[valid_uid]

    rows = []

    for st in states_to_use:
        uid_st = initial_state[initial_state == st].index

        if len(uid_st) == 0:
            continue

        subE = pivot_E.loc[uid_st]

        for h in available_hours:
            cols_up_to_h = [x for x in available_hours if x <= h]
            ever_hot = (subE[cols_up_to_h] > 0).any(axis=1)

            rows.append({
                group_col: group_name,
                "initial_state": st,
                "hour": h,
                "share_ever_hot_pct": ever_hot.mean() * 100.0,
                "n_state": len(uid_st)
            })

    return pd.DataFrame(rows)


# ============================================================
# 3) Read and prepare daily data
# ============================================================

df = pd.read_csv(IN_DAILY_CSV, encoding="utf-8-sig")

df["sid"] = df["sid"].astype(str).str.strip()
df["tc_uid"] = pd.to_numeric(df["tc_uid"], errors="coerce").astype("Int64")
df["landfall_time_utc"] = pd.to_datetime(
    df["landfall_time_utc"],
    errors="coerce"
).dt.floor("h")

df["grid_lat"] = pd.to_numeric(df["grid_lat"], errors="coerce")
df["grid_lon"] = pd.to_numeric(df["grid_lon"], errors="coerce")

df["lag_day"] = pd.to_numeric(
    df["lag_day_from_landfall_day"],
    errors="coerce"
).astype("Int64")

df["lag_hour"] = df["lag_day"].apply(
    lambda x: lag_to_hour(x) if pd.notna(x) else np.nan
)

df["E"] = pd.to_numeric(df["Tmax_minus_T90_C"], errors="coerce")

df = df[
    df["tc_uid"].notna()
    & df["landfall_time_utc"].notna()
    & df["lag_day"].isin(LAGS_TO_USE)
    & np.isfinite(df["E"])
].copy()

df["tc_uid"] = df["tc_uid"].astype(int)
df["year"] = df["landfall_time_utc"].dt.year
df["period"] = df["year"].apply(assign_period)

df["E_zone"] = df["E"].apply(classify_E_zone)
df["near_flag"] = ((df["E"] > -1.0) & (df["E"] <= 0.0)).astype(float)
df["hot_flag"] = (df["E"] > 0.0).astype(float)

df_units = df[
    ["tc_uid", "sid", "landfall_time_utc", "grid_lat", "grid_lon"]
].drop_duplicates().copy()

df_units["lat_key"] = df_units["grid_lat"].round(4)
df_units["lon_key"] = df_units["grid_lon"].round(4)


# ============================================================
# 4) Read TC-HW event table and map to tc_uid
# ============================================================

df_evt = pd.read_excel(IN_TCHW_EVENT, sheet_name=EVENT_SHEET, engine="openpyxl")

df_evt["sid"] = df_evt["sid"].astype(str).str.strip()
df_evt["landfall_time_utc"] = pd.to_datetime(
    df_evt["landfall_time_utc"],
    errors="coerce"
).dt.floor("h")

df_evt["hw_start"] = pd.to_datetime(df_evt["hw_start"], errors="coerce")
df_evt["grid_lat"] = pd.to_numeric(df_evt["grid_lat"], errors="coerce")
df_evt["grid_lon"] = pd.to_numeric(df_evt["grid_lon"], errors="coerce")

df_evt["lat_key"] = df_evt["grid_lat"].round(4)
df_evt["lon_key"] = df_evt["grid_lon"].round(4)

df_evt = df_evt.dropna(
    subset=["sid", "landfall_time_utc", "hw_start", "lat_key", "lon_key"]
).copy()

df_evt_m = df_evt.merge(
    df_units[["tc_uid", "sid", "landfall_time_utc", "lat_key", "lon_key"]],
    on=["sid", "landfall_time_utc", "lat_key", "lon_key"],
    how="left"
)

print("========================================")
print("TC-HW event mapping")
print("========================================")
print(
    f"Matched TC-HW event rows to all-TC units: "
    f"{df_evt_m['tc_uid'].notna().sum()}/{len(df_evt_m)}"
)

df_evt_m = df_evt_m.dropna(subset=["tc_uid"]).copy()
df_evt_m["tc_uid"] = df_evt_m["tc_uid"].astype(int)

df_evt_m["dt_hw_hours"] = (
    (df_evt_m["hw_start"] - df_evt_m["landfall_time_utc"])
    .dt.total_seconds() / 3600.0
)

df_evt_m = df_evt_m[np.isfinite(df_evt_m["dt_hw_hours"])].copy()
df_evt_m = df_evt_m[df_evt_m["dt_hw_hours"] > 0].copy()

df_evt_min = (
    df_evt_m.sort_values("dt_hw_hours")
    .drop_duplicates(subset=["tc_uid"], keep="first")
    .reset_index(drop=True)
)

tchw_uid_set = set(df_evt_min["tc_uid"].tolist())

df["is_tchw"] = df["tc_uid"].isin(tchw_uid_set)

print("Unique TC-HW tc_uid matched:", len(tchw_uid_set))
print("All TC tc_uid:", df["tc_uid"].nunique())

if df["tc_uid"].nunique() > 0:
    print(
        "TC-HW share:",
        f"{len(tchw_uid_set) / df['tc_uid'].nunique() * 100:.2f}%"
    )


# ============================================================
# 5) Statistics for panels a/b/c: All TC vs TC-HW
# ============================================================

group_defs = {
    "All TC": df.copy(),
    "TC-HW": df[df["is_tchw"]].copy()
}

# ---------- Panel a: E-zone distribution ----------
rows_cat_group = []

for gname, dfg in group_defs.items():
    for lag in LAGS_TO_USE:
        sub = dfg[dfg["lag_day"] == lag].copy()
        n = len(sub)

        if n == 0:
            continue

        counts = sub["E_zone"].value_counts(dropna=True)

        for cat in CATEGORY_ORDER:
            rows_cat_group.append({
                "group": gname,
                "lag_day": lag,
                "lag_hour": lag_to_hour(lag),
                "E_zone": cat,
                "share": counts.get(cat, 0) / n
            })

df_cat_group = pd.DataFrame(rows_cat_group)


# ---------- Panel b: transition-to-hot ----------
df_curve_all = build_transition_to_hot_curve(
    dfin=df.copy(),
    group_col="group",
    group_name="All TC",
    initial_group_hour=INITIAL_GROUP_HOUR,
    hours_to_use=HOUR_TICKS,
    states_to_use=PANEL_STATES
)

df_curve_tchw = build_transition_to_hot_curve(
    dfin=df[df["is_tchw"]].copy(),
    group_col="group",
    group_name="TC-HW",
    initial_group_hour=INITIAL_GROUP_HOUR,
    hours_to_use=HOUR_TICKS,
    states_to_use=PANEL_STATES
)

df_curve_group = pd.concat([df_curve_all, df_curve_tchw], ignore_index=True)


# ---------- Panel c: 3-day exceedance probability ----------
pivot_group = df.pivot_table(
    index=["tc_uid", "sid"],
    columns="lag_day",
    values="E",
    aggfunc="first"
)

rows_3d_group = []

for (tc_uid, sid), row in pivot_group.iterrows():
    is_tchw = tc_uid in tchw_uid_set

    for s in range(0, 5):
        vals = [
            row.get(s, np.nan),
            row.get(s + 1, np.nan),
            row.get(s + 2, np.nan)
        ]

        if np.all(np.isfinite(vals)):
            rows_3d_group.append({
                "tc_uid": tc_uid,
                "sid": sid,
                "start_lag": s,
                "start_hour": lag_to_hour(s),
                "exceed3": float((vals[0] > 0) and (vals[1] > 0) and (vals[2] > 0)),
                "is_tchw": is_tchw
            })

df_3d_group_raw = pd.DataFrame(rows_3d_group)

rows_3d_group_stats = []

for gname in ["All TC", "TC-HW"]:
    if gname == "All TC":
        dfg = df_3d_group_raw.copy()
    else:
        dfg = df_3d_group_raw[df_3d_group_raw["is_tchw"]].copy()

    for s in range(0, 5):
        sub = dfg[dfg["start_lag"] == s].copy()

        if sub.empty:
            continue

        p = sub["exceed3"].mean()

        lo, hi = bootstrap_ci_binary(
            sub["exceed3"].values,
            B=BOOTSTRAP_N,
            alpha=CI_ALPHA,
            seed=RNG_SEED + 200 + s + (0 if gname == "All TC" else 500)
        )

        rows_3d_group_stats.append({
            "group": gname,
            "start_lag": s,
            "start_hour": lag_to_hour(s),
            "p_exceed3": p,
            "p_exceed3_lo": lo,
            "p_exceed3_hi": hi,
            "N_units": sub["tc_uid"].nunique()
        })

stats_3d_group = pd.DataFrame(rows_3d_group_stats)


# ============================================================
# 6) Statistics for panels d/e: Early vs Late
# ============================================================

df_period = df[df["period"].notna()].copy()
period_names = list(PERIOD_DEFS.keys())

print("\n========================================")
print("Period settings")
print("========================================")

for pname, (y0, y1) in PERIOD_DEFS.items():
    dtmp = df_period[df_period["period"] == pname]
    print(f"{pname}: {y0}-{y1}")
    print(f"  rows: {len(dtmp)}")
    print(f"  unique tc_uid: {dtmp['tc_uid'].nunique()}")
    print(f"  unique sid: {dtmp['sid'].nunique()}")


# ---------- Panel d: hot-day proportion ----------
rows_period_stats = []

for pname in period_names:
    dfg = df_period[df_period["period"] == pname].copy()

    for lag in LAGS_TO_USE:
        sub = dfg[dfg["lag_day"] == lag].copy()
        vals = sub["E"].values

        if len(vals) == 0:
            continue

        hot = stat_hot(vals)
        seed_offset = 0 if pname == "Early" else 500

        hot_lo, hot_hi = bootstrap_ci(
            vals,
            stat_hot,
            B=BOOTSTRAP_N,
            alpha=CI_ALPHA,
            seed=RNG_SEED + 100 + lag + seed_offset
        )

        rows_period_stats.append({
            "period": pname,
            "period_label": PERIOD_LABELS[pname],
            "lag_day": lag,
            "lag_hour": lag_to_hour(lag),
            "hot_prop": hot,
            "hot_prop_lo": hot_lo,
            "hot_prop_hi": hot_hi,
            "N_rows": len(sub),
            "N_tcuid": sub["tc_uid"].nunique(),
            "N_sid": sub["sid"].nunique()
        })

stats_period = pd.DataFrame(rows_period_stats)


# ---------- Panel e: Early/Late 3-day exceedance probability ----------
pivot_period = df_period.pivot_table(
    index=["tc_uid", "sid", "period"],
    columns="lag_day",
    values="E",
    aggfunc="first"
)

rows_3d_period = []

for (tc_uid, sid, period), row in pivot_period.iterrows():
    for s in range(0, 5):
        vals = [
            row.get(s, np.nan),
            row.get(s + 1, np.nan),
            row.get(s + 2, np.nan)
        ]

        if np.all(np.isfinite(vals)):
            rows_3d_period.append({
                "tc_uid": tc_uid,
                "sid": sid,
                "period": period,
                "start_lag": s,
                "start_hour": lag_to_hour(s),
                "exceed3": float((vals[0] > 0) and (vals[1] > 0) and (vals[2] > 0)),
            })

df_3d_period_raw = pd.DataFrame(rows_3d_period)

rows_3d_period_stats = []

for pname in period_names:
    dfg = df_3d_period_raw[df_3d_period_raw["period"] == pname].copy()

    for s in range(0, 5):
        sub = dfg[dfg["start_lag"] == s].copy()

        if sub.empty:
            continue

        p = sub["exceed3"].mean()
        seed_offset = 0 if pname == "Early" else 500

        lo, hi = bootstrap_ci_binary(
            sub["exceed3"].values,
            B=BOOTSTRAP_N,
            alpha=CI_ALPHA,
            seed=RNG_SEED + 400 + s + seed_offset
        )

        rows_3d_period_stats.append({
            "period": pname,
            "period_label": PERIOD_LABELS[pname],
            "start_lag": s,
            "start_hour": lag_to_hour(s),
            "p_exceed3": p,
            "p_exceed3_lo": lo,
            "p_exceed3_hi": hi,
            "N_units": sub["tc_uid"].nunique()
        })

stats_3d_period = pd.DataFrame(rows_3d_period_stats)


# ============================================================
# 7) Plot
# ============================================================

fig = plt.figure(figsize=(12.2, 9.6), dpi=600)

outer = fig.add_gridspec(
    3, 2,
    height_ratios=[1.0, 1.0, 1.06],
    width_ratios=[1.05, 1.0],
    hspace=0.34,
    wspace=0.25
)

# Panel a: legend + plot, spanning top two rows in left column
gs_a = outer[0:2, 0].subgridspec(
    2, 1,
    height_ratios=[0.22, 0.78],
    hspace=0.02
)

axA_leg = fig.add_subplot(gs_a[0, 0])
axA = fig.add_subplot(gs_a[1, 0])

# Panels b/c/d/e
axB = fig.add_subplot(outer[0, 1])
axC = fig.add_subplot(outer[1, 1])
axD = fig.add_subplot(outer[2, 0])
axE = fig.add_subplot(outer[2, 1])


# ============================================================
# Panel a: E distribution, All TC vs TC-HW
# ============================================================

axA_leg.axis("off")

axA_leg.text(
    0.02, 0.90, "(a)",
    ha="left",
    va="top",
    fontsize=20,
    fontweight="bold",
    fontfamily="Arial",
    transform=axA_leg.transAxes
)

ROW1_Y = 0.82
ROW2_Y = 0.50
ROW3_Y = 0.18

axA_leg.text(
    0.50,
    ROW1_Y,
    r"$E = T_{\max} - T_{90}$",
    ha="center",
    va="center",
    fontsize=13,
    fontfamily="Arial",
    transform=axA_leg.transAxes
)

cat_handles = [
    Patch(facecolor=CAT_COLORS[c], edgecolor="black", label=c)
    for c in CATEGORY_ORDER
]

group_handles = [
    Patch(facecolor="white", edgecolor="black", hatch="//", label="All TC"),
    Patch(facecolor="white", edgecolor="black", label="TC-HW"),
]

leg_cat = axA_leg.legend(
    handles=cat_handles,
    ncol=5,
    frameon=False,
    loc="center",
    bbox_to_anchor=(0.50, ROW2_Y),
    columnspacing=0.85,
    handlelength=0.95,
    handleheight=0.75,
    borderpad=0.15,
    labelspacing=0.35,
    prop=arial_font
)

axA_leg.add_artist(leg_cat)

axA_leg.legend(
    handles=group_handles,
    ncol=2,
    frameon=False,
    loc="center",
    bbox_to_anchor=(0.50, ROW3_Y),
    columnspacing=1.65,
    handlelength=1.90,
    handleheight=0.75,
    borderpad=0.15,
    labelspacing=0.35,
    prop=arial_font
)

bar_w = 8.0
offset = 5.0

for gname, xshift, alpha, hatch in [
    ("All TC", -offset, 0.55, "//"),
    ("TC-HW", offset, 0.90, None),
]:
    sub = df_cat_group[df_cat_group["group"] == gname].copy()
    bottom = np.zeros(len(HOUR_TICKS), dtype=float)

    for cat in CATEGORY_ORDER:
        vals = []

        for h in HOUR_TICKS:
            x = sub[
                (sub["lag_hour"] == h)
                & (sub["E_zone"] == cat)
            ]["share"]

            vals.append(float(x.iloc[0]) if len(x) else 0.0)

        vals = np.array(vals, dtype=float)

        axA.bar(
            np.array(HOUR_TICKS) + xshift,
            vals * 100,
            width=bar_w,
            bottom=bottom * 100,
            color=CAT_COLORS[cat],
            edgecolor="black",
            linewidth=0.4,
            alpha=alpha,
            hatch=hatch
        )

        bottom += vals

axA.set_xlim(12, 180)
axA.set_xticks(HOUR_TICKS)
axA.set_xlabel("Time after TC landfall (hours)")
axA.set_ylabel("Proportion (%)")
axA.set_ylim(0, 100)
axA.grid(axis="y", linestyle="--", alpha=0.25)


# ============================================================
# Panel b: transition-to-hot, All TC vs TC-HW
# ============================================================

for st in PANEL_STATES:
    sub_all = df_curve_group[
        (df_curve_group["group"] == "All TC")
        & (df_curve_group["initial_state"] == st)
    ].sort_values("hour")

    if not sub_all.empty:
        axB.plot(
            sub_all["hour"],
            sub_all["share_ever_hot_pct"],
            linestyle="--",
            linewidth=1.8,
            color=PANEL_STATE_COLORS[st]
        )

    sub_tchw = df_curve_group[
        (df_curve_group["group"] == "TC-HW")
        & (df_curve_group["initial_state"] == st)
    ].sort_values("hour")

    if not sub_tchw.empty:
        axB.plot(
            sub_tchw["hour"],
            sub_tchw["share_ever_hot_pct"],
            linestyle="-",
            linewidth=2.2,
            color=PANEL_STATE_COLORS[st]
        )

axB.set_xlim(18, 174)
axB.set_xticks(HOUR_TICKS)
axB.set_xlabel("Time after TC landfall (hours)")
axB.set_ylabel("Proportion reaching\n$T_{\max}>T_{90}$ by time $t$ (%)")
axB.set_ylim(-5, 105)
axB.grid(False)

state_handles = [
    Line2D([], [], linestyle="None", label=st)
    for st in PANEL_STATES
]

leg_b = axB.legend(
    handles=state_handles,
    ncol=len(PANEL_STATES),
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.58, 0.98),
    handlelength=0,
    handletextpad=0.25,
    columnspacing=0.90,
    borderpad=0.1,
    labelspacing=0.25,
    prop=arial_font_bold
)

for txt, st in zip(leg_b.get_texts(), PANEL_STATES):
    txt.set_color(PANEL_STATE_COLORS[st])
    txt.set_fontfamily("Arial")
    txt.set_fontweight("bold")

# Line-style annotation
x0, x1 = 0.06, 0.13
y_top, y_bot = 0.62, 0.50

axB.plot(
    [x0, x1],
    [y_top, y_top],
    transform=axB.transAxes,
    color="black",
    linestyle="--",
    linewidth=1.8,
    clip_on=False
)

axB.text(
    x1 + 0.03,
    y_top,
    "All TC",
    transform=axB.transAxes,
    ha="left",
    va="center",
    fontsize=10.5,
    fontfamily="Arial"
)

axB.plot(
    [x0, x1],
    [y_bot, y_bot],
    transform=axB.transAxes,
    color="black",
    linestyle="-",
    linewidth=2.2,
    clip_on=False
)

axB.text(
    x1 + 0.03,
    y_bot,
    "TC-HW",
    transform=axB.transAxes,
    ha="left",
    va="center",
    fontsize=10.5,
    fontfamily="Arial"
)


# ============================================================
# Panel c: 3-day exceedance, All TC vs TC-HW
# ============================================================

axC_right = axC.twinx()

d_tchw_3d = stats_3d_group[
    stats_3d_group["group"] == "TC-HW"
].sort_values("start_hour")

d_all_3d = stats_3d_group[
    stats_3d_group["group"] == "All TC"
].sort_values("start_hour")

h_tchw = line_with_ci(
    axC,
    d_tchw_3d,
    "start_hour",
    "p_exceed3",
    "p_exceed3_lo",
    "p_exceed3_hi",
    COLOR_TCHW,
    "TC-HW",
    yscale=100
)

h_all = line_with_ci(
    axC_right,
    d_all_3d,
    "start_hour",
    "p_exceed3",
    "p_exceed3_lo",
    "p_exceed3_hi",
    COLOR_ALL,
    "All TC",
    yscale=100
)

axC.set_xlim(12, 132)
axC_right.set_xlim(12, 132)

axC.set_xticks(HOUR_TICKS_3D)
axC.set_xlabel("Time after TC landfall (hours)")

axC.set_ylabel("3-day exceedance\nprobability (%)",
    color="black")
#axC_right.set_ylabel("All TC 3-day\nexceedance probability (%)")

axC.grid(False)
axC.set_ylim(bottom=0)
axC_right.set_ylim(bottom=0)

set_axis_color(axC, COLOR_TCHW, side="left")
set_axis_color(axC_right, COLOR_ALL, side="right")

# 左侧坐标轴只保留红色刻度和红色轴线，但轴名改回黑色
axC.tick_params(axis="y", colors=COLOR_TCHW)
axC.spines["left"].set_color(COLOR_TCHW)
axC.yaxis.label.set_color("black")

axC.legend(
    handles=[h_all, h_tchw],
    labels=["All TC", "TC-HW"],
    frameon=False,
    loc="center left",
    prop=arial_font
)


# ============================================================
# Panel d: hot-day proportion, Early vs Late
# ============================================================

for pname in period_names:
    d = stats_period[
        stats_period["period"] == pname
    ].sort_values("lag_hour")

    line_with_ci(
        axD,
        d,
        "lag_hour",
        "hot_prop",
        "hot_prop_lo",
        "hot_prop_hi",
        PERIOD_COLORS[pname],
        PERIOD_LABELS[pname],
        yscale=100
    )

axD.set_xlim(12, 180)
axD.set_xticks(HOUR_TICKS)
axD.set_xlabel("Time after TC landfall (hours)")
axD.set_ylabel("Proportion of $T_{\\max}>T_{90}$ (%)")
axD.set_ylim(bottom=0)
axD.grid(False)
axD.legend(frameon=False, loc="lower right", prop=arial_font)


# ============================================================
# Panel e: 3-day exceedance, Early vs Late
# ============================================================

for pname in period_names:
    d = stats_3d_period[
        stats_3d_period["period"] == pname
    ].sort_values("start_hour")

    line_with_ci(
        axE,
        d,
        "start_hour",
        "p_exceed3",
        "p_exceed3_lo",
        "p_exceed3_hi",
        PERIOD_COLORS[pname],
        PERIOD_LABELS[pname],
        yscale=100
    )

axE.set_xlim(12, 132)
axE.set_xticks(HOUR_TICKS_3D)
axE.set_xlabel("Time after TC landfall (hours)")
axE.set_ylabel("3-day exceedance\nprobability (%)")
axE.grid(False)
axE.legend(frameon=False, loc="center left", bbox_to_anchor=(0.02, 0.67), prop=arial_font)


# ============================================================
# Panel labels and frames
# ============================================================

for ax, lab in zip([axB, axC, axD, axE], ["(b)", "(c)", "(d)", "(e)"]):
    add_panel_label(ax, lab)

for ax in [axA, axB, axC, axD, axE]:
    set_frame(ax)

set_frame(axC_right)

plt.show()


# ============================================================
# 8) Diagnostics
# ============================================================

print("\n========================================")
print("Quick diagnostics: All TC vs TC-HW")
print("========================================")

for h in HOUR_TICKS_3D:
    d_all = stats_3d_group[
        (stats_3d_group["group"] == "All TC")
        & (stats_3d_group["start_hour"] == h)
    ]

    d_hw = stats_3d_group[
        (stats_3d_group["group"] == "TC-HW")
        & (stats_3d_group["start_hour"] == h)
    ]

    if not d_all.empty and not d_hw.empty:
        print(
            f"Start hour {h}: "
            f"All TC = {d_all['p_exceed3'].iloc[0] * 100:.2f}% | "
            f"TC-HW = {d_hw['p_exceed3'].iloc[0] * 100:.2f}%"
        )

print("\n========================================")
print("Quick diagnostics: Early vs Late")
print("========================================")

for h in HOUR_TICKS:
    d_early = stats_period[
        (stats_period["period"] == "Early")
        & (stats_period["lag_hour"] == h)
    ]

    d_late = stats_period[
        (stats_period["period"] == "Late")
        & (stats_period["lag_hour"] == h)
    ]

    if not d_early.empty and not d_late.empty:
        hot_early = d_early["hot_prop"].iloc[0] * 100
        hot_late = d_late["hot_prop"].iloc[0] * 100

        print(
            f"Hour {h}: "
            f"Hot-day Early = {hot_early:.2f}% | "
            f"Late = {hot_late:.2f}% | "
            f"Difference = {hot_late - hot_early:+.2f} pp"
        )

print("\nDONE.")