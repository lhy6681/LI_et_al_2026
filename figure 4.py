import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t


# =========================================================
# 0) PATH
# =========================================================
XLSX_PATH = (
    r"G:\LHY\[new]\sensitive"
    r"\event_level_energy_loss_multi_recovery_only_Pdam0.9"
    r"_with_CDD_sensitive_within168h.xlsx"
)

XLSX_PATH = os.path.expanduser(XLSX_PATH)


# =========================================================
# 1) GLOBAL SETTINGS
# =========================================================
PERIODS = [
    ("1980–1999", 1980, 1999),
    ("2000–2024", 2000, 2024),
]

TARGET_LOSS_LEVELS = [20, 30, 40, 50, 60, 70, 80, 90]

TIME_COL = "landfall_time_utc"

N_BOOT = 1000
BOOT_RANDOM_SEED = 42

DPI = 800
FIGSIZE = (14, 6.2)

COLOR_1 = "#2166ac"
COLOR_2 = "#b2182b"

SHOW_CI_BAND = True
SHOW_IQR = True

X_OFFSET = 3.0
IQR_LINEWIDTH = 1.2
IQR_CAP_WIDTH = 2.0

# Inset positions
INSET_A_POS = [0.50, 0.22, 0.45, 0.34]
INSET_B_POS = [0.47, 0.16, 0.39, 0.25]

# Summary metric ranges
SUMMARY_A_R_MIN = 24
SUMMARY_A_R_MAX = 168

SUMMARY_B_L_MIN = 20
SUMMARY_B_L_MAX = 90

# Summary text positions
SUMMARY_A_TEXT_POS = (0.52, 0.63)
SUMMARY_B_TEXT_POS = (0.06, 0.60)

plt.rcParams["font.family"] = "Times New Roman"


# =========================================================
# 2) HELPER FUNCTIONS
# =========================================================
def mean_and_ci95(arr):
   
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]

    n = len(arr)

    if n == 0:
        return np.nan, np.nan, np.nan, 0

    mean = np.mean(arr)

    if n == 1:
        return mean, np.nan, np.nan, 1

    sd = np.std(arr, ddof=1)
    se = sd / np.sqrt(n)
    tcrit = t.ppf(0.975, n - 1)

    ci_low = mean - tcrit * se
    ci_high = mean + tcrit * se

    return mean, ci_low, ci_high, n


def q1_median_q3(arr):
   
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return np.nan, np.nan, np.nan

    return (
        np.nanpercentile(arr, 25),
        np.nanpercentile(arr, 50),
        np.nanpercentile(arr, 75),
    )


def percentile_ci(arr, alpha=0.05):
   
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return np.nan, np.nan

    return (
        np.nanpercentile(arr, 100 * alpha / 2),
        np.nanpercentile(arr, 100 * (1 - alpha / 2)),
    )


def beautify_spines(ax, linewidth=1.0):
   
    for spine in ax.spines.values():
        spine.set_linewidth(linewidth)


def draw_iqr_bars(
    ax,
    x,
    q1,
    q3,
    color,
    cap_width=2.0,
    linewidth=1.2,
    alpha=0.95,
    zorder=2,
):
    
    x = np.asarray(x, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    q3 = np.asarray(q3, dtype=float)

    valid = np.isfinite(x) & np.isfinite(q1) & np.isfinite(q3)

    for xi, lower, upper in zip(x[valid], q1[valid], q3[valid]):
        ax.vlines(
            xi,
            lower,
            upper,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
        )

        ax.hlines(
            lower,
            xi - cap_width,
            xi + cap_width,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
        )

        ax.hlines(
            upper,
            xi - cap_width,
            xi + cap_width,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
        )


def parse_year_from_timecol(series):
  
    series_datetime = pd.to_datetime(series, errors="coerce")

    if series_datetime.notna().sum() > 0:
        return series_datetime.dt.year.astype("Int64")

    series_string = series.astype(str)

    year = series_string.str.extract(
        r"((?:18|19|20)\d{2})",
        expand=False,
    )

    year = pd.to_numeric(year, errors="coerce")

    return year.astype("Int64")


def get_clean_series_percent(subdf, column):
   
    values = pd.to_numeric(subdf[column], errors="coerce")
    values = values[np.isfinite(values.values)]

    return values.values.astype(float) * 100.0


def monotone_non_decreasing(values):
   
    values = np.asarray(values, dtype=float).copy()

    if len(values) == 0:
        return values

    for i in range(1, len(values)):
        if np.isfinite(values[i - 1]):
            if not np.isfinite(values[i]) or values[i] < values[i - 1]:
                values[i] = values[i - 1]

    return values


def inverse_time_for_target_loss(hours, mean_curve, target_loss):
   
    x = np.asarray(hours, dtype=float)
    y = monotone_non_decreasing(
        np.asarray(mean_curve, dtype=float)
    )

    valid = np.isfinite(x) & np.isfinite(y)

    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        return np.nan

    if target_loss < np.nanmin(y) or target_loss > np.nanmax(y):
        return np.nan

    y_unique, unique_indices = np.unique(
        y,
        return_index=True,
    )

    x_unique = x[unique_indices]

    if len(y_unique) < 2:
        return np.nan

    return float(
        np.interp(
            target_loss,
            y_unique,
            x_unique,
        )
    )


def average_vertical_gap(x, y, xmin=None, xmax=None):
    
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)

    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        return np.nan

    sort_indices = np.argsort(x)

    x = x[sort_indices]
    y = y[sort_indices]

    if xmin is None:
        xmin = np.nanmin(x)

    if xmax is None:
        xmax = np.nanmax(x)

    if xmin >= xmax:
        return np.nan

    if xmin < np.nanmin(x) or xmax > np.nanmax(x):
        return np.nan

    x_augmented = x.copy()
    y_augmented = y.copy()

    if not np.isclose(xmin, x_augmented).any():
        y_xmin = np.interp(
            xmin,
            x_augmented,
            y_augmented,
        )

        x_augmented = np.append(
            x_augmented,
            xmin,
        )

        y_augmented = np.append(
            y_augmented,
            y_xmin,
        )

    if not np.isclose(xmax, x_augmented).any():
        y_xmax = np.interp(
            xmax,
            x_augmented,
            y_augmented,
        )

        x_augmented = np.append(
            x_augmented,
            xmax,
        )

        y_augmented = np.append(
            y_augmented,
            y_xmax,
        )

    sort_indices = np.argsort(x_augmented)

    x_augmented = x_augmented[sort_indices]
    y_augmented = y_augmented[sort_indices]

    within_range = (
        (x_augmented >= xmin)
        & (x_augmented <= xmax)
    )

    x_used = x_augmented[within_range]
    y_used = y_augmented[within_range]

    if len(x_used) < 2:
        return np.nan

    area = np.trapz(
        y_used,
        x_used,
    )

    return float(
        area / (xmax - xmin)
    )


def summarize_period(subdf, hours, loss_columns, period_label):
  
    means = []
    ci_lows = []
    ci_highs = []
    sample_sizes = []

    q1_values = []
    medians = []
    q3_values = []

    raw_values = []

    print(f"\n=== {period_label} ===")
    print(f"Rows in this period = {len(subdf)}")

    for hour in hours:
        values = get_clean_series_percent(
            subdf,
            loss_columns[hour],
        )

        raw_values.append(values)

        mean, ci_low, ci_high, sample_size = mean_and_ci95(values)
        q1, median, q3 = q1_median_q3(values)

        means.append(mean)
        ci_lows.append(ci_low)
        ci_highs.append(ci_high)
        sample_sizes.append(sample_size)

        q1_values.append(q1)
        medians.append(median)
        q3_values.append(q3)

    return {
        "means": np.array(means, dtype=float),
        "ci_lows": np.array(ci_lows, dtype=float),
        "ci_highs": np.array(ci_highs, dtype=float),
        "ns": np.array(sample_sizes, dtype=int),
        "q1": np.array(q1_values, dtype=float),
        "median": np.array(medians, dtype=float),
        "q3": np.array(q3_values, dtype=float),
        "raw": raw_values,
    }


def extract_matrix(subdf, hours, loss_columns):

    columns = [
        loss_columns[hour]
        for hour in hours
    ]

    matrix_df = (
        subdf[columns]
        .apply(pd.to_numeric, errors="coerce")
        * 100.0
    )

    valid_rows = np.all(
        np.isfinite(matrix_df.values),
        axis=1,
    )

    complete_matrix = matrix_df.loc[valid_rows].copy()

    return (
        complete_matrix.values.astype(float),
        valid_rows,
    )


def bootstrap_curves_from_matrix(
    matrix,
    n_boot=1000,
    seed=42,
):
   
    rng = np.random.default_rng(seed)

    n_rows, n_columns = matrix.shape

    bootstrap_means = np.full(
        (n_boot, n_columns),
        np.nan,
        dtype=float,
    )

    for bootstrap_index in range(n_boot):
        sampled_indices = rng.integers(
            0,
            n_rows,
            size=n_rows,
        )

        sampled_matrix = matrix[
            sampled_indices,
            :
        ]

        bootstrap_means[bootstrap_index, :] = np.nanmean(
            sampled_matrix,
            axis=0,
        )

    return bootstrap_means


# =========================================================
# 3) LOAD AND PREPARE DATA
# =========================================================
if not os.path.exists(XLSX_PATH):
    raise FileNotFoundError(
        f"Excel file not found:\n{XLSX_PATH}"
    )

print(f"Reading loss-ratio file:\n{XLSX_PATH}")

df = pd.read_excel(
    XLSX_PATH,
    engine="openpyxl",
)

if TIME_COL not in df.columns:
    raise RuntimeError(
        f"Column '{TIME_COL}' was not found in the Excel file."
    )


# Identify recovery-duration loss-ratio columns
loss_column_pattern = re.compile(
    r"^L_total_recover_(\d+)h$",
    re.IGNORECASE,
)

loss_columns = {}

for column in df.columns:
    match = loss_column_pattern.match(str(column))

    if match:
        recovery_hour = int(match.group(1))
        loss_columns[recovery_hour] = column

hours = sorted(loss_columns.keys())

if len(hours) == 0:
    raise RuntimeError(
        "No columns matching 'L_total_recover_XXh' were found."
    )


# Extract event year
df = df.copy()

df["__year__"] = parse_year_from_timecol(
    df[TIME_COL]
)

df = df[
    df["__year__"].notna()
].copy()

df["__year__"] = df["__year__"].astype(int)


# Calculate period-specific statistics
period_results = []
period_dataframes = []

for label, start_year, end_year in PERIODS:
    period_df = df[
        (df["__year__"] >= start_year)
        & (df["__year__"] <= end_year)
    ].copy()

    summary = summarize_period(
        period_df,
        hours,
        loss_columns,
        label,
    )

    period_results.append(
        (
            label,
            start_year,
            end_year,
            summary,
        )
    )

    period_dataframes.append(
        (
            label,
            period_df,
        )
    )


# Extract complete curves for bootstrapping
matrix_information = {}

for label, period_df in period_dataframes:
    matrix, _ = extract_matrix(
        period_df,
        hours,
        loss_columns,
    )

    if matrix.shape[0] < 5:
        raise RuntimeError(
            f"Too few complete-case rows in period {label}."
        )

    matrix_information[label] = {
        "matrix": matrix,
    }


# Bootstrap period-specific mean curves
bootstrap_curves = {}

for period_index, label in enumerate(matrix_information):
    bootstrap_curves[label] = bootstrap_curves_from_matrix(
        matrix_information[label]["matrix"],
        n_boot=N_BOOT,
        seed=BOOT_RANDOM_SEED
        + 1000 * (period_index + 1),
    )


# Separate early- and late-period results
x_hours = np.array(
    hours,
    dtype=float,
)

label_early, _, _, result_early = period_results[0]
label_late, _, _, result_late = period_results[1]

means_early = result_early["means"]
means_late = result_late["means"]

ci_early_low = result_early["ci_lows"]
ci_early_high = result_early["ci_highs"]

ci_late_low = result_late["ci_lows"]
ci_late_high = result_late["ci_highs"]

q1_early = result_early["q1"]
q3_early = result_early["q3"]

q1_late = result_late["q1"]
q3_late = result_late["q3"]

bootstrap_early = bootstrap_curves[label_early]
bootstrap_late = bootstrap_curves[label_late]


# =========================================================
# 4) DIFFERENCE IN MEAN LOSS RATIO
# =========================================================
difference_rows = []

for index, hour in enumerate(hours):
    early_values = result_early["raw"][index]
    late_values = result_late["raw"][index]

    early_mean = np.nanmean(early_values)
    late_mean = np.nanmean(late_values)

    difference = late_mean - early_mean

    bootstrap_difference = (
        bootstrap_late[:, index]
        - bootstrap_early[:, index]
    )

    difference_ci_low, difference_ci_high = percentile_ci(
        bootstrap_difference
    )

    difference_rows.append(
        {
            "recovery_hour": hour,
            "late_minus_early_pct_point": difference,
            "diff_ci_low": difference_ci_low,
            "diff_ci_high": difference_ci_high,
        }
    )

df_difference = pd.DataFrame(
    difference_rows
)


# =========================================================
# 5) EQUAL-LOSS RECOVERY DURATION
# =========================================================
equal_loss_rows = []

for target_loss in TARGET_LOSS_LEVELS:
    early_required_hour = inverse_time_for_target_loss(
        hours,
        means_early,
        target_loss,
    )

    late_required_hour = inverse_time_for_target_loss(
        hours,
        means_late,
        target_loss,
    )

    bootstrap_early_hour = np.array(
        [
            inverse_time_for_target_loss(
                hours,
                bootstrap_early[index, :],
                target_loss,
            )
            for index in range(N_BOOT)
        ],
        dtype=float,
    )

    bootstrap_late_hour = np.array(
        [
            inverse_time_for_target_loss(
                hours,
                bootstrap_late[index, :],
                target_loss,
            )
            for index in range(N_BOOT)
        ],
        dtype=float,
    )

    bootstrap_advance = (
        bootstrap_early_hour
        - bootstrap_late_hour
    )

    early_ci_low, early_ci_high = percentile_ci(
        bootstrap_early_hour
    )

    late_ci_low, late_ci_high = percentile_ci(
        bootstrap_late_hour
    )

    advance_ci_low, advance_ci_high = percentile_ci(
        bootstrap_advance
    )

    if (
        np.isfinite(early_required_hour)
        and np.isfinite(late_required_hour)
    ):
        advance_hour = (
            early_required_hour
            - late_required_hour
        )
    else:
        advance_hour = np.nan

    equal_loss_rows.append(
        {
            "target_loss_pct": target_loss,
            f"{label_early}_required_hour": early_required_hour,
            f"{label_early}_ci_low": early_ci_low,
            f"{label_early}_ci_high": early_ci_high,
            f"{label_late}_required_hour": late_required_hour,
            f"{label_late}_ci_low": late_ci_low,
            f"{label_late}_ci_high": late_ci_high,
            "advance_hour": advance_hour,
            "advance_ci_low": advance_ci_low,
            "advance_ci_high": advance_ci_high,
        }
    )

df_equal_loss = pd.DataFrame(
    equal_loss_rows
)

valid_equal_loss = (
    np.isfinite(
        df_equal_loss[
            f"{label_early}_required_hour"
        ].values
    )
    & np.isfinite(
        df_equal_loss[
            f"{label_late}_required_hour"
        ].values
    )
)

df_equal_loss = (
    df_equal_loss.loc[valid_equal_loss]
    .reset_index(drop=True)
)


# =========================================================
# 6) SUMMARY METRICS
# =========================================================
summary_a_value = average_vertical_gap(
    x=df_difference[
        "recovery_hour"
    ].values.astype(float),
    y=df_difference[
        "late_minus_early_pct_point"
    ].values.astype(float),
    xmin=SUMMARY_A_R_MIN,
    xmax=SUMMARY_A_R_MAX,
)

summary_b_value = average_vertical_gap(
    x=df_equal_loss[
        "target_loss_pct"
    ].values.astype(float),
    y=df_equal_loss[
        "advance_hour"
    ].values.astype(float),
    xmin=SUMMARY_B_L_MIN,
    xmax=SUMMARY_B_L_MAX,
)

print("\n==================== Summary metrics ====================")

print(
    "Panel (a): average vertical gap over recovery duration "
    f"{SUMMARY_A_R_MIN}–{SUMMARY_A_R_MAX} h = "
    f"{summary_a_value:.2f} percentage points"
)

print(
    "Panel (b): average vertical gap over target loss ratio "
    f"{SUMMARY_B_L_MIN}–{SUMMARY_B_L_MAX}% = "
    f"{summary_b_value:.2f} hours"
)


# =========================================================
# 7) PLOT
# =========================================================
fig, axes = plt.subplots(
    1,
    2,
    figsize=FIGSIZE,
    dpi=DPI,
)

plt.subplots_adjust(
    wspace=0.28,
)


# =========================================================
# PANEL (a)
# =========================================================
ax = axes[0]

x_early = x_hours - X_OFFSET
x_late = x_hours + X_OFFSET


# Confidence intervals
if SHOW_CI_BAND:
    valid_early_ci = (
        np.isfinite(x_hours)
        & np.isfinite(ci_early_low)
        & np.isfinite(ci_early_high)
    )

    if valid_early_ci.sum() >= 2:
        ax.fill_between(
            x_hours[valid_early_ci],
            ci_early_low[valid_early_ci],
            ci_early_high[valid_early_ci],
            color=COLOR_1,
            alpha=0.16,
            zorder=1,
        )

    valid_late_ci = (
        np.isfinite(x_hours)
        & np.isfinite(ci_late_low)
        & np.isfinite(ci_late_high)
    )

    if valid_late_ci.sum() >= 2:
        ax.fill_between(
            x_hours[valid_late_ci],
            ci_late_low[valid_late_ci],
            ci_late_high[valid_late_ci],
            color=COLOR_2,
            alpha=0.16,
            zorder=1,
        )


# Interquartile ranges
if SHOW_IQR:
    draw_iqr_bars(
        ax,
        x_early,
        q1_early,
        q3_early,
        color=COLOR_1,
        cap_width=IQR_CAP_WIDTH,
        linewidth=IQR_LINEWIDTH,
        alpha=0.95,
        zorder=2,
    )

    draw_iqr_bars(
        ax,
        x_late,
        q1_late,
        q3_late,
        color=COLOR_2,
        cap_width=IQR_CAP_WIDTH,
        linewidth=IQR_LINEWIDTH,
        alpha=0.95,
        zorder=2,
    )


# Mean curves
ax.plot(
    x_hours,
    means_early,
    color=COLOR_1,
    marker="o",
    markersize=4.8,
    linewidth=1.5,
    label=label_early,
    zorder=4,
)

ax.plot(
    x_hours,
    means_late,
    color=COLOR_2,
    marker="o",
    markersize=4.8,
    linewidth=1.5,
    label=label_late,
    zorder=4,
)


ax.text(
    0.07,
    0.97,
    "(a)",
    transform=ax.transAxes,
    fontsize=20,
    fontweight="bold",
    ha="right",
    va="top",
    zorder=10,
)

ax.set_xlabel(
    "Allowable recovery duration scenario (hours)",
    fontsize=18,
)

ax.set_ylabel(
    "Mean loss ratio during HW period (%)",
    fontsize=18,
)

ax.set_xticks(
    x_hours
)

ax.set_xticklabels(
    [f"{int(hour)}" for hour in x_hours],
    rotation=90,
    ha="center",
)


# Automatically determine the y-axis range
y_candidates = []

for values in [
    means_early,
    means_late,
    ci_early_low,
    ci_early_high,
    ci_late_low,
    ci_late_high,
    q1_early,
    q3_early,
    q1_late,
    q3_late,
]:
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) > 0:
        y_candidates.append(values)

if len(y_candidates) > 0:
    all_y_values = np.concatenate(
        y_candidates
    )

    y_min = np.nanmin(all_y_values)
    y_max = np.nanmax(all_y_values)

    y_range = y_max - y_min

    if not np.isfinite(y_range) or y_range <= 0:
        y_range = max(
            abs(y_max),
            1.0,
        )

    y_min_plot = y_min - 0.05 * y_range
    y_max_plot = y_max + 0.05 * y_range

    if y_min_plot > 0:
        y_min_plot = max(
            0,
            y_min - 0.06 * y_range,
        )

    ax.set_ylim(
        y_min_plot,
        y_max_plot,
    )


ax.axhline(
    y=0,
    color="gray",
    linestyle="--",
    linewidth=0.4,
    zorder=0,
)

ax.grid(False)

LEGEND_A_POS = (0.025, 0.895)
LEGEND_B_POS = (0.025, 0.895)

ax.legend(
    frameon=False,
    loc="upper left",
    bbox_to_anchor=LEGEND_A_POS,
    borderaxespad=0,
    handlelength=0.8,
    handletextpad=0.5,
    borderpad=0.2,
    labelspacing=0.25,
    fontsize=16,
)

beautify_spines(
    ax,
    1.1,
)


# Summary annotation
ax.text(
    SUMMARY_A_TEXT_POS[0],
    SUMMARY_A_TEXT_POS[1],
    f"Average increase = {summary_a_value:.1f}%",
    transform=ax.transAxes,
    fontsize=18,
    ha="left",
    va="center",
)


# Inset for panel (a)
ax_inset_a = ax.inset_axes(
    INSET_A_POS
)

df_difference_inset = df_difference[
    (
        df_difference["recovery_hour"]
        >= SUMMARY_A_R_MIN
    )
    & (
        df_difference["recovery_hour"]
        <= SUMMARY_A_R_MAX
    )
].copy()

x_inset_a = df_difference_inset[
    "recovery_hour"
].values.astype(float)

y_inset_a = df_difference_inset[
    "late_minus_early_pct_point"
].values.astype(float)

if len(x_inset_a) >= 2:
    inset_spacing = np.median(
        np.diff(x_inset_a)
    )

    inset_bar_width_reference = (
        0.7 * inset_spacing
    )
else:
    inset_bar_width_reference = 12.0

bars_a = ax_inset_a.bar(
    x_inset_a,
    y_inset_a,
    width=12,
    color="k",
    alpha=0.25,
    align="center",
)

ax_inset_a.axhline(
    0,
    color="gray",
    linewidth=0.45,
    linestyle="--",
)

ax_inset_a.set_xlim(
    SUMMARY_A_R_MIN
    - inset_bar_width_reference,
    SUMMARY_A_R_MAX
    + inset_bar_width_reference,
)

ax_inset_a.set_ylim(
    0,
    30,
)

ax_inset_a.set_xticks(
    [24, 48, 72, 96, 120, 144, 168]
)

ax_inset_a.tick_params(
    axis="x",
    labelsize=16,
    rotation=90,
    pad=1,
)

ax_inset_a.tick_params(
    axis="y",
    labelsize=16,
    pad=1,
)

ax_inset_a.set_ylabel(
    "Difference in\nmean loss ratio (%)",
    fontsize=16,
    labelpad=1,
)

ax_inset_a.set_xlabel(
    "Recovery duration (hours)",
    fontsize=16,
    labelpad=3,
)


for rectangle, value in zip(
    bars_a,
    y_inset_a,
):
    if np.isfinite(value):
        if value >= 0:
            text_y = min(
                rectangle.get_height() + 0.4,
                29.2,
            )

            vertical_alignment = "bottom"

        else:
            text_y = (
                rectangle.get_height()
                - 0.4
            )

            vertical_alignment = "top"

        ax_inset_a.text(
            rectangle.get_x()
            + rectangle.get_width() / 2,
            text_y,
            f"{value:.1f}",
            ha="center",
            va=vertical_alignment,
            fontsize=16,
        )


ax_inset_a.grid(False)

beautify_spines(
    ax_inset_a,
    0.8,
)


# =========================================================
# PANEL (b)
# =========================================================
ax = axes[1]

x_target_loss = df_equal_loss[
    "target_loss_pct"
].values.astype(float)

early_required_hours = df_equal_loss[
    f"{label_early}_required_hour"
].values.astype(float)

late_required_hours = df_equal_loss[
    f"{label_late}_required_hour"
].values.astype(float)


ax.plot(
    x_target_loss,
    early_required_hours,
    color=COLOR_1,
    marker="o",
    markersize=4.8,
    linewidth=1.5,
    label=label_early,
)

ax.plot(
    x_target_loss,
    late_required_hours,
    color=COLOR_2,
    marker="o",
    markersize=4.8,
    linewidth=1.5,
    label=label_late,
)


# Confidence intervals
for index in range(len(df_equal_loss)):
    early_low = df_equal_loss.loc[
        index,
        f"{label_early}_ci_low",
    ]

    early_high = df_equal_loss.loc[
        index,
        f"{label_early}_ci_high",
    ]

    late_low = df_equal_loss.loc[
        index,
        f"{label_late}_ci_low",
    ]

    late_high = df_equal_loss.loc[
        index,
        f"{label_late}_ci_high",
    ]

    if (
        np.isfinite(early_low)
        and np.isfinite(early_high)
    ):
        ax.vlines(
            x_target_loss[index],
            early_low,
            early_high,
            color=COLOR_1,
            linewidth=1.1,
            alpha=0.9,
        )

        ax.hlines(
            [early_low, early_high],
            x_target_loss[index] - 0.4,
            x_target_loss[index] + 0.4,
            color=COLOR_1,
            linewidth=1.1,
            alpha=0.9,
        )

    if (
        np.isfinite(late_low)
        and np.isfinite(late_high)
    ):
        ax.vlines(
            x_target_loss[index],
            late_low,
            late_high,
            color=COLOR_2,
            linewidth=1.1,
            alpha=0.9,
        )

        ax.hlines(
            [late_low, late_high],
            x_target_loss[index] - 0.4,
            x_target_loss[index] + 0.4,
            color=COLOR_2,
            linewidth=1.1,
            alpha=0.9,
        )


ax.text(
    0.07,
    0.97,
    "(b)",
    transform=ax.transAxes,
    fontsize=20,
    fontweight="bold",
    ha="right",
    va="top",
    zorder=10,
)

ax.set_xlabel(
    "Target mean loss ratio during HW period (%)",
    fontsize=18,
)

ax.set_ylabel(
    "Allowable recovery duration (hours)",
    fontsize=18,
)

ax.legend(
    frameon=False,
    loc="upper left",
    bbox_to_anchor=LEGEND_B_POS,
    borderaxespad=0,
    handlelength=0.8,
    handletextpad=0.5,
    borderpad=0.2,
    labelspacing=0.25,
    fontsize=16,
)

ax.grid(False)

beautify_spines(
    ax,
    1.1,
)


# Summary annotation
ax.text(
    SUMMARY_B_TEXT_POS[0],
    SUMMARY_B_TEXT_POS[1],
    f"Average shortening = {summary_b_value:.1f} h",
    transform=ax.transAxes,
    fontsize=18,
    ha="left",
    va="center",
)


# Inset for panel (b)
ax_inset_b = ax.inset_axes(
    INSET_B_POS
)

x_inset_b = df_equal_loss[
    "target_loss_pct"
].values.astype(float)

y_inset_b = df_equal_loss[
    "advance_hour"
].values.astype(float)

bars_b = ax_inset_b.bar(
    x_inset_b,
    y_inset_b,
    width=5.5,
    color="k",
    alpha=0.25,
    align="center",
)

ax_inset_b.axhline(
    0,
    color="gray",
    linewidth=0.45,
    linestyle="--",
)

ax_inset_b.set_ylim(
    0,
    60,
)

ax_inset_b.set_xticks(
    x_inset_b
)

ax_inset_b.tick_params(
    axis="x",
    labelsize=16,
    rotation=0,
    pad=1,
)

ax_inset_b.tick_params(
    axis="y",
    labelsize=16,
    pad=1,
)

ax_inset_b.set_ylabel(
    "Advance in recovery\nduration (hours)",
    fontsize=16,
    labelpad=1,
)

ax_inset_b.set_xlabel(
    "Target loss ratio (%)",
    fontsize=16,
    labelpad=3,
)


# Move the inset y-axis labels to the right
ax_inset_b.yaxis.set_label_position(
    "right"
)

ax_inset_b.yaxis.tick_right()

ax_inset_b.tick_params(
    axis="y",
    labelright=True,
    labelleft=False,
)

ax_inset_b.spines[
    "right"
].set_visible(True)

ax_inset_b.spines[
    "left"
].set_visible(True)


for rectangle, value in zip(
    bars_b,
    y_inset_b,
):
    if np.isfinite(value):
        if value >= 0:
            text_y = (
                rectangle.get_height()
                + 0.4
            )

            vertical_alignment = "bottom"

        else:
            text_y = (
                rectangle.get_height()
                - 0.4
            )

            vertical_alignment = "top"

        ax_inset_b.text(
            rectangle.get_x()
            + rectangle.get_width() / 2,
            text_y,
            f"{value:.0f}",
            ha="center",
            va=vertical_alignment,
            fontsize=16,
        )


ax_inset_b.grid(False)

beautify_spines(
    ax_inset_b,
    0.8,
)


# =========================================================
# 8) FINAL LAYOUT
# =========================================================
for axis in axes:
    beautify_spines(
        axis,
        1.15,
    )

plt.tight_layout()
plt.show()

print("\nDONE.")