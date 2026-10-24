import os
import numpy as np
import pandas as pd

# =========================================================
# 0) User settings (EDIT)
# =========================================================
IN_CSV = r"G:\LHY\[new]\TC-HW compound\TC-HW\compound_events_post7d_hourly_mincols__no_small_islands.csv"

# Output: create a new file next to the input (do NOT overwrite by default)
# Keep xlsx as in your original style; you can change to .csv if you want.
OUT_XLSX = r"G:\LHY\[new]\TC-HW compound\TC-HW\compound_events_post7d_hourly_mincols__no_small_islands__with_Pdam.xlsx"

POST_VALUE = "post"  # if rel_to_landfall exists, keep only this (case-insensitive)

# =========================================================
# 1) Constants & DS1 fragility curve (UNCHANGED)
# =========================================================
ALPHA = 1.0 / 7.0
H_TGT = 80.0
H_REF = 10.0
SCALE_WIND_80_10 = (H_TGT / H_REF) ** ALPHA

ws_frag = np.array([20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50], dtype=float)
p_frag  = np.array([0,  0,  0,  0,  0,  0.02, 0.10, 0.29, 0.50, 0.77, 0.90, 0.98, 1.0, 1.0, 1.0, 1.0], dtype=float)

def classify_ss_from_ms(vmax_ms: float) -> float:
   
    if not np.isfinite(vmax_ms):
        return np.nan
    vmax_ms = float(vmax_ms)
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

def wind_to_damage_prob(v80: float) -> float:
    
    if not np.isfinite(v80):
        return np.nan
    return float(np.interp(v80, ws_frag, p_frag, left=p_frag[0], right=p_frag[-1]))

# =========================================================
# 2) Robust helpers
# =========================================================
def _norm_sid(s):
    return str(s).strip()

def _lower_map(cols):
    
    m = {}
    for c in cols:
        m[str(c).lower().strip()] = c
    return m

def _pick_case_insensitive(cols, candidates_lower):
   
    lm = _lower_map(cols)
    for cand in candidates_lower:
        if cand in lm:
            return lm[cand]
    return None

def _read_csv_robust(csv_path: str) -> pd.DataFrame:
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    # Try common encodings
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            return df
        except UnicodeDecodeError:
            continue

    # Last resort: let pandas guess (may still fail)
    return pd.read_csv(csv_path)

def _write_out(df: pd.DataFrame, out_path: str):
    
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    ext = os.path.splitext(out_path)[1].lower()
    if ext == ".csv":
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
    else:
        df.to_excel(out_path, index=False, engine="openpyxl")

# =========================================================
# 3) Load TC-HW events from ONE CSV (and detect key columns)
# =========================================================
def load_events_from_single_csv(csv_path):
    df = _read_csv_robust(csv_path)

    if df.empty:
        raise RuntimeError(f"CSV is empty: {csv_path}")

    # --- detect sid column ---
    sid_col = _pick_case_insensitive(df.columns, ["sid", "storm_id", "stormid", "tc_id", "tcid"])
    if sid_col is None:
        raise KeyError(f"Cannot find SID column in CSV. Tried: sid/storm_id/tc_id. Columns: {list(df.columns)}")

    # --- detect landfall time column ---
    lf_time_col = _pick_case_insensitive(df.columns, [
        "landfall_time_utc", "landfall_time", "landfall_utc", "lf_time_utc", "time_landfall_utc"
    ])

    # --- detect rel_to_landfall (optional) ---
    rel_col = _pick_case_insensitive(df.columns, ["rel_to_landfall", "rel2landfall", "rel"])

    # --- detect landfall-day v10 column (PRIORITY) ---
    v10_col = _pick_case_insensitive(df.columns, [
        "vmax_landfall_day_1min",
        "vmax_landfall_day_1min_ms",
        "vmax_landfall_day",
        "vmax_lfday_1min",
        "landfall_day_vmax_1min",
        "landfall_day_wind_1min_ms"
    ])

    if v10_col is None:
        v10_col = _pick_case_insensitive(df.columns, [
            "wind_1min_ms", "vmax_1min_ms", "vmax_1min", "wind_ms", "wind"
        ])

    if v10_col is None:
        raise KeyError(
            "Cannot find landfall-day 10m 1-min wind column (v10) in the CSV.\n"
            "Please ensure the CSV contains one of these columns:\n"
            "  vmax_landfall_day_1min / vmax_landfall_day_1min_ms / wind_1min_ms / vmax_1min_ms / ...\n"
            f"Available columns: {list(df.columns)}"
        )

    # --- detect lifetime vmax column (optional, for tc_cat) ---
    life_col = _pick_case_insensitive(df.columns, ["vmax_lifetime_1min", "lifetime_vmax_1min", "vmax_life_1min"])

    # --- normalize & filter ---
    out = df.copy()
    out[sid_col] = out[sid_col].astype(str).map(_norm_sid)

    if lf_time_col is not None:
        out[lf_time_col] = pd.to_datetime(out[lf_time_col], errors="coerce")

    # rel_to_landfall filter if exists
    if rel_col is not None:
        rel = out[rel_col].astype(str).str.lower().str.strip()
        out = out[rel == POST_VALUE].copy()

    # ensure v10 numeric
    out[v10_col] = pd.to_numeric(out[v10_col], errors="coerce")

    # lifetime vmax numeric (optional)
    if life_col is not None:
        out[life_col] = pd.to_numeric(out[life_col], errors="coerce")

    meta = {
        "sid_col": sid_col,
        "landfall_time_col": lf_time_col,
        "rel_col": rel_col,
        "v10_col": v10_col,
        "life_col": life_col
    }
    return out, meta

# =========================================================
# 4) Main
# =========================================================
def main():
    print("1) Loading TC-HW events from ONE CSV ...")
    df_evt, meta = load_events_from_single_csv(IN_CSV)

    print(f"   - sid column: {meta['sid_col']}")
    print(f"   - landfall time column: {meta['landfall_time_col']}")
    print(f"   - rel_to_landfall column: {meta['rel_col']}")
    print(f"   - v10 (landfall-day 10m 1-min wind) column: {meta['v10_col']}")
    print(f"   - lifetime vmax column: {meta['life_col']}")
    print(f"   - rows after POST filter (if any): {len(df_evt)}")

    if df_evt.empty:
        raise RuntimeError("No rows left after loading/filtering. Check rel_to_landfall values or CSV content.")

    # coverage check for v10
    v10 = pd.to_numeric(df_evt[meta["v10_col"]], errors="coerce").to_numpy(dtype=float)
    n_has_v10 = int(np.isfinite(v10).sum())
    print(f"2) v10 coverage: {n_has_v10}/{len(df_evt)} = {n_has_v10/max(1,len(df_evt))*100:.1f}%")
    if n_has_v10 == 0:
        raise RuntimeError("No valid v10 values found in the CSV. Cannot compute Pdam.")

    print("3) Computing Pdam (UNCHANGED logic: v10 -> v80 -> DS1 interp) ...")
    v80 = v10 * SCALE_WIND_80_10
    pdam = np.array([wind_to_damage_prob(x) for x in v80], dtype=float)

    # Optional: compute tc_cat from lifetime vmax if present
    if meta["life_col"] is not None:
        life_v10 = pd.to_numeric(df_evt[meta["life_col"]], errors="coerce").to_numpy(dtype=float)
        tc_cat = np.array([classify_ss_from_ms(x) for x in life_v10], dtype=float)
    else:
        tc_cat = None

    # Append new columns to the dataframe (keeping all original columns)
    # => This automatically keeps vmax_landfall_day_1min and vmax_lifetime_1min if they exist in your CSV.
    df_out = df_evt.copy()
    df_out["vmax_landfall_day_80m"] = v80  # helpful for sanity check (you can delete if you want)
    df_out["Pdam"] = pdam

    # optional extra columns (kept consistent with your old spirit, but not required)
    df_out["has_damage_Pdam_gt0"] = (np.isfinite(pdam) & (pdam > 0)).astype(np.int8)
    df_out["Pdam_isfinite"] = np.isfinite(pdam).astype(np.int8)
    if tc_cat is not None:
        df_out["tc_cat_from_lifetime_v10"] = tc_cat

    # quick sanity prints
    print("4) Sanity check (first 5 rows):")
    cols_show = [meta["sid_col"], meta["v10_col"]]

    # explicitly show these two if present (for your peace of mind)
    if "vmax_landfall_day_1min" in df_out.columns:
        cols_show.append("vmax_landfall_day_1min")
    if "vmax_lifetime_1min" in df_out.columns:
        cols_show.append("vmax_lifetime_1min")

    cols_show += ["vmax_landfall_day_80m", "Pdam", "has_damage_Pdam_gt0"]
    cols_show = [c for c in cols_show if c in df_out.columns]
    print(df_out[cols_show].head(5).to_string(index=False))

    print("5) Writing output ...")
    _write_out(df_out, OUT_XLSX)
    print(f"[SAVED] Output with Pdam -> {OUT_XLSX}")

if __name__ == "__main__":
    main()
