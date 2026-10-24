import os
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

# =========================
# 路径 & 参数设置
# =========================
XLSX_PATH = r"H:\LHY\[new]\TC-HW compound\TC-HW\compound_events_post7d_from_post30d__no_small_islands__with_Pdam.xlsx"
POT_NC_PATH = r"H:\LHY\[new]\supply--potential wind power\TC_HW_wind_P_pot_pm7.nc"

OUT_DIR = r"H:\LHY\[new]\supply and demand\supply"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_NC = os.path.join(OUT_DIR, "TC_HW_wind_power_allinone__7windows_only_Pdam0.9--with 0h shutdown.nc")

# ---- 台风窗口 ----
TC_WINDOW_DAYS = 1.0

# =========================
# 常数（仅用于 10m→80m 外推、SS 分类等；本次不再用 SS 决定停机，但变量仍保留兼容）
# =========================
ALPHA = 1.0 / 7.0
H_TGT = 80.0
H_GUST_REF = 10.0
SCALE_GUST_80_10 = (H_TGT / H_GUST_REF) ** ALPHA
SCALE_WIND_80_10 = SCALE_GUST_80_10

# =========================
# 风控阈值（保留常量定义以保持脚本结构兼容；但 storm 策略已取消，不再使用）
# =========================
STORM_CUT_OUT = 25.0
STORM_RE_CUT_IN = 20.0
GUST_CUT_OUT = 35.0

# =========================
# ✅ 固定停机/恢复（仅对 Pdam≥0.9 的事件）
# =========================
FIXED_PDAM_THRESHOLD = 0.9
FIXED_SHUTDOWN_HOURS = 0.0   # 从 landfall 开始先停机 0h
FIXED_RECOVERY_HOURS = 72.0   # 停机结束后恢复 72h（你强调：指数型先慢后快 + 72h 必须满恢复）

# storm 参数保留但不再使用（为了尽量不改变量体系）
STORM_LOCK_HOURS = 6.0

# 取消旧的“基础锁机 6h”等逻辑（保持与原脚本结构兼容，但不再参与）
BASE_LOCK_HOURS = 0.0
PRE_SHUT_HOURS = 0.0

# =========================
# ✅ 先慢后快的“指数型”恢复形状参数
# k 越大：越“前期更平、后期更陡”（更明显先慢后快）
# =========================
RECOVERY_EXP_K = 5.0

def exp_ease_in_progress(s: np.ndarray, k: float) -> np.ndarray:
   
    s = np.asarray(s, dtype=float)
    s = np.clip(s, 0.0, 1.0)
    if np.isfinite(k) and abs(k) > 1e-12:
        denom = np.expm1(k)  # exp(k)-1
        # 数值安全：如果 denom 极小（理论上不会），退化为线性
        if abs(denom) < 1e-12:
            return s
        return np.expm1(k * s) / denom
    else:
        return s

# =========================
# 通用工具函数
# =========================
def normalize_lon_to_minus180_180(lon_val: float) -> float:
    return float(((lon_val + 180) % 360) - 180)

def classify_ss_from_ms(vmax_ms: float) -> float:
   
    if not np.isfinite(vmax_ms):
        return np.nan
    if vmax_ms < 33:
        return 0.0  # TD/TS
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

# =========================
# 1. 读取 Excel & 潜在发电量库 NC
# =========================
print(f"Reading compound events from Excel: {XLSX_PATH}")
df_all = pd.read_excel(XLSX_PATH, engine="openpyxl")
df_all = df_all.dropna(subset=["sid"]).copy()
df_all = df_all.reset_index(drop=True)
if df_all.empty:
    raise RuntimeError("Excel contains no valid events (after dropping empty sid).")

# 必需列检查
required_cols = [
    "sid", "grid_lat", "grid_lon", "start_clip", "end_clip",
    "vmax_landfall_day_1min", "vmax_lifetime_1min", "Pdam"
]
missing = [c for c in required_cols if c not in df_all.columns]
if missing:
    raise RuntimeError(f"Excel missing required columns: {missing}")

# ✅ 仅保留 Pdam≥0.9
pdam_all = pd.to_numeric(df_all["Pdam"], errors="coerce")
sel_mask = np.isfinite(pdam_all) & (pdam_all >= FIXED_PDAM_THRESHOLD)
sel_indices = np.where(sel_mask.to_numpy())[0].astype(int)

print(f"Total events in Excel: {len(df_all)}")
print(f"Selected events with Pdam >= {FIXED_PDAM_THRESHOLD}: {len(sel_indices)}")

if len(sel_indices) == 0:
    raise RuntimeError(f"No events satisfy Pdam >= {FIXED_PDAM_THRESHOLD}. Nothing to do.")

df = df_all.iloc[sel_indices].copy().reset_index(drop=True)

n_events = len(df)
print(f"Number of events to process (Pdam>=0.9): {n_events}")

sid_arr = df["sid"].astype(str).to_numpy()
lat_arr = pd.to_numeric(df["grid_lat"], errors="coerce").to_numpy(dtype=float)
lon_arr = pd.to_numeric(df["grid_lon"], errors="coerce").to_numpy(dtype=float)

start_arr = pd.to_datetime(df["start_clip"], errors="coerce")
end_arr   = pd.to_datetime(df["end_clip"], errors="coerce")

# 登陆时间列：Excel 提供 landfall_time_utc
if "landfall_time_utc" in df.columns:
    print("Using 'landfall_time_utc' as landfall time.")
    landfall_arr = pd.to_datetime(df["landfall_time_utc"], errors="coerce")
else:
    print("WARNING: 'landfall_time_utc' not found; using 'start_clip' as landfall time surrogate.")
    landfall_arr = start_arr.copy()

landfall_vmax_10m_ms_arr = pd.to_numeric(df["vmax_landfall_day_1min"], errors="coerce").to_numpy(dtype=float)
lifetime_vmax_10m_ms_arr = pd.to_numeric(df["vmax_lifetime_1min"], errors="coerce").to_numpy(dtype=float)

# ✅ 直接读 Excel 的 Pdam（筛选后）
pdam_excel_arr = pd.to_numeric(df["Pdam"], errors="coerce").to_numpy(dtype=float)

# SS 等级（基于生命周期最大风）——不用于停机，仅保留输出
tc_category_ss_arr = np.array([classify_ss_from_ms(v) for v in lifetime_vmax_10m_ms_arr], dtype="float32")

print(f"Opening precomputed potential-power library: {POT_NC_PATH}")
ds_pot_all = xr.open_dataset(POT_NC_PATH)

if "event" not in ds_pot_all.dims:
    raise RuntimeError("Potential-power library NC missing 'event' dimension.")

n_events_pot = int(ds_pot_all.dims["event"])
if n_events_pot < int(sel_indices.max()) + 1:
    raise RuntimeError(
        f"Potential-power NC has only {n_events_pot} events, but selected index max is {int(sel_indices.max())}."
    )

# 提示一致性（不强制）
if "sid" in ds_pot_all:
    sid_pot_all = ds_pot_all["sid"].values.astype(str)
    try:
        sid_pot_sel = sid_pot_all[sel_indices]
        if sid_pot_sel.shape[0] == n_events and (not np.array_equal(sid_pot_sel, sid_arr.astype(str))):
            print("WARNING: sid mismatch between filtered Excel and potential-power NC at selected indices.")
            print("         Assuming event index mapping is still consistent.")
    except Exception:
        print("WARNING: Unable to validate sid alignment between Excel and NC.")

# =========================
# 2. 逐事件计算
# =========================
v80c_list = []
v80_list  = []
time_list = []
lengths   = []

v80_tcmax_list    = []
gust80_tcmax_list = []
P_dam_TC_list     = []
A_TC_event_list   = []

P_pot_list       = []
A_TC_time_list   = []
P_avail_TC_list  = []

run_flag_list                = []
shutdown_hours_list          = []
first_shutdown_rel_time_list = []

shutdown_flag_event_list           = []
shutdown_start_rel_time_event_list = []
lock_total_hours_event_list        = []
shutdown_by_damage_list            = []
shutdown_by_ss_list                = []
shutdown_by_storm_list             = []

for j in tqdm(range(n_events), desc="Processing events (Pdam>=0.9)", unit="event"):
    # 原始事件索引（用于从潜在发电量库读取）
    idx0 = int(sel_indices[j])

    sid = sid_arr[j]
    lat_ev = float(lat_arr[j])
    lon_ev_raw = float(lon_arr[j])
    lon_ev = normalize_lon_to_minus180_180(lon_ev_raw)

    hw_start_ts = start_arr.iloc[j]
    hw_end_ts   = end_arr.iloc[j]
    landfall_ts = landfall_arr.iloc[j]

    # === 2.0 从潜在发电量库取该事件的 V80/V80c/gust80/P_pot/time ===
    ds_ev = ds_pot_all.isel(event=idx0)

    time_full = ds_ev["time"].values
    valid_time_mask = ~np.isnat(time_full)
    if not np.any(valid_time_mask):
        continue

    time_vals = time_full[valid_time_mask]

    V80_vals_out  = ds_ev["V80"].values.astype("float32")[valid_time_mask]
    V80c_vals_out = ds_ev["V80c"].values.astype("float32")[valid_time_mask]
    gust_arr      = ds_ev["gust80"].values.astype("float32")[valid_time_mask]
    P_pot_vals    = ds_ev["P_pot"].values.astype("float32")[valid_time_mask]

    # 事件级 max（从全库按原索引取）
    v80_tcmax    = float(ds_pot_all["V80_TCmax"].values[idx0])
    gust80_tcmax = float(ds_pot_all["gust80_TCmax"].values[idx0])

    v80_tcmax_list.append(v80_tcmax)
    gust80_tcmax_list.append(gust80_tcmax)

    # === 2.1 ✅ 事件级 Pdam 直接来自 Excel（且已筛选≥0.9） ===
    reminder_pdam = pdam_excel_arr[j]
    P_dam_event = float(reminder_pdam) if np.isfinite(reminder_pdam) else np.nan
    if (not np.isfinite(P_dam_event)) or (P_dam_event < FIXED_PDAM_THRESHOLD):
        continue

    A_TC_event = 1.0 - P_dam_event

    # 固定停机与恢复参数
    damage_lock_hours = float(FIXED_SHUTDOWN_HOURS)   # 先停机 0h
    T_rec_hours = float(FIXED_RECOVERY_HOURS)         # 恢复 72h（停机结束后开始算）
    # 注：tau 不再用于本次“先慢后快”的指数型恢复（这里保留变量以尽量不破坏原结构）
    tau_hours = T_rec_hours / 3.0

    P_dam_TC_list.append(P_dam_event)
    A_TC_event_list.append(A_TC_event)

    # === 2.2 时间轴相对登陆日（小时/天） ===
    landfall_np = np.datetime64(landfall_ts)
    delta_hours = (time_vals - landfall_np) / np.timedelta64(1, "h")
    delta_days  = delta_hours / 24.0

    # =========================================================
    # ✅ storm 停机策略已取消：不触发、不延长停机窗口
    # =========================================================
    storm_triggered = False
    t_trigger = None

    # === 2.4 最终停机窗口：仅 landfall 固定停机 0h ===
    lock_start_time = landfall_np
    lock_end_time = landfall_np + np.timedelta64(int(np.round(damage_lock_hours)), "h")

    # 记录“锁机总时长”
    lock_total_hours = float((lock_end_time - lock_start_time) / np.timedelta64(1, "h"))
    lock_total_steps = int(np.floor(lock_total_hours)) if lock_total_hours > 0.0 else 0

    # 事件级标记（保持变量名体系不变）
    shutdown_flag_event = 1 if (lock_total_hours > 0.0) else 0
    shutdown_start_rel_time_event = 0.0 if shutdown_flag_event == 1 else np.nan
    shutdown_by_damage = 1
    shutdown_by_ss     = 0
    shutdown_by_storm  = 0  # ✅ storm 策略已取消

    shutdown_flag_event_list.append(shutdown_flag_event)
    shutdown_start_rel_time_event_list.append(shutdown_start_rel_time_event)
    lock_total_hours_event_list.append(lock_total_hours)
    shutdown_by_damage_list.append(shutdown_by_damage)
    shutdown_by_ss_list.append(shutdown_by_ss)
    shutdown_by_storm_list.append(shutdown_by_storm)

    # === 2.5 run_flag（停机窗口置 0）===
    nT = V80c_vals_out.shape[0]
    run_flag = np.ones(nT, dtype="float32")

    hours_since_landfall = (time_vals - landfall_np) / np.timedelta64(1, "h")
    lock_mask = (hours_since_landfall >= 0.0) & (hours_since_landfall < lock_total_hours)
    run_flag[lock_mask] = 0.0

    # =========================================================
    # === 2.6 A_TC_time：停机结束后开始恢复（72h）
    # ✅ 采用“指数型先慢后快”的加速恢复，并且在 t=72h 时强制满恢复（Pdam(t)=0）
    #
    # 设 s=t/T (0..1)，恢复进度：
    #   g(s) = (exp(k*s)-1)/(exp(k)-1)   (k>0)  -> 先慢后快
    # 可用度：
    #   A(t) = (1-Pdam) + Pdam * g(t/T)
    # 并且：
    #   t>=T -> A=1（不允许任何残余）
    # =========================================================
    recovery_start_time = lock_end_time
    t_since_rec_h = (time_vals - recovery_start_time) / np.timedelta64(1, "h")  # hours

    # 默认先设为 1（对停机前/停机中无影响，因为 run_flag 会为 0；保留原脚本风格）
    A_time = np.ones_like(t_since_rec_h, dtype=float)

    m_rec = t_since_rec_h >= 0.0
    if np.any(m_rec):
        t_eff = t_since_rec_h[m_rec].astype(float)

        # s in [0,1] for the 72h recovery window
        s = np.clip(t_eff / T_rec_hours, 0.0, 1.0)

        g = exp_ease_in_progress(s, RECOVERY_EXP_K)

        # A(t) from (1-Pdam) to 1, slow->fast
        A_temp = (1.0 - P_dam_event) + P_dam_event * g
        A_time[m_rec] = A_temp

        # ✅ 在恢复结束（停机结束后72h）时刻起：满恢复，严格 A=1（即 Pdam(t)=0）
        A_time = np.where(t_since_rec_h >= T_rec_hours, 1.0, A_time)

    # === 2.7 最终可用功率 ===
    P_avail_vals = (P_pot_vals * A_time * run_flag).astype("float32")

    shutdown_hours = int(np.sum(run_flag == 0.0))
    if shutdown_hours > 0:
        first_idx = int(np.where(run_flag == 0.0)[0][0])
        first_shutdown_rel_time = float(delta_days[first_idx])
    else:
        first_shutdown_rel_time = np.nan

    shutdown_hours_list.append(shutdown_hours)
    first_shutdown_rel_time_list.append(first_shutdown_rel_time)

    v80_list.append(V80_vals_out)
    v80c_list.append(V80c_vals_out)
    time_list.append(time_vals)
    lengths.append(len(time_vals))

    P_pot_list.append(P_pot_vals)
    A_TC_time_list.append(A_time.astype("float32"))
    P_avail_TC_list.append(P_avail_vals)
    run_flag_list.append(run_flag.astype("float32"))

# =========================
# 3. 不等长时序填充 & 全局标准化（不变）
# =========================
if len(lengths) == 0:
    raise RuntimeError("No valid events were processed (all skipped due to missing time, etc.).")

lengths = np.array(lengths, dtype=int)
max_len = int(lengths.max())
print(f"Max number of hours among processed events: {max_len}")

V80c_arr = np.full((n_events, max_len), np.nan, dtype="float32")
V80_arr  = np.full((n_events, max_len), np.nan, dtype="float32")
time_arr = np.full((n_events, max_len), np.datetime64("NaT"), dtype="datetime64[ns]")

P_pot_arr       = np.full((n_events, max_len), np.nan, dtype="float32")
P_avail_TC_arr  = np.full((n_events, max_len), np.nan, dtype="float32")
P_avail_std_arr = np.full((n_events, max_len), np.nan, dtype="float32")
A_TC_time_arr   = np.full((n_events, max_len), np.nan, dtype="float32")
run_flag_arr    = np.full((n_events, max_len), np.nan, dtype="float32")

for i in tqdm(range(n_events), desc="Stacking event arrays", unit="event"):
    L = lengths[i]
    V80c_arr[i, :L] = v80c_list[i]
    V80_arr[i, :L]  = v80_list[i]
    time_arr[i, :L] = time_list[i]

    P_pot_arr[i, :L]       = P_pot_list[i]
    P_avail_TC_arr[i, :L]  = P_avail_TC_list[i]
    A_TC_time_arr[i, :L]   = A_TC_time_list[i]
    run_flag_arr[i, :L]    = run_flag_list[i]

V80_TCmax_arr       = np.array(v80_tcmax_list, dtype="float32")
gust80_TCmax_arr    = np.array(gust80_tcmax_list, dtype="float32")
P_dam_TC_arr        = np.array(P_dam_TC_list, dtype="float32")
A_TC_event_arr      = np.array(A_TC_event_list, dtype="float32")
shutdown_hours_arr  = np.array(shutdown_hours_list, dtype="int32")
first_shutdown_rel_time_arr = np.array(first_shutdown_rel_time_list, dtype="float32")

shutdown_flag_event_arr           = np.array(shutdown_flag_event_list, dtype="int32")
shutdown_start_rel_time_event_arr = np.array(shutdown_start_rel_time_event_list, dtype="float32")
lock_total_hours_event_arr        = np.array(lock_total_hours_event_list, dtype="float32")
shutdown_by_damage_arr            = np.array(shutdown_by_damage_list, dtype="int32")
shutdown_by_ss_arr                = np.array(shutdown_by_ss_list, dtype="int32")
shutdown_by_storm_arr             = np.array(shutdown_by_storm_list, dtype="int32")

event_start_arr   = start_arr.to_numpy(dtype="datetime64[ns]")
event_end_arr     = end_arr.to_numpy(dtype="datetime64[ns]")
landfall_time_arr = landfall_arr.to_numpy(dtype="datetime64[ns]")

valid_all = np.isfinite(P_avail_TC_arr)
if np.any(valid_all):
    mu_all = float(P_avail_TC_arr[valid_all].mean())
    sigma_all = float(P_avail_TC_arr[valid_all].std())
    print(f"Global mean of P_avail_TC over all events/times: {mu_all:.6f} kW")
    print(f"Global std  of P_avail_TC over all events/times: {sigma_all:.6f} kW")

    if sigma_all > 0.0:
        P_avail_std_arr[valid_all] = ((P_avail_TC_arr[valid_all] - mu_all) / sigma_all).astype("float32")
    else:
        print("WARNING: Global std of P_avail_TC is zero; P_avail_std remains NaN.")
else:
    print("WARNING: No valid P_avail_TC values found; P_avail_std remains NaN.")

# =========================
# 4. 组装 xarray.Dataset 并写出（不变）
# =========================
event_idx = np.arange(n_events, dtype=int)

ds_out = xr.Dataset(
    data_vars={
        "V80c": (("event", "time"), V80c_arr),
        "V80":  (("event", "time"), V80_arr),
        "V80_TCmax":    (("event",), V80_TCmax_arr),
        "gust80_TCmax": (("event",), gust80_TCmax_arr),

        "P_pot":       (("event", "time"), P_pot_arr),
        "P_avail_TC":  (("event", "time"), P_avail_TC_arr),
        "P_avail_std": (("event", "time"), P_avail_std_arr),

        "P_dam_TC":   (("event",), P_dam_TC_arr),
        "A_TC":       (("event",), A_TC_event_arr),
        "A_TC_time":  (("event", "time"), A_TC_time_arr),
        "run_flag":   (("event", "time"), run_flag_arr),

        "shutdown_hours":          (("event",), shutdown_hours_arr),
        "first_shutdown_rel_time": (("event",), first_shutdown_rel_time_arr),

        "shutdown_flag_event":           (("event",), shutdown_flag_event_arr),
        "shutdown_start_rel_time_event": (("event",), shutdown_start_rel_time_event_arr),
        "lock_total_hours_event":        (("event",), lock_total_hours_event_arr),
        "shutdown_by_damage":            (("event",), shutdown_by_damage_arr),
        "shutdown_by_ss":                (("event",), shutdown_by_ss_arr),
        "shutdown_by_storm":             (("event",), shutdown_by_storm_arr),

        "sid": (("event",), sid_arr),
        "tc_category_ss": (("event",), tc_category_ss_arr),

        "crs": xr.DataArray(
            0,
            attrs={
                "grid_mapping_name": "latitude_longitude",
                "epsg_code": "EPSG:4326",
                "semi_major_axis": 6378137.0,
                "inverse_flattening": 298.257223563,
            },
        ),
    },
    coords={
        "event": event_idx,
        "time": (("event", "time"), time_arr),
        "grid_lat": (("event",), lat_arr.astype("float32")),
        "grid_lon": (("event",), lon_arr.astype("float32")),
        "event_start":   (("event",), event_start_arr),
        "event_end":     (("event",), event_end_arr),
        "landfall_time": (("event",), landfall_time_arr),
    },
    attrs={
        "TC_window_days": TC_WINDOW_DAYS,
        "base_lock_hours": BASE_LOCK_HOURS,
        "storm_lock_hours": STORM_LOCK_HOURS,   # 保留字段，但 storm 已取消
        "pre_shutdown_hours": PRE_SHUT_HOURS,
        "events_source_excel": XLSX_PATH,
        "potential_power_library_nc": POT_NC_PATH,
        "pdam_source": "Excel column 'Pdam'",
        "filter_rule": f"Only events with Pdam >= {FIXED_PDAM_THRESHOLD} are included in this output NC.",
        "shutdown_rule": "For included events: shutdown starts at landfall_time for 0h (storm-trigger REMOVED).",
        "recovery_rule": (
            f"Recovery starts right after shutdown ends; duration {FIXED_RECOVERY_HOURS}h; "
            f"EXP ease-in (slow->fast) with k={RECOVERY_EXP_K}; "
            f"A(t=0)=1-Pdam; A(t>=T)=1 (NO residual; Pdam(t)=0). "
            f"Note: full recovery occurs at landfall+0h+72h = landfall+96h."
        ),
        "ss_lock_rule": "REMOVED (no SS-based inspection downtime); tc_category_ss kept only for compatibility.",
    },
)

ds_out["sid"].attrs.update({"description": "Unique identifier of TC-HW compound event (string from Excel)"})

print(f"Saving selected events to: {OUT_NC}")
comp = dict(zlib=True, complevel=4)
encoding = {k: comp for k in [
    "V80c","V80","V80_TCmax","gust80_TCmax","P_pot","P_avail_TC","P_avail_std",
    "P_dam_TC","A_TC","A_TC_time","run_flag","shutdown_hours","first_shutdown_rel_time",
    "shutdown_flag_event","shutdown_start_rel_time_event","lock_total_hours_event",
    "shutdown_by_damage","shutdown_by_ss","shutdown_by_storm","tc_category_ss"
]}
ds_out.to_netcdf(OUT_NC, encoding=encoding)
ds_out.close()
ds_pot_all.close()
print("Done. Final NetCDF created (ONLY Pdam>=0.9 events + precomputed potential-power library).")
