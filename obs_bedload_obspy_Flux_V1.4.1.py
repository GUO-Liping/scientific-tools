# -*- coding: utf-8 -*-
"""
OBS 推移质 PSD 处理（Luong-style）+ 可视化
并基于 seismic_bedload（Luong 2024 / Tsai 2012）反演床载输沙通量 qb

【流程概览】
1) 读取 OBS BHZ 数据（SAC）
2) 去趋势/去均值 → 去仪器响应（counts → m/s）
3) 每小时滑动 Welch PSD（Luong 参数）→ 按分钟取中位数（Luong-minute）
4) 合并全时段 PSD → resample 为严格 1-min 时间轴（strict 1-min）
5) 读取水位 Excel → 统一到 UTC → 裁剪到地震处理时间窗 → 对齐到 strict 1-min
6) 反演 qb（strict 1-min）并绘图（PSD 热力图 + 水深叠加；qb 时间序列）

【主要输入】
- 当前目录下的 *.BHZ 文件：OBS 垂向分量 SAC/miniseed 等 ObsPy 可读波形文件
- MT_river_height_pad.xlsx：至少包含 time 和 height 两列

【主要输出】
- output_files_v1.4.1/river_depth.png：水深时间序列
- output_files_v1.4.1/PSD_heatmap.png：PSD 频率-时间热力图，并叠加水深
- output_files_v1.4.1/qb.png：反演床载通量 Qb 时间序列
- output_files_v1.4.1/OBS_results.xlsx：水深、qb、PSD 频带指标

【注意】
- fs=100 Hz ⇒ Nyquist=50 Hz，频带上限必须 ≤ 50
- NPERSEG=2^14=163.84s，STEP=81.92s（50% overlap），每小时≈42个“Luong-minute”
  所以 Luong-minute 不是 60/min，而是≈42/h
- 反演强依赖时间轴一致性：PSD（UTC）与水位（UTC）必须严格对齐
- 当前版本逐小时做响应校正和 PSD。strict 1-min PSD 在每个小时切片内部取窗，
  因此每小时前后约 NPERSEG/2 个样本可能因为窗长不足变成 NaN；如需减少小时边界
  缺口，建议对小时切片增加半窗长度的重叠缓冲，或改为对完整连续 trace 生成 strict PSD。
"""

# =============================================================================
# 0) 导入
# =============================================================================
from obspy import read, UTCDateTime, Stream
import glob  # 多个.BHZ文件读入
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import welch

# ---- seismic-bedload ----
from seismic_bedload import SaltationModel
from seismic_bedload.utils import log_raised_cosine_pdf

import os
OUT_DIR = "output_files_v1.4.1"
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# 1) 配置参数
# =============================================================================
# DATA_FILE = r"MT_202050630_65A42E04.15A.BHZ"  # 读入单个BHZ文件
DATA_FILES = sorted(glob.glob("*.BHZ"))   # 读入多个BHZ文件（指定路径）
RIVER_XLSX = r"MT_river_height_pad.xlsx"

# ---- 频带设置 ----
# BAND_* 用于输出 PSD 频带指标和热力图显示；INV_* 用于 SaltationModel 反演。
# 注意所有频率上限都必须小于等于 Nyquist 频率（fs/2）。
BED_BOTTOM = 710.6  # MT = 710.6, PZ = 2895.0
BAND_FMIN, BAND_FMAX = 2.0, 30.0
HEATMAP_FMIN_SHOW, HEATMAP_FMAX_SHOW = BAND_FMIN, BAND_FMAX

INV_FMIN, INV_FMAX = 2.0, 20.0  # 反演频带（可后续改回 20Hz）

# 时间窗参数：
# USE_AUTO_TIME_RANGE=True 时直接使用波形文件完整覆盖范围，N_days_delay/N_days_process 不生效。
# USE_AUTO_TIME_RANGE=False 时从数据起点向后推迟 N_days_delay 天，再处理 N_days_process 天。
USE_AUTO_TIME_RANGE = True
N_days_delay = 0     # 向后推迟 0 天
N_days_process = 12    # 需要处理的数据时长：12天

# ---- Luong Welch 参数 ----
# NPERSEG_LUONG 决定单个 PSD 窗长；fs=100 Hz 时 2**14 对应 163.84 s。
# STEP 为滑动 Welch 的窗间隔；这里取半窗实现 50% overlap。
NPERSEG_LUONG = 2**14
STEP = NPERSEG_LUONG // 2  # 50% overlap


# ---- 仪器参数 ----
# PAZ 用于 ObsPy Trace.simulate() 去仪器响应：
# 输入 counts，remove_sensitivity=True 后输出近似地面速度 m/s。
ADC_SENS = 1.6777e6
PAZ = {
    "poles": [-30.61, -15.8, -0.1693, -0.09732, -0.03333],
    "zeros": [0j, 0j, 0j, -30.21, -16.15],
    "gain": 1021.9,
    "sensitivity": 1021.9 * ADC_SENS,
}

# ---- MT模型参数----
# D/D50/sigma/pD 描述粒径分布；W/theta/r0 等为 SaltationModel 的场地和河道参数。
# 这些参数直接影响 qb 绝对量级，换测站或河段时必须重新核对。
D = 0.08
D50 = 0.02
sigma = 0.45
mu = D50
s = sigma / np.sqrt(1/3 - 2/np.pi**2)
pD = log_raised_cosine_pdf(D, mu, s) / D

W = 160-27.5  # MT河宽W = 160-27.5；PZ河宽 W = 205 - 25
theta = np.tan(0.4 * np.pi / 180)
r0 = 500.0
rho_s = 2550.0
qb0 = 1e-5
tau_c50 = 0.045

# ---- PZ模型参数----
'''
D = 0.015
D50 = 0.008
sigma = 0.45
mu = D50
s = sigma / np.sqrt(1/3 - 2/np.pi**2)
pD = log_raised_cosine_pdf(D, mu, s) / D

W = 205 - 25  # MT河宽W = 160-27.5；PZ河宽 W = 205 - 25
theta = np.tan(0.4 * np.pi / 180)
r0 = 300.0
rho_s = 2550.0
qb0 = 1e-5
tau_c50 = 0.045
'''

# =============================================================================
# 2) 通用工具函数（时间处理 / 单位转换 / QC）
# =============================================================================
def read_river_from_excel(
    xlsx,
    time_col=0,
    height_col=1,
    sheet=0,
    bed_elev=None,
    utc=True,
    local_tz="Asia/Shanghai",
):
    """
    【功能】从 Excel 读取两列（水位时间、绝对水位），并输出：
        - time_river: UTC tz-aware DatetimeIndex
        - depth_river: 河深（m） = height - bed_elev

    【参数】
    xlsx      : Excel 文件路径
    time_col  : 时间列名或列索引
    height_col: 水位列名或列索引
    bed_elev  : 河床高程（m）
    utc       : True 表示 Excel 时间已经是 UTC；False 表示为本地时区（例如北京时间）
    local_tz  : 本地时区名称（北京时间用 Asia/Shanghai）

    【返回】
    time_river(UTC), depth_river(np.ndarray)
    """
    if bed_elev is None:
        raise ValueError("[ERROR] bed_elev must be provided to convert water level to depth.")

    df = pd.read_excel(xlsx, sheet_name=sheet)

    # 只取两列（防止 Excel 里其它列干扰）
    df = df[[time_col, height_col]].copy()
    df.columns = ["time", "height"]

    # 解析时间（不在这里强制 utc=True，因为你有 utc/local 两种情况）
    t = pd.to_datetime(df["time"], errors="coerce")

    # 时区统一到 UTC（这是后续对齐的关键）。
    # Excel 通常读出来是 naive datetime；如果源表已经带时区，也兼容处理。
    if getattr(t.dt, "tz", None) is None:
        if utc:
            # Excel 时间没有时区信息时：直接当作 UTC
            t = t.dt.tz_localize("UTC")
        else:
            # Excel 时间为本地时间（如北京时间），先 localize，再转 UTC
            t = t.dt.tz_localize(local_tz).dt.tz_convert("UTC")
    else:
        t = t.dt.tz_convert("UTC")

    # 水位转数值并计算河深
    h = pd.to_numeric(df["height"], errors="coerce").to_numpy(dtype=float)
    depth = h - bed_elev

    # 清理无效行，并按时间排序（resample/interp 必须时间递增）
    m = t.notna() & np.isfinite(depth)
    t = t[m]
    depth = depth[m]

    order = np.argsort(t.values)
    time_river = pd.to_datetime(t.values[order], utc=True)
    depth_river = depth[order]
    return time_river, depth_river


def crop_time_series(time_utc, y, t_start_utc, t_end_utc, pad_minutes=10):
    """
    【功能】将时间序列裁剪到地震处理时间窗附近（可选 pad 缓冲）
    目的：保证水位数据只保留与地震处理段一致的部分，避免插值跨天/跨段造成偏差

    time_utc     : UTC tz-aware 时间数组/Index
    y            : 对应数值数组
    t_start_utc  : obspy.UTCDateTime
    t_end_utc    : obspy.UTCDateTime
    pad_minutes  : 前后额外保留的分钟数（防止边界插值时头尾变 NaN）
    """
    t0 = pd.Timestamp(t_start_utc.datetime, tz="UTC") - pd.Timedelta(minutes=pad_minutes)
    t1 = pd.Timestamp(t_end_utc.datetime,   tz="UTC") + pd.Timedelta(minutes=pad_minutes)

    s = pd.Series(np.asarray(y, dtype=float), index=pd.to_datetime(time_utc, utc=True)).sort_index()
    s = s.loc[t0:t1]
    return s.index, s.to_numpy()


def safe_db(psd_linear, floor=1e-30):
    """PSD 线性值转 dB：10*log10(P)，并对极小值加 floor 防止 log(0)。"""
    psd_linear = np.asarray(psd_linear, dtype=float)
    psd_linear = np.maximum(psd_linear, floor)
    return 10.0 * np.log10(psd_linear)


def qc_print_time_coverage(name, idx, t0=None, t1=None):
    """
    简单 QC：打印时间覆盖范围与长度。
    idx: DatetimeIndex
    t0/t1: 可选对比目标窗口（Timestamp）
    """
    if len(idx) == 0:
        print(f"[QC] {name}: EMPTY | n=0")
        if t0 is not None and t1 is not None:
            print(f"[QC] target: {t0} -> {t1}")
        return

    print(f"[QC] {name}: {idx.min()} -> {idx.max()} | n={len(idx)}")
    if t0 is not None and t1 is not None:
        print(f"[QC] target: {t0} -> {t1}")


# =============================================================================
# 3) 地震信号处理函数（去趋势/去响应/PSD）
# =============================================================================
def detrend_demean_stream(st_in: Stream) -> Stream:
    """
    合并间断（尽量插值填补）→ 线性去趋势 → 去均值。

    说明：
    - merge(method=1, fill_value="interpolate") 会对小间断做线性插值，适合保持连续处理；
    - 若存在长时间缺测，插值可能低估缺口附近 PSD，建议先做缺测记录/QC。
    """
    st = st_in.copy()
    try:
        st.merge(method=1, fill_value="interpolate")
    except Exception:
        pass
    st.detrend("linear")
    st.detrend("demean")
    return st


def remove_response_to_velocity(st_in: Stream, paz: dict):
    """
    去仪器响应：counts → m/s
    pre_filt 用于稳定去响应（高低频端做缓冲滤波）。

    pre_filt = (低频停止, 低频通过, 高频通过, 高频停止)。
    高频端按 Nyquist 的 0.7/0.9 设置，避免在接近 Nyquist 处放大响应误差。
    """
    st = st_in.copy()
    fs = st[0].stats.sampling_rate
    fN = fs / 2.0
    pre_filt = (0.2, 0.5, 0.7 * fN, 0.9 * fN)
    for tr in st:
        tr.simulate(paz_remove=paz, remove_sensitivity=True, pre_filt=pre_filt)
    return st, pre_filt


def sliding_welch_psd(x, fs, nperseg, step):
    """
    滑动 Welch PSD（每个窗口独立 welch，nperseg 固定）。

    注意：
    - 这里外层已经切出了长度 nperseg 的滑动片段，所以 welch 内部 noverlap=0；
    - 外层 step 控制相邻 PSD 窗口之间的重叠程度。

    返回：
      f          : 频率轴
      psd_arr    : shape=(nwin, nfreq)
      times_ctr  : shape=(nwin,) datetime64[ns]（后面再填真实中心时间）
    """
    if len(x) < nperseg:
        return None, None, None

    nwin = (len(x) - nperseg) // step + 1
    if nwin <= 0:
        return None, None, None

    psd_list = []
    times_center = np.empty(nwin, dtype="datetime64[ns]")

    for i in range(nwin):
        i0 = i * step
        seg = x[i0:i0 + nperseg]

        f, Pxx = welch(
            seg,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=0,
            detrend=False,
            scaling="density",
        )
        psd_list.append(Pxx)
        times_center[i] = np.datetime64("1970-01-01")  # 先占位

    psd_arr = np.vstack(psd_list)
    return f, psd_arr, times_center


def strict_minute_index(t_start_utc, t_end_utc):
    """
    生成严格 1-min UTC 时间轴：[start, end)。

    使用左闭右开区间可以避免相邻处理段在整分钟边界重复一个时间点。
    """
    t0 = pd.Timestamp(t_start_utc.datetime, tz="UTC").floor("min")
    t1 = pd.Timestamp(t_end_utc.datetime,   tz="UTC").ceil("min")
    return pd.date_range(t0, t1, freq="1min", tz="UTC", inclusive="left")


def strict_1min_psd_from_trace(tr_vel, f_ref, nperseg, fmin=None, fmax=None):
    """
    【功能】从速度 trace（m/s）直接构造“严格 1-min PSD”（每分钟一个 PSD）
    关键点：
    - 以每一分钟的“中心”作为采样（minute + 30s），取长度 nperseg 的数据窗
    - 对每个窗做 welch（no overlap），得到该分钟 PSD
    - 对超界/缺数据的分钟返回 NaN（但通常只有边界少量）

    重要限制：
    - 如果 tr_vel 是逐小时切出来的片段，则每个小时片段的前后半窗都会取不到完整数据。
      这会让每小时边界附近出现 NaN，而不是只有整个数据集的首尾有 NaN。
      若需要连续分钟结果，应给小时片段额外加半窗重叠缓冲，或对完整 trace 统一计算。

    【返回】
    psd_1min: DataFrame(index=minute_utc, columns=f_ref)  (单位：线性 PSD)
    """
    fs = tr_vel.stats.sampling_rate
    t0 = tr_vel.stats.starttime  # UTCDateTime
    npts = tr_vel.stats.npts

    # minute index：严格 1-min（用 trace 覆盖范围）
    idx = pd.date_range(
        pd.Timestamp(t0.datetime, tz="UTC").floor("min"),
        pd.Timestamp((t0 + (npts - 1) / fs).datetime, tz="UTC").ceil("min"),
        freq="1min", tz="UTC", inclusive="left"
    )

    psd_out = np.full((len(idx), len(f_ref)), np.nan, dtype=float)

    # 每分钟取窗中心在 minute+30s
    half = nperseg // 2
    x = tr_vel.data.astype(np.float64)
    x = x - np.mean(x)

    for k, minute in enumerate(idx):
        tc = UTCDateTime(minute.to_pydatetime()) + 30.0  # minute中心（+30s）
        ic = int(round((tc - t0) * fs))                 # 中心点样本号
        i1 = ic - half
        i2 = i1 + nperseg
        if i1 < 0 or i2 > npts:
            continue  # 边界超界：保持 NaN

        seg = x[i1:i2]
        f, Pxx = welch(seg, fs=fs, window="hann",
                       nperseg=nperseg, noverlap=0,
                       detrend=False, scaling="density")

        # 频率轴必须与 f_ref 一致（如果不一致，强制报错）
        if (len(f) != len(f_ref)) or (np.max(np.abs(f - f_ref)) > 1e-12):
            raise ValueError("[ERROR] strict_1min PSD frequency axis mismatch. Check fs/nperseg.")

        psd_out[k, :] = Pxx

    psd_1min = pd.DataFrame(psd_out, index=idx, columns=f_ref)
    psd_1min.index.name = "minute_utc"

    # 可选：只保留某个频带列（减少内存）
    if fmin is not None and fmax is not None:
        cols = (psd_1min.columns >= fmin) & (psd_1min.columns <= fmax)
        psd_1min = psd_1min.loc[:, cols]

    return psd_1min

def minute_median_psd(f, psd_arr, times_center_dt64):
    """
    将滑动窗口 PSD 归并到分钟（UTC）：
    - 把每个窗口的中心时间 floor 到 minute
    - 同一分钟内多个窗口取 median（对离群值更稳健）
    """
    df = pd.DataFrame(psd_arr)
    df["minute"] = pd.to_datetime(times_center_dt64, utc=True).floor("min")
    psd_min = df.groupby("minute").median()
    psd_min.index.name = "minute_utc"
    psd_min.columns = f
    return psd_min


def band_index_from_psd(psd_minute_df, fmin, fmax):
    """
    计算一个频带的 PSD 指标（作为 proxy）：
    - mean: 频带内各频点 PSD 均值
    - int : 对频率积分（梯形积分）

    输出为线性 PSD 指标，不是 dB。若用于统计回归，请保持和后续模型的单位一致。
    """
    f = psd_minute_df.columns.to_numpy(dtype=float)
    idx = (f >= fmin) & (f <= fmax)
    if not np.any(idx):
        raise ValueError(f"[ERROR] Band {fmin}-{fmax} Hz has no freq bins.")

    band_mean = psd_minute_df.loc[:, idx].mean(axis=1)
    band_int = psd_minute_df.loc[:, idx].apply(lambda r: np.trapezoid(r.values, f[idx]), axis=1)

    out = pd.DataFrame(
        {
            f"PSD_{fmin:g}_{fmax:g}_mean": band_mean,
            f"PSD_{fmin:g}_{fmax:g}_int": band_int,
        }
    )
    out.index.name = "minute_utc"
    return out


def align_river_to_minutes(psd_minutes_index, time_river, depth_river):
    """
    将水位/河深对齐到 PSD 的 strict 1-min 时间轴：
    1) 先把河深序列 resample 到 1min（mean）
    2) 用 time 插值补齐缺失（两端也补）
    3) reindex 到 PSD 分钟轴，再做一次插值（保证完全同轴）

    注意：
    - limit_direction="both" 会外推到两端；如果水位数据离地震窗口太远，
      外推会掩盖真实缺测。调用前应先 crop，并检查覆盖范围。
    """
    if len(time_river) == 0:
        raise ValueError("[ERROR] River series is empty after cropping. Check Excel time zone and time range.")

    river = pd.Series(depth_river, index=time_river).sort_index()

    # 先把水位自身变成 1-min（方便后续对齐）
    river_1min = river.resample("1min").mean()
    river_1min = river_1min.interpolate(method="time", limit_direction="both")

    # 对齐到 PSD 的分钟轴
    river_on_psd = river_1min.reindex(psd_minutes_index)
    river_on_psd = river_on_psd.interpolate(method="time", limit_direction="both")
    return river_on_psd


# =============================================================================
# 4) 主程序：读取地震数据 + 检查时间窗 + 绘制选取的原始数据
# =============================================================================
st = Stream()

if len(DATA_FILES) == 0:
    raise FileNotFoundError("[ERROR] No *.BHZ files found in current working directory.")

for f in DATA_FILES:
    try:
        print("loading .BHZ File:", f)
        st_tmp = read(f)
        tr_tmp = st_tmp[0]
        print(tr_tmp)
        st += st_tmp
    except Exception as e:
        print(f"[SKIP] {f} -> {e}")
        continue

if len(st) == 0:
    raise RuntimeError("[ERROR] All input waveforms failed to load. Check file format and paths.")

# ObsPy merge 会把同一台站/通道的连续 trace 合并为一条；间断处用插值补齐。
# 如果数据来自多个台站或多个通道，这里只取 st[0] 会丢掉其它 trace，需要先筛选台站/通道。
st.merge(method=1, fill_value="interpolate")

tr_full = st[0]
fs = tr_full.stats.sampling_rate
fN = fs / 2.0
print(f"[INFO] fs={fs} Hz, Nyquist={fN} Hz")

if max(BAND_FMAX, HEATMAP_FMAX_SHOW, INV_FMAX) > fN:
    raise ValueError(
        "[ERROR] Frequency upper limit exceeds Nyquist. "
        f"Nyquist={fN}, BAND_FMAX={BAND_FMAX}, "
        f"HEATMAP_FMAX_SHOW={HEATMAP_FMAX_SHOW}, INV_FMAX={INV_FMAX}."
    )

# ---- 处理时间窗（UTC）----
T_GLOBAL_START = tr_full.stats.starttime
T_GLOBAL_END = tr_full.stats.endtime

# 处理时间窗确定
if USE_AUTO_TIME_RANGE:
    N_HOURS = int((T_GLOBAL_END - T_GLOBAL_START) // 3600)
    print(f"[INFO] 自动时间窗: {T_GLOBAL_START} -> {T_GLOBAL_END} ({N_HOURS} 小时)")
else:
    T_GLOBAL_START = T_GLOBAL_START + N_days_delay * 24 * 3600   # 向后推迟 0 天
    N_HOURS = int(N_days_process*24)     # 处理时长（小时）
    T_GLOBAL_END = T_GLOBAL_START + N_HOURS * 3600
    print(f"[INFO] 处理时间窗: {T_GLOBAL_START} -> {T_GLOBAL_END} ({N_HOURS} 小时)")

# 时间窗覆盖检查（避免 trim 超界）
if T_GLOBAL_START < tr_full.stats.starttime or T_GLOBAL_END > tr_full.stats.endtime:
    raise ValueError(
        "[ERROR] Requested time range is out of data coverage.\n"
        f"Data: {tr_full.stats.starttime} ~ {tr_full.stats.endtime}\n"
        f"Req : {T_GLOBAL_START} ~ {T_GLOBAL_END}"
    )


'''
# --------- 只取窗口数据（最关键：先 trim，在绘图）---------
# 若要启用本 QC 绘图块，请取消三引号注释；计时需要 import time。
import time

time_plot_start = time.time()
st_win = st.copy().trim(T_GLOBAL_START, T_GLOBAL_END)

# 原始数据
tr_raw = st_win[0].copy()
tr_raw.stats.channel = "RAW (cnt)"

# 去趋势去均值（只对窗口）
st_det = detrend_demean_stream(st_win.copy())
tr_det = st_det[0].copy()
tr_det.stats.channel = "DET (cnt)"

# 去响应（只对窗口）
st_vel, _ = remove_response_to_velocity(st_det.copy(), PAZ)
tr_vel = st_vel[0].copy()
tr_vel.stats.channel = "VEL (m/s)"

# 打包 QC
st_qc = Stream([tr_raw, tr_det, tr_vel])

st_qc.plot(
    method="fast",
    equal_scale=False,
    linewidth=0.6,
    size=(1400, 600),
    title=f"OBS waveform QC (window only)\n{T_GLOBAL_START} → {T_GLOBAL_END}",
)
time_plot_end = time.time()
print("total time for plot raw data = ", time_plot_end - time_plot_start, 's')
'''

# =============================================================================
# 5) PSD 主循环：逐小时处理（Luong-minute）
# =============================================================================
all_minutes_band = []
all_strict_1min_fullspec = []
f_ref = None

for ih in range(N_HOURS):
    t1 = T_GLOBAL_START + ih * 3600
    t2 = t1 + 3600

    # 取这一小时的原始数据。
    # 注意：ObsPy slice 默认包含端点，整点附近可能多 1 个样本；对 Welch 影响很小。
    st_seg = st.slice(t1, t2)

    # 数据过短：跳过（会导致后面 strict 1-min 有 NaN，这是正常的）
    if len(st_seg) == 0 or st_seg[0].stats.npts < NPERSEG_LUONG:
        npts = st_seg[0].stats.npts if len(st_seg) > 0 else 0
        print(f"[WARN] Empty/too short segment: {t1} - {t2}, npts={npts}, skip.")
        continue

    # 预处理：去趋势去均值 → 去响应得到速度。
    # 这里按小时单独去响应，计算量较低；代价是小时边界处滤波/窗函数可能出现边缘效应。
    st_seg = detrend_demean_stream(st_seg)
    st_vel, _ = remove_response_to_velocity(st_seg, PAZ)

    trv = st_vel[0]
    x = trv.data.astype(np.float64)
    x -= np.mean(x)

    # Luong-style 滑动 Welch：每小时内约 42 个窗口中心。
    # 这些窗口中心再 floor 到分钟，因此不是严格每分钟一个 PSD。
    f, psd_arr, times_center = sliding_welch_psd(x, fs, NPERSEG_LUONG, STEP)
    if f is None:
        print(f"[WARN] nwin<=0: {t1} - {t2}, skip.")
        continue

    # 把占位时间填成真实中心时间（UTC）
    nwin = psd_arr.shape[0]
    t0 = trv.stats.starttime
    for i in range(nwin):
        i0 = i * STEP
        tc = t0 + (i0 + NPERSEG_LUONG / 2) / fs
        times_center[i] = np.datetime64(tc.datetime)

    # Luong-minute：一分钟内取 PSD 中位数。
    # 该结果保留 Luong 风格的窗口定义，主要用于输出 PSD 频带指标。
    psd_minute = minute_median_psd(f, psd_arr, times_center)

    # strict 1-min PSD：以每个整分钟 +30 s 为中心重算 PSD。
    # 该结果用于热力图和 qb 反演，使 PSD、水位、qb 共用同一条 1-min 时间轴。
    psd_strict_1min = strict_1min_psd_from_trace(
        tr_vel=trv,
        f_ref=psd_minute.columns.to_numpy(dtype=float),  # 统一频率轴
        nperseg=NPERSEG_LUONG,
    )
    
    all_strict_1min_fullspec.append(psd_strict_1min)

    # 频率轴一致性检查（不同小时必须相同）
    if f_ref is None:
        f_ref = psd_minute.columns.to_numpy(dtype=float)
    else:
        f_now = psd_minute.columns.to_numpy(dtype=float)
        if len(f_now) != len(f_ref) or np.max(np.abs(f_now - f_ref)) > 1e-12:
            raise ValueError("[ERROR] Frequency axis changed between hours. Check fs/Welch settings.")

    # 频带指标
    band_df = band_index_from_psd(psd_minute, BAND_FMIN, BAND_FMAX)

    all_minutes_band.append(band_df)

    print(f"[INFO] {t1} - {t2}: Luong-minute points = {band_df.shape[0]}")

if len(all_minutes_band) == 0:
    raise RuntimeError("[ERROR] No valid hourly segments processed. Check data/time range.")


# =============================================================================
# 6) 合并全时段 PSD，并生成 strict 1-min 时间轴
# =============================================================================
# band 指标：Luong-minute 结果先合并，再按 1 min resample。
# 注意 out_band_1min 不是 strict PSD 的频带积分，而是 Luong-minute 指标的分钟网格版本。
out_band = pd.concat(all_minutes_band).sort_index()
out_band = out_band.groupby(out_band.index).median()
out_band_1min = out_band.resample("1min").median()

# full spectrum PSD：strict 1-min full spectrum PSD（理论上每分钟一个）。
# 若每小时切片边界或原始数据缺测导致取不到完整 nperseg 窗，对应分钟保留 NaN。
psd_all_1min = pd.concat(all_strict_1min_fullspec).sort_index()
psd_all_1min = psd_all_1min.groupby(psd_all_1min.index).median()  # 防重

# 用于画热图：允许短缺口插值（仅用于可视化，不影响反演输入）
psd_all_1min_plot = psd_all_1min.interpolate(limit=10, limit_direction="both")

# 强制 PSD 时间轴覆盖处理窗口（严格 1-min）
target_index = strict_minute_index(T_GLOBAL_START, T_GLOBAL_END)
psd_all_1min = psd_all_1min.reindex(target_index)
psd_all_1min_plot = psd_all_1min_plot.reindex(target_index)


# =============================================================================
# 7) 水位读取 → UTC → 裁剪到地震窗口 → 对齐到 PSD 1-min
# =============================================================================
time_river, depth_river = read_river_from_excel(
    RIVER_XLSX,
    time_col="time",
    height_col="height",
    bed_elev=BED_BOTTOM,
    utc=False,  # Excel 是北京时间的话：False；若已是UTC：True
)

fig, ax = plt.subplots(figsize=(14, 4.2))

ax.plot(time_river, depth_river, color="black", lw=1.2, label="River depth (m)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
ax.legend(frameon=False)
ax.grid(alpha=0.3)

# 裁剪到地震处理窗口（加 10 分钟缓冲，防边界插值 NaN）
time_river, depth_river = crop_time_series(
    time_river, depth_river,
    T_GLOBAL_START, T_GLOBAL_END,
    pad_minutes=10,
)

# QC：打印水位覆盖范围 vs 地震窗口
qc_print_time_coverage(
    "river(after crop)",
    pd.DatetimeIndex(time_river),
    t0=pd.Timestamp(T_GLOBAL_START.datetime, tz="UTC"),
    t1=pd.Timestamp(T_GLOBAL_END.datetime, tz="UTC"),
)

# 对齐到 PSD strict 1-min
river_on_minute = align_river_to_minutes(psd_all_1min.index, time_river, depth_river)
print("[QC] river_on_minute NaNs:", river_on_minute.isna().sum(), "/", river_on_minute.size)


# =============================================================================
# 8) 反演 qb（SaltationModel，strict 1-min）
# =============================================================================
f_all = psd_all_1min.columns.to_numpy(dtype=float)
inv_band = (f_all >= INV_FMIN) & (f_all <= INV_FMAX)
if not np.any(inv_band):
    raise ValueError(f"[ERROR] Inversion band {INV_FMIN}-{INV_FMAX} Hz has no bins.")

# 反演输入：每分钟在反演频带内取 PSD(dB) 的中位数
P_lin = psd_all_1min.to_numpy(dtype=float)
PSD_dB = safe_db(P_lin, floor=1e-30)
PSD_obs_dB = np.nanmedian(PSD_dB[:, inv_band], axis=1)

# 水深 H（严格与 PSD 同轴）
H = river_on_minute.values.astype(float)

PSD_missing = ~np.isfinite(PSD_obs_dB)
H_missing   = ~np.isfinite(H)

print("[QC] PSD_obs_dB missing:", PSD_missing.sum())
print("[QC] H missing:", H_missing.sum())
print("[QC] both ok:", np.isfinite(PSD_obs_dB).sum(), np.isfinite(H).sum())
print("[QC] inversion valid mask:", (np.isfinite(PSD_obs_dB) & np.isfinite(H)).sum())

# 反演只用 PSD 与 H 都是有限值的分钟
mask = np.isfinite(PSD_obs_dB) & np.isfinite(H)
if not np.any(mask):
    raise RuntimeError(
        "[ERROR] No valid samples for inversion. "
        "Check PSD NaNs, river time zone, river coverage, and processing time range."
    )

PSD_obs_dB_use = PSD_obs_dB[mask]
H_use = H[mask]
f_inv = f_all[inv_band]

print("\n[INFO] Inversion inputs:")
print("[INFO] PSD_obs_dB range:", np.nanmin(PSD_obs_dB_use), np.nanmax(PSD_obs_dB_use))
print("[INFO] H range (m):", np.nanmin(H_use), np.nanmax(H_use))
print("[INFO] f_inv range:", f_inv[0], "to", f_inv[-1], "Hz, nfreq=", f_inv.size)


model = SaltationModel()
qb_use = model.inverse_bedload(
    PSD_obs_dB_use, f_inv, D, H_use, W, theta, r0, qb0,
    D50=D50, tau_c50=tau_c50, pdf=pD,
)

# 把反演结果放回完整 strict 1-min 时间轴
qb_1min = pd.Series(np.nan, index=psd_all_1min.index, name="qb_m2_s")
qb_1min.loc[psd_all_1min.index[mask]] = qb_use

qb_df = pd.DataFrame(index=qb_1min.index)
qb_df["qb_vol_m2_s"]   = qb_1min
qb_df["qb_mass_kg_m_s"] = qb_df["qb_vol_m2_s"] * rho_s
qb_df["Qb_mass_kg_s"] = qb_df["qb_mass_kg_m_s"] * W

# 输出 qb

print("[INFO] Qb_mass range (kg/s):", np.nanmin(qb_df["Qb_mass_kg_s"]), np.nanmax(qb_df["Qb_mass_kg_s"]))
print("[QC] qb NaNs:", qb_df["Qb_mass_kg_s"].isna().sum(), "/", qb_df.shape[0])


# =============================================================================
# 9) 绘图：PSD 热图 + 水深；qb 时间序列
# =============================================================================
# ---- PSD heatmap ----
P_hm = psd_all_1min_plot.values.T
P_db = safe_db(P_hm, floor=1e-30)

f_show = psd_all_1min_plot.columns.to_numpy(dtype=float)
freq_mask_show = (f_show >= HEATMAP_FMIN_SHOW) & (f_show <= HEATMAP_FMAX_SHOW)
if not np.any(freq_mask_show):
    raise ValueError(
        f"[ERROR] Heatmap band {HEATMAP_FMIN_SHOW}-{HEATMAP_FMAX_SHOW} Hz has no frequency bins."
    )

P_db_show = P_db[freq_mask_show, :]
f_show_use = f_show[freq_mask_show]

vmin = np.nanpercentile(P_db_show, 5)
vmax = np.nanpercentile(P_db_show, 95)

times = psd_all_1min_plot.index.to_pydatetime()
t0_num = mdates.date2num(times[0])
t1_num = mdates.date2num(times[-1])

fig_hm, ax_hm = plt.subplots(figsize=(14, 4.2))
im_hm = ax_hm.imshow(
    P_db_show,
    aspect="auto",
    origin="lower",
    extent=[t0_num, t1_num, f_show_use[0], f_show_use[-1]],
    vmin=vmin, vmax=vmax,
    cmap="viridis",
)

ax_hm.set_ylabel("Frequency (Hz)")
ax_hm.set_xlabel("Time (UTC)")
ax_hm.set_ylim(HEATMAP_FMIN_SHOW, HEATMAP_FMAX_SHOW)
ax_hm.xaxis_date()
ax_hm.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
fig_hm.autofmt_xdate()

cbar_hm = fig_hm.colorbar(im_hm, ax=ax_hm, pad=0.05)
cbar_hm.set_label(r"PSD (dB rel. (m/s)$^2$/Hz)")
ax_hm.set_title("Minute-median PSD heatmap (strict 1-min grid)")

# ---- 叠加水深（右轴）----
ax_river = ax_hm.twinx()
ax_river.plot(psd_all_1min.index, river_on_minute.values, color="black", linewidth=2.0,
              label="River depth", zorder=10)
ax_river.set_ylabel("River depth (m)", color="black")
ax_river.tick_params(axis="y", labelcolor="black")
ax_river.legend(loc="upper left", frameon=False)

plt.tight_layout()

# ---- qb time series ----
fig_qb, ax_qb = plt.subplots(figsize=(14, 4.2))
ax_qb.plot(qb_df.index, qb_df["Qb_mass_kg_s"].interpolate(limit=5), linewidth=0.8)
ax_qb.set_xlabel("Time (UTC)")
ax_qb.set_ylabel(r"$q_b$ (kg s$^{-1}$)")
ax_qb.set_title("Inverted bedload flux (SaltationModel, strict 1-min)")
ax_qb.grid(True, which="both", alpha=0.25)
fig_qb.autofmt_xdate()

# 可选：强制 x 轴范围（避免绘图自动裁掉两端）
ax_qb.set_xlim(qb_df.index.min(), qb_df.index.max())

print("[DEBUG] psd tz:", psd_all_1min.index.tz, "river tz:", pd.DatetimeIndex(time_river).tz)
plt.tight_layout()
plt.show()

fig.savefig(os.path.join(OUT_DIR, "river_depth.png"), dpi=300, bbox_inches="tight")
fig_hm.savefig(os.path.join(OUT_DIR, "PSD_heatmap.png"), dpi=300, bbox_inches="tight")
fig_qb.savefig(os.path.join(OUT_DIR, "qb.png"), dpi=300, bbox_inches="tight")

with pd.ExcelWriter(os.path.join(OUT_DIR, "OBS_results.xlsx"), engine="openpyxl") as writer:
    river_on_minute.copy().tz_localize(None).to_frame("river_depth").to_excel(writer, sheet_name="river")
    qb_df.copy().tz_localize(None).to_excel(writer, sheet_name="qb")
    out_band_1min.copy().tz_localize(None).to_excel(writer, sheet_name="PSD_band")


