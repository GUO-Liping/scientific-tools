# -*- coding: utf-8 -*-
"""
OBS bedload PSD processing (Luong-style) + paper-grade visualization
Procedural (no OOP)

Main outputs
1) Band index time series (Luong-minute + strict 1-min):
   - PSD_Band_LuongMinute.csv
   - PSD_Band_Strict1min.csv

2) Paper-style PSD heatmap (minute-median full spectrum, dB) + black curve overlay:
   - uses strict 1-min bins for consistent time axis

Notes
- fs=100 Hz => Nyquist=50 Hz. Make sure fmax <= 50.
- Luong-style window: nperseg=2^14=163.84 s, 50% overlap => step=81.92 s
  => ~42 Welch windows per hour => ~42 "Luong-minute" points per hour (NOT 60).
- For strict 1-min alignment (hydrology/engineering), we resample("1min").
"""

from obspy import read, UTCDateTime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import Normalize
from scipy.signal import welch


# =============================================================================
# 0) USER CONFIG (edit here)
# =============================================================================

# ---- Input BHZ SAC file ----
DATA_FILE = r"E:\\项目3-YJ项目-推移质监测\\雅江YaJiang-推移质OBS监测数据\\101MT\\C-00001_250618\\65A42E04.15A.BHZ"

# 水位过程
time_river_str = ["2025-06-18 00:00", "2025-06-18 01:00", "2025-06-18 02:00", "2025-06-18 03:00", "2025-06-18 04:00", "2025-06-18 05:00", "2025-06-18 06:00", "2025-06-18 07:00", "2025-06-18 08:00", "2025-06-18 08:30", "2025-06-18 09:00", "2025-06-18 09:40", "2025-06-18 10:00", "2025-06-18 11:00", "2025-06-18 12:00", "2025-06-18 13:00", "2025-06-18 14:00", "2025-06-18 15:00", "2025-06-18 16:00", "2025-06-18 17:00", "2025-06-18 18:00", "2025-06-18 19:00", "2025-06-18 20:00", "2025-06-18 21:00", "2025-06-18 22:00", "2025-06-18 23:00", "2025-06-19 00:00", "2025-06-19 01:00", "2025-06-19 02:00", "2025-06-19 03:00", "2025-06-19 04:00", "2025-06-19 05:00", "2025-06-19 06:00", "2025-06-19 07:00", "2025-06-19 08:00", "2025-06-19 09:00", "2025-06-19 10:00", "2025-06-19 11:00", "2025-06-19 12:00", "2025-06-19 13:00", "2025-06-19 14:00", "2025-06-19 15:00", "2025-06-19 16:00", "2025-06-19 16:30", "2025-06-19 17:00", "2025-06-19 17:30", "2025-06-19 18:00", "2025-06-19 19:00", "2025-06-19 20:00", "2025-06-19 21:00", "2025-06-19 22:00", "2025-06-19 23:00", "2025-06-20 00:00", "2025-06-20 01:00", "2025-06-20 02:00", "2025-06-20 03:00", "2025-06-20 04:00", "2025-06-20 05:00", "2025-06-20 06:00", "2025-06-20 07:00", "2025-06-20 08:00", "2025-06-20 09:00", "2025-06-20 10:00", "2025-06-20 11:00", "2025-06-20 12:00", "2025-06-20 13:00", "2025-06-20 14:00", "2025-06-20 15:00", "2025-06-20 16:00", "2025-06-20 17:00", "2025-06-20 18:00", "2025-06-20 19:00", "2025-06-20 20:00", "2025-06-20 21:00", "2025-06-20 22:00", "2025-06-20 23:00", "2025-06-21 00:00", "2025-06-21 01:00", "2025-06-21 02:00", "2025-06-21 03:00", "2025-06-21 04:00", "2025-06-21 05:00", "2025-06-21 06:00", "2025-06-21 07:00", "2025-06-21 08:00", "2025-06-21 09:00", "2025-06-21 10:00", "2025-06-21 11:00", "2025-06-21 12:00", "2025-06-21 13:00", "2025-06-21 13:20", "2025-06-21 13:50", "2025-06-21 14:00", "2025-06-21 15:00", "2025-06-21 16:00", "2025-06-21 17:00", "2025-06-21 18:00", "2025-06-21 19:00", "2025-06-21 20:00", "2025-06-21 21:00", "2025-06-21 22:00", "2025-06-21 23:00", "2025-06-22 00:00", "2025-06-22 01:00", "2025-06-22 02:00", "2025-06-22 03:00", "2025-06-22 04:00", "2025-06-22 05:00", "2025-06-22 06:00", "2025-06-22 07:00", "2025-06-22 08:00", "2025-06-22 09:00", "2025-06-22 10:00", "2025-06-22 11:00", "2025-06-22 12:00", "2025-06-22 13:00", "2025-06-22 13:20", "2025-06-22 14:00", "2025-06-22 15:00", "2025-06-22 16:00", "2025-06-22 17:00", "2025-06-22 18:00", "2025-06-22 19:00", "2025-06-22 20:00", "2025-06-22 21:00", "2025-06-22 22:00", "2025-06-22 23:00", "2025-06-23 00:00", "2025-06-23 01:00", "2025-06-23 02:00", "2025-06-23 03:00", "2025-06-23 04:00", "2025-06-23 05:00", "2025-06-23 06:00", "2025-06-23 07:00", "2025-06-23 08:00", "2025-06-23 09:00", "2025-06-23 10:00", "2025-06-23 11:00", "2025-06-23 12:00", "2025-06-23 13:00", "2025-06-23 14:00", "2025-06-23 15:00", "2025-06-23 16:00", "2025-06-23 17:00", "2025-06-23 18:00", "2025-06-23 19:00", "2025-06-23 20:00", "2025-06-23 21:00", "2025-06-23 22:00", "2025-06-23 23:00", "2025-06-24 00:00", "2025-06-24 01:00", "2025-06-24 02:00", "2025-06-24 03:00", "2025-06-24 04:00", "2025-06-24 05:00", "2025-06-24 06:00", "2025-06-24 07:00", "2025-06-24 08:00", "2025-06-24 09:00", "2025-06-24 10:00", "2025-06-24 11:00", "2025-06-24 12:00", "2025-06-24 13:00", "2025-06-24 14:00", "2025-06-24 15:00", "2025-06-24 16:00", "2025-06-24 17:00", "2025-06-24 18:00", "2025-06-24 19:00", "2025-06-24 20:00", "2025-06-24 21:00", "2025-06-24 22:00", "2025-06-24 23:00", "2025-06-25 00:00", "2025-06-25 01:00", "2025-06-25 02:00", "2025-06-25 03:00", "2025-06-25 04:00", "2025-06-25 05:00", "2025-06-25 06:00", "2025-06-25 07:00", "2025-06-25 08:00", "2025-06-25 09:00", "2025-06-25 10:00", "2025-06-25 11:00", "2025-06-25 12:00", "2025-06-25 13:00", "2025-06-25 14:00", "2025-06-25 15:00", "2025-06-25 16:00", "2025-06-25 16:50", "2025-06-25 17:00", "2025-06-25 17:50", "2025-06-25 18:00", "2025-06-25 19:00", "2025-06-25 20:00", "2025-06-25 21:00", "2025-06-25 22:00", "2025-06-25 23:00", "2025-06-26 00:00", "2025-06-26 01:00", "2025-06-26 02:00", "2025-06-26 03:00", "2025-06-26 04:00", "2025-06-26 05:00", "2025-06-26 06:00", "2025-06-26 07:00", "2025-06-26 08:00", "2025-06-26 09:00", "2025-06-26 10:00", "2025-06-26 11:00", "2025-06-26 12:00", "2025-06-26 13:00", "2025-06-26 14:00", "2025-06-26 15:00", "2025-06-26 16:00", "2025-06-26 17:00", "2025-06-26 18:00", "2025-06-26 19:00", "2025-06-26 20:00", "2025-06-26 21:00", "2025-06-26 22:00", "2025-06-26 23:00" ]
height_river = np.array([729.70, 729.82, 729.81, 729.84, 729.79, 729.81, 729.92, 729.87, 729.92, 729.94, 729.91, 729.94, 729.98, 730.08, 730.16, 730.20, 730.27, 730.30, 730.35, 730.33, 730.43, 730.43, 730.45, 730.53, 730.63, 730.60, 730.62, 730.71, 730.73, 730.83, 730.82, 730.87, 730.93, 730.92, 731.03, 731.04, 731.08, 731.11, 731.17, 731.22, 731.27, 731.33, 731.43, 731.51, 731.51, 731.51, 731.60, 731.71, 731.79, 731.87, 731.94, 732.00, 732.13, 732.16, 732.28, 732.36, 732.46, 732.50, 732.58, 732.71, 732.76, 732.81, 732.88, 732.94, 733.06, 733.11, 733.19, 733.22, 733.35, 733.38, 733.42, 733.54, 733.60, 733.80, 734.01, 734.19, 734.20, 734.09, 734.04, 733.98, 733.84, 733.82, 733.75, 733.67, 733.67, 733.63, 733.56, 733.55, 733.47, 733.45, 733.41, 733.41, 733.39, 733.41, 733.34, 733.30, 733.22, 733.27, 733.15, 733.09, 733.12, 733.07, 733.03, 732.98, 732.93, 732.91, 732.82, 732.81, 732.78, 732.79, 732.76, 732.73, 732.74, 732.72, 732.64, 732.66, 732.52, 732.59, 732.49, 732.37, 732.26, 732.27, 732.21, 732.19, 732.12, 732.00, 731.85, 731.86, 731.83, 731.82, 731.63, 731.62, 731.61, 731.65, 731.67, 731.64, 731.64, 731.57, 731.55, 731.53, 731.50, 731.45, 731.36, 731.33, 731.36, 731.32, 731.29, 731.28, 731.25, 731.11, 731.00, 730.92, 730.89, 730.89, 730.97, 731.13, 731.24, 731.33, 731.54, 731.66, 731.72, 731.81, 731.94, 732.01, 732.02, 731.96, 731.91, 731.83, 731.76, 731.67, 731.60, 731.60, 731.52, 731.45, 731.37, 731.26, 731.27, 731.21, 731.08, 731.09, 731.02, 731.04, 731.01, 730.97, 730.89, 730.91, 730.87, 730.88, 730.89, 730.86, 730.83, 730.83, 730.86, 730.77, 730.86, 730.78, 730.73, 730.75, 730.69, 730.76, 730.69, 730.67, 730.58, 730.56, 730.59, 730.60, 730.55, 730.59, 730.65, 730.62, 730.69, 730.65, 730.64, 730.62, 730.61, 730.60, 730.56, 730.47, 730.47, 730.42, 730.38, 730.33, 730.32, 730.30, 730.26])
time_river = pd.to_datetime(time_river_str, utc=True)  # 统一 UTC（与你 PSD 的 UTC 对齐）
deepth_river = height_river - 710.6  # 单位：m

# =============================================================================
# 1) Helpers (procedural)
# =============================================================================

def detrend_demean_stream(st_in):
    """Merge (if needed), detrend + demean. Return a new Stream."""
    st = st_in.copy()
    # 对单道也安全；多段/间断数据建议 merge 再处理
    try:
        st.merge(method=1, fill_value="interpolate")
    except Exception:
        # 单条 Trace 通常不需要 merge，merge 失败也不致命
        pass
    st.detrend("linear")
    st.detrend("demean")
    return st


def remove_response_to_velocity(st_in, paz):
    """
    Remove response to ground velocity (m/s).
    pre_filt uses fs-dependent Nyquist.
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
    Luong-style: sliding Welch windows with noverlap=0 inside welch(),
    overlap is implemented by STEP between successive segments.

    Returns:
      f (nfreq,)
      psd_arr (nwin, nfreq)
      times_center (nwin,) as datetime64[ns]
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
            seg, fs=fs, window="hann",
            nperseg=nperseg, noverlap=0,
            detrend=False, scaling="density"
        )

        # 窗口中心时间（由外部传入 t0 更好；这里返回相对 index，由主循环赋值）
        psd_list.append(Pxx)
        times_center[i] = np.datetime64("1970-01-01")  # placeholder, 主循环覆盖

    psd_arr = np.vstack(psd_list)
    return f, psd_arr, times_center


def minute_median_psd(f, psd_arr, times_center_dt64):
    """
    Group PSD windows into minutes by Welch-window center time,
    take median at each frequency.
    Return: DataFrame (index=minute, columns=freq values)
    """
    df = pd.DataFrame(psd_arr)
    df["minute"] = pd.to_datetime(times_center_dt64).floor("min")
    psd_min = df.groupby("minute").median()
    psd_min.index.name = "minute_utc"
    psd_min.columns = f  # use frequency as columns (float)
    return psd_min


def band_index_from_psd(psd_minute_df, fmin, fmax):
    """
    Compute band mean PSD and band integrated power for each minute.
    """
    f = psd_minute_df.columns.to_numpy(dtype=float)
    idx = (f >= fmin) & (f <= fmax)
    if not np.any(idx):
        raise ValueError(f"[ERROR] Band {fmin}-{fmax} Hz has no freq bins. Check fs/nperseg.")

    band_mean = psd_minute_df.loc[:, idx].mean(axis=1)
    # 对每行做积分
    band_int = psd_minute_df.loc[:, idx].apply(lambda r: np.trapezoid(r.values, f[idx]), axis=1)

    out = pd.DataFrame({
        f"PSD_{fmin:g}_{fmax:g}_mean": band_mean,
        f"PSD_{fmin:g}_{fmax:g}_int": band_int
    })
    out.index.name = "minute_utc"
    return out


def make_edges_from_index_and_freq(time_index, f):
    """
    Build pcolormesh edges for time and frequency.
    time_index: pandas DatetimeIndex
    f: 1D float array (freq centers)
    """
    t_num = mdates.date2num(time_index.to_pydatetime())
    if len(t_num) > 1:
        dt = np.median(np.diff(t_num))
    else:
        dt = 1 / 1440.0  # 1 minute in days

    t_edges = np.r_[t_num[0] - dt/2, (t_num[:-1] + t_num[1:]) / 2, t_num[-1] + dt/2]

    f = np.asarray(f, dtype=float)
    if len(f) > 1:
        df = np.median(np.diff(f))
    else:
        df = 1.0
    f_edges = np.r_[f[0] - df/2, (f[:-1] + f[1:]) / 2, f[-1] + df/2]
    return t_edges, f_edges


def map_to_range(x, y0, y1):
    """Map a series to [y0,y1] for overlay on frequency axis."""
    x = np.asarray(x, dtype=float)
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    if (not np.isfinite(xmin)) or (not np.isfinite(xmax)) or (xmax <= xmin):
        return np.full_like(x, (y0 + y1) / 2.0)
    return y0 + (x - xmin) * (y1 - y0) / (xmax - xmin)


# =============================================================================
# 2) Read data + determine processing range
# =============================================================================

st = read(DATA_FILE)
tr_full = st[0]  # assume single trace

print("\n[INFO] Trace header:")
print(tr_full)
fs = tr_full.stats.sampling_rate
fN = fs / 2.0
print(f"[INFO] fs={fs} Hz, Nyquist={fN} Hz")

# ---- Processing time range (UTC) ----
# 推荐：用数据头自动确定时间范围（不越界）
USE_AUTO_TIME_RANGE = True
T_GLOBAL_START = tr_full.stats.starttime   # 仅当 USE_AUTO_TIME_RANGE=False 时生效
N_HOURS = 24 * 1                                      # 仅当 USE_AUTO_TIME_RANGE=False 时生效

# ---- Band index frequency range (bedload proxy) ----
BAND_FMIN = 2.0
BAND_FMAX = 30   # <= Nyquist

# ---- Heatmap frequency display range ----
HEATMAP_FMIN_SHOW = BAND_FMIN
HEATMAP_FMAX_SHOW = BAND_FMAX

# ---- Luong Welch parameters ----
NPERSEG_LUONG = 2**14
NOVERLAP_LUONG = NPERSEG_LUONG // 2
STEP = NPERSEG_LUONG - NOVERLAP_LUONG

# ---- Instrument response removal (counts -> m/s) ----
ADC_SENS = 1.6777e6  # counts/V
PAZ = {
    "poles": [-30.61, -15.8, -0.1693, -0.09732, -0.03333],
    "zeros": [0j, 0j, 0j, -30.21, -16.15],
    "gain": 1021.9,                    # V/(m/s)
    "sensitivity": 1021.9 * ADC_SENS   # counts/(m/s)
}

# ---- Output CSV ----
OUT_CSV_LUONG_MINUTE = "PSD_Band_LuongMinute.csv"
OUT_CSV_STRICT_1MIN  = "PSD_Band_Strict1min.csv"

# ---- Plot styles ----
FIG_DPI = 300

if BAND_FMAX > fN:
    raise ValueError(f"[ERROR] BAND_FMAX={BAND_FMAX} exceeds Nyquist={fN}.")

if USE_AUTO_TIME_RANGE:
    T_GLOBAL_START = tr_full.stats.starttime
    T_GLOBAL_END = tr_full.stats.endtime
    N_HOURS = int((T_GLOBAL_END - T_GLOBAL_START) // 3600)
    print(f"[INFO] Auto time range: {T_GLOBAL_START} -> {T_GLOBAL_END}")
    print(f"[INFO] Auto N_HOURS = {N_HOURS}")
else:
    T_GLOBAL_END = T_GLOBAL_START + N_HOURS * 3600

# 强制边界检查：避免全程 skip
if T_GLOBAL_START < tr_full.stats.starttime or T_GLOBAL_END > tr_full.stats.endtime:
    raise ValueError(
        "[ERROR] Requested time range is out of data coverage.\n"
        f"Data: {tr_full.stats.starttime} ~ {tr_full.stats.endtime}\n"
        f"Req : {T_GLOBAL_START} ~ {T_GLOBAL_END}"
    )


# =============================================================================
# 3) Quick plot: raw counts time series (optional paper-grade overview)
# =============================================================================

fig0 = st.plot(handle=True, color="k", equal_scale=False, linewidth=0.4)
fig0.set_size_inches(12, 3.5)
ax0 = fig0.axes[0]
ax0.xaxis.set_major_locator(mdates.HourLocator(interval=12))
ax0.xaxis.set_minor_locator(mdates.HourLocator(interval=2))
ax0.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
ax0.set_xlabel("Time (UTC)")
ax0.set_ylabel("Counts")
fig0.autofmt_xdate()


# =============================================================================
# 4) Main loop: hourly processing -> collect band index + full spectrum PSD minutes
# =============================================================================

all_minutes_band = []       # list of DataFrame (band indices)
all_minutes_fullspec = []   # list of DataFrame (minute-median full spectrum)
f_ref = None                # reference frequency vector for consistency check
all_psd_minutes = []

for ih in range(N_HOURS):
    t1 = T_GLOBAL_START + ih * 3600
    t2 = t1 + 3600

    # ---- slice 1 hour from raw counts ----
    st_seg = st.copy()
    st_seg.trim(t1, t2)

    # 数据可能存在间断/空段
    if len(st_seg) == 0 or st_seg[0].stats.npts < NPERSEG_LUONG:
        print(f"[WARN] Empty/too short segment: {t1} - {t2}, npts={st_seg[0].stats.npts if len(st_seg)>0 else 0}, skip.")
        continue

    # ---- detrend + demean (counts) ----
    st_seg = detrend_demean_stream(st_seg)

    # ---- remove response -> velocity ----
    st_vel, pre_filt = remove_response_to_velocity(st_seg, PAZ)

    trv = st_vel[0]
    x = trv.data.astype(np.float64)
    x -= np.mean(x)

    if len(x) < NPERSEG_LUONG:
        print(f"[WARN] Too short after preprocessing: {t1} - {t2}, skip.")
        continue

    # ---- sliding Welch PSD (Luong-style) ----
    f, psd_arr, times_center = sliding_welch_psd(x, fs, NPERSEG_LUONG, STEP)
    if f is None:
        print(f"[WARN] nwin<=0: {t1} - {t2}, skip.")
        continue

    # ---- fill correct center times (absolute UTC) ----
    # 注意：sliding_welch_psd 内部只留了 placeholder，这里统一赋值，避免重复传参
    nwin = psd_arr.shape[0]
    t0 = trv.stats.starttime
    for i in range(nwin):
        i0 = i * STEP
        tc = t0 + (i0 + NPERSEG_LUONG / 2) / fs
        times_center[i] = np.datetime64(tc.datetime)

    # ---- minute-median PSD (full spectrum) ----
    psd_minute = minute_median_psd(f, psd_arr, times_center)
    all_psd_minutes.append(psd_minute)   # 保存“分钟×频率”的完整谱

    # ---- ensure frequency axis consistent across hours ----
    if f_ref is None:
        f_ref = psd_minute.columns.to_numpy(dtype=float)
    else:
        f_now = psd_minute.columns.to_numpy(dtype=float)
        if len(f_now) != len(f_ref) or np.max(np.abs(f_now - f_ref)) > 1e-12:
            raise ValueError("[ERROR] Frequency axis changed between hours. Check fs or Welch settings.")

    # ---- band index (per minute) ----
    band_df = band_index_from_psd(psd_minute, BAND_FMIN, BAND_FMAX)

    all_minutes_fullspec.append(psd_minute)
    all_minutes_band.append(band_df)

    print(f"[INFO] {t1} - {t2}: Luong-minute points = {band_df.shape[0]}")


if len(all_minutes_band) == 0:
    raise RuntimeError("[ERROR] No valid hourly segments processed. Check data continuity/time range.")


# =============================================================================
# 5) Merge results + export CSV
# =============================================================================

# ---- merge band index ----
out_band = pd.concat(all_minutes_band).sort_index()
out_band = out_band.groupby(out_band.index).median()  # 去重

# strict 1-min
out_band_1min = out_band.resample("1min").median()

out_band.to_csv(OUT_CSV_LUONG_MINUTE)
out_band_1min.to_csv(OUT_CSV_STRICT_1MIN)

print("\n[INFO] Saved:")
print(" -", OUT_CSV_LUONG_MINUTE, "(Luong-minute)")
print(" -", OUT_CSV_STRICT_1MIN, "(strict 1-min)")
print("[INFO] Luong-minute points:", out_band.shape[0])
print("[INFO] Strict 1-min points:", out_band_1min.shape[0])

# ---- merge full spectrum (for heatmap) ----
psd_full = pd.concat(all_minutes_fullspec).sort_index()
psd_full = psd_full.groupby(psd_full.index).median()       # 去重
psd_full_1min = psd_full.resample("1min").median()         # strict 1-min，与 out_band_1min 时间轴一致

# 合并全部 Luong-minute 频谱
psd_all = pd.concat(all_psd_minutes).sort_index()
psd_all = psd_all.groupby(psd_all.index).median()    # 去重（跨小时重复minute）
# 变成严格 1-min 网格（每列是一个频点）
psd_all_1min = psd_all.resample("1min").median()

# 可选：对缺失分钟做插值（画图更连续；不建议用于定量分析）
psd_all_1min_plot = psd_all_1min.interpolate(limit=10, limit_direction="both")
# =============================================================================
# 6) Plot A: strict 1-min band index time series (log y)
# =============================================================================

band_col_mean = f"PSD_{BAND_FMIN:g}_{BAND_FMAX:g}_mean"

fig_ts, ax_ts = plt.subplots(figsize=(12, 3))
ax_ts.plot(out_band_1min.index, out_band_1min[band_col_mean], linewidth=0.8)
ax_ts.set_yscale("log")
ax_ts.set_xlabel("Time (UTC)")
ax_ts.set_ylabel(rf"Median PSD mean ({BAND_FMIN:g}–{BAND_FMAX:g} Hz) ((m/s)$^2$/Hz)")
ax_ts.set_title("Luong-style band index (strict 1-min)")
ax_ts.grid(True, which="both", alpha=0.25)
fig_ts.autofmt_xdate()


# =============================================================================
# 7) Plot B: paper-style PSD heatmap (dB) + black overlay curve
# =============================================================================

# 1) 转 dB（避免 log10(0)）
P_hm = psd_all_1min_plot.values.T                      # (nfreq, ntime)
P_hm = np.maximum(P_hm, 1e-30)
P_db = 10.0 * np.log10(P_hm)

# 2) 设置颜色范围（用分位数更稳）
vmin = -230# np.nanpercentile(P_db, 5)
vmax = -200# np.nanpercentile(P_db, 95)

times = psd_all_1min_plot.index.to_pydatetime()

fig_hm, ax_hm = plt.subplots(figsize=(14, 4.2))

im_hm = ax_hm.imshow(
    P_db,
    aspect="auto",
    origin="lower",
    extent=[mdates.date2num(times[0]), mdates.date2num(times[-1]), f[0], f[-1]],
    vmin=vmin, vmax=vmax,
    cmap="viridis"
)

ax_hm.set_ylabel("Frequency (Hz)")
ax_hm.set_xlabel("Time (UTC)")
ax_hm.set_ylim(BAND_FMIN, BAND_FMAX)
ax_hm.xaxis_date()
ax_hm.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
fig_hm.autofmt_xdate()

cbar_hm = fig_hm.colorbar(im_hm, ax=ax_hm, pad=0.01)
cbar_hm.set_label(r"PSD (dB rel. (m/s)$^2$/Hz)")

ax_hm.set_title("Minute-median PSD heatmap (strict 1-min grid)")

# =========================================================
# Overlay river depth time series (black curve)
# =========================================================

# 1) 第二个 y 轴（共享 x 轴）
ax_river = ax_hm.twinx()

# 2) 画水深曲线（黑色）
ax_river.plot(
    time_river,
    deepth_river,
    color="black",
    linewidth=2.0,
    label="River depth",
    zorder=10
)

# 3) 右侧 y 轴设置
ax_river.set_ylabel("River depth (m)", color="black")
ax_river.tick_params(axis="y", labelcolor="black")

# （可选）y轴翻转
#ax_river.invert_yaxis()

# 4) 图例（只给水位）
ax_river.legend(
    loc="upper right",
    frameon=False
)

plt.tight_layout()
plt.show()
