# -*- coding: utf-8 -*-
"""
OBS bedload PSD processing (Luong-style) + paper-grade visualization
+ Seismic-bedload (Luong 2024 / Tsai 2012) inversion for bedload flux (qb)

Procedural (no OOP)

Main outputs
1) Band index time series (Luong-minute + strict 1-min):
   - PSD_Band_LuongMinute.csv
   - PSD_Band_Strict1min.csv

2) Paper-style PSD heatmap (minute-median full spectrum, dB) + river depth overlay:
   - uses strict 1-min bins for consistent time axis

3) Bedload flux inversion output (strict 1-min):
   - BedloadFlux_qb_Strict1min.csv

Notes
- fs=100 Hz => Nyquist=50 Hz. Make sure fmax <= 50.
- Luong-style window: nperseg=2^14=163.84 s, 50% overlap => step=81.92 s
  => ~42 Welch windows per hour => ~42 "Luong-minute" points per hour (NOT 60).
- For strict 1-min alignment (hydrology/engineering), we resample("1min").
"""

from obspy import read, UTCDateTime, Stream
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import welch

# ---- seismic-bedload ----
from seismic_bedload import SaltationModel
from seismic_bedload.utils import log_raised_cosine_pdf


# =============================================================================
# 0) 用户配置
# =============================================================================

# ---- Input BHZ SAC file ----
DATA_FILE = r"E:\\项目3-YJ项目-推移质监测\\雅江YaJiang-推移质OBS监测数据\\101MT\\C-00001_250618\\65A42E04.15A.BHZ"
#DATA_FILE = r"E:\\雅江OBS数据\\65A42E04.15A.BHZ"

# ---- Water level/depth series ----
# 水位过程
time_river_str = ["2025-06-18 00:00", "2025-06-18 01:00", "2025-06-18 02:00", "2025-06-18 03:00", "2025-06-18 04:00", "2025-06-18 05:00", "2025-06-18 06:00", "2025-06-18 07:00", "2025-06-18 08:00", "2025-06-18 08:30", "2025-06-18 09:00", "2025-06-18 09:40", "2025-06-18 10:00", "2025-06-18 11:00", "2025-06-18 12:00", "2025-06-18 13:00", "2025-06-18 14:00", "2025-06-18 15:00", "2025-06-18 16:00", "2025-06-18 17:00", "2025-06-18 18:00", "2025-06-18 19:00", "2025-06-18 20:00", "2025-06-18 21:00", "2025-06-18 22:00", "2025-06-18 23:00", "2025-06-19 00:00", "2025-06-19 01:00", "2025-06-19 02:00", "2025-06-19 03:00", "2025-06-19 04:00", "2025-06-19 05:00", "2025-06-19 06:00", "2025-06-19 07:00", "2025-06-19 08:00", "2025-06-19 09:00", "2025-06-19 10:00", "2025-06-19 11:00", "2025-06-19 12:00", "2025-06-19 13:00", "2025-06-19 14:00", "2025-06-19 15:00", "2025-06-19 16:00", "2025-06-19 16:30", "2025-06-19 17:00", "2025-06-19 17:30", "2025-06-19 18:00", "2025-06-19 19:00", "2025-06-19 20:00", "2025-06-19 21:00", "2025-06-19 22:00", "2025-06-19 23:00", "2025-06-20 00:00", "2025-06-20 01:00", "2025-06-20 02:00", "2025-06-20 03:00", "2025-06-20 04:00", "2025-06-20 05:00", "2025-06-20 06:00", "2025-06-20 07:00", "2025-06-20 08:00", "2025-06-20 09:00", "2025-06-20 10:00", "2025-06-20 11:00", "2025-06-20 12:00", "2025-06-20 13:00", "2025-06-20 14:00", "2025-06-20 15:00", "2025-06-20 16:00", "2025-06-20 17:00", "2025-06-20 18:00", "2025-06-20 19:00", "2025-06-20 20:00", "2025-06-20 21:00", "2025-06-20 22:00", "2025-06-20 23:00", "2025-06-21 00:00", "2025-06-21 01:00", "2025-06-21 02:00", "2025-06-21 03:00", "2025-06-21 04:00", "2025-06-21 05:00", "2025-06-21 06:00", "2025-06-21 07:00", "2025-06-21 08:00", "2025-06-21 09:00", "2025-06-21 10:00", "2025-06-21 11:00", "2025-06-21 12:00", "2025-06-21 13:00", "2025-06-21 13:20", "2025-06-21 13:50", "2025-06-21 14:00", "2025-06-21 15:00", "2025-06-21 16:00", "2025-06-21 17:00", "2025-06-21 18:00", "2025-06-21 19:00", "2025-06-21 20:00", "2025-06-21 21:00", "2025-06-21 22:00", "2025-06-21 23:00", "2025-06-22 00:00", "2025-06-22 01:00", "2025-06-22 02:00", "2025-06-22 03:00", "2025-06-22 04:00", "2025-06-22 05:00", "2025-06-22 06:00", "2025-06-22 07:00", "2025-06-22 08:00", "2025-06-22 09:00", "2025-06-22 10:00", "2025-06-22 11:00", "2025-06-22 12:00", "2025-06-22 13:00", "2025-06-22 13:20", "2025-06-22 14:00", "2025-06-22 15:00", "2025-06-22 16:00", "2025-06-22 17:00", "2025-06-22 18:00", "2025-06-22 19:00", "2025-06-22 20:00", "2025-06-22 21:00", "2025-06-22 22:00", "2025-06-22 23:00", "2025-06-23 00:00", "2025-06-23 01:00", "2025-06-23 02:00", "2025-06-23 03:00", "2025-06-23 04:00", "2025-06-23 05:00", "2025-06-23 06:00", "2025-06-23 07:00", "2025-06-23 08:00", "2025-06-23 09:00", "2025-06-23 10:00", "2025-06-23 11:00", "2025-06-23 12:00", "2025-06-23 13:00", "2025-06-23 14:00", "2025-06-23 15:00", "2025-06-23 16:00", "2025-06-23 17:00", "2025-06-23 18:00", "2025-06-23 19:00", "2025-06-23 20:00", "2025-06-23 21:00", "2025-06-23 22:00", "2025-06-23 23:00", "2025-06-24 00:00", "2025-06-24 01:00", "2025-06-24 02:00", "2025-06-24 03:00", "2025-06-24 04:00", "2025-06-24 05:00", "2025-06-24 06:00", "2025-06-24 07:00", "2025-06-24 08:00", "2025-06-24 09:00", "2025-06-24 10:00", "2025-06-24 11:00", "2025-06-24 12:00", "2025-06-24 13:00", "2025-06-24 14:00", "2025-06-24 15:00", "2025-06-24 16:00", "2025-06-24 17:00", "2025-06-24 18:00", "2025-06-24 19:00", "2025-06-24 20:00", "2025-06-24 21:00", "2025-06-24 22:00", "2025-06-24 23:00", "2025-06-25 00:00", "2025-06-25 01:00", "2025-06-25 02:00", "2025-06-25 03:00", "2025-06-25 04:00", "2025-06-25 05:00", "2025-06-25 06:00", "2025-06-25 07:00", "2025-06-25 08:00", "2025-06-25 09:00", "2025-06-25 10:00", "2025-06-25 11:00", "2025-06-25 12:00", "2025-06-25 13:00", "2025-06-25 14:00", "2025-06-25 15:00", "2025-06-25 16:00", "2025-06-25 16:50", "2025-06-25 17:00", "2025-06-25 17:50", "2025-06-25 18:00", "2025-06-25 19:00", "2025-06-25 20:00", "2025-06-25 21:00", "2025-06-25 22:00", "2025-06-25 23:00", "2025-06-26 00:00", "2025-06-26 01:00", "2025-06-26 02:00", "2025-06-26 03:00", "2025-06-26 04:00", "2025-06-26 05:00", "2025-06-26 06:00", "2025-06-26 07:00", "2025-06-26 08:00", "2025-06-26 09:00", "2025-06-26 10:00", "2025-06-26 11:00", "2025-06-26 12:00", "2025-06-26 13:00", "2025-06-26 14:00", "2025-06-26 15:00", "2025-06-26 16:00", "2025-06-26 17:00", "2025-06-26 18:00", "2025-06-26 19:00", "2025-06-26 20:00", "2025-06-26 21:00", "2025-06-26 22:00", "2025-06-26 23:00" ]
height_river = np.array([729.70, 729.82, 729.81, 729.84, 729.79, 729.81, 729.92, 729.87, 729.92, 729.94, 729.91, 729.94, 729.98, 730.08, 730.16, 730.20, 730.27, 730.30, 730.35, 730.33, 730.43, 730.43, 730.45, 730.53, 730.63, 730.60, 730.62, 730.71, 730.73, 730.83, 730.82, 730.87, 730.93, 730.92, 731.03, 731.04, 731.08, 731.11, 731.17, 731.22, 731.27, 731.33, 731.43, 731.51, 731.51, 731.51, 731.60, 731.71, 731.79, 731.87, 731.94, 732.00, 732.13, 732.16, 732.28, 732.36, 732.46, 732.50, 732.58, 732.71, 732.76, 732.81, 732.88, 732.94, 733.06, 733.11, 733.19, 733.22, 733.35, 733.38, 733.42, 733.54, 733.60, 733.80, 734.01, 734.19, 734.20, 734.09, 734.04, 733.98, 733.84, 733.82, 733.75, 733.67, 733.67, 733.63, 733.56, 733.55, 733.47, 733.45, 733.41, 733.41, 733.39, 733.41, 733.34, 733.30, 733.22, 733.27, 733.15, 733.09, 733.12, 733.07, 733.03, 732.98, 732.93, 732.91, 732.82, 732.81, 732.78, 732.79, 732.76, 732.73, 732.74, 732.72, 732.64, 732.66, 732.52, 732.59, 732.49, 732.37, 732.26, 732.27, 732.21, 732.19, 732.12, 732.00, 731.85, 731.86, 731.83, 731.82, 731.63, 731.62, 731.61, 731.65, 731.67, 731.64, 731.64, 731.57, 731.55, 731.53, 731.50, 731.45, 731.36, 731.33, 731.36, 731.32, 731.29, 731.28, 731.25, 731.11, 731.00, 730.92, 730.89, 730.89, 730.97, 731.13, 731.24, 731.33, 731.54, 731.66, 731.72, 731.81, 731.94, 732.01, 732.02, 731.96, 731.91, 731.83, 731.76, 731.67, 731.60, 731.60, 731.52, 731.45, 731.37, 731.26, 731.27, 731.21, 731.08, 731.09, 731.02, 731.04, 731.01, 730.97, 730.89, 730.91, 730.87, 730.88, 730.89, 730.86, 730.83, 730.83, 730.86, 730.77, 730.86, 730.78, 730.73, 730.75, 730.69, 730.76, 730.69, 730.67, 730.58, 730.56, 730.59, 730.60, 730.55, 730.59, 730.65, 730.62, 730.69, 730.65, 730.64, 730.62, 730.61, 730.60, 730.56, 730.47, 730.47, 730.42, 730.38, 730.33, 730.32, 730.30, 730.26])
time_river = pd.to_datetime(time_river_str, utc=True)  # 统一 UTC（与你 PSD 的 UTC 对齐）
deepth_river = height_river - 710.6  # 单位：m

# ---- Processing time range (UTC) ----
USE_AUTO_TIME_RANGE = True  # False
T_GLOBAL_START =  UTCDateTime("2025-06-18T03:00:00")
N_HOURS = 24 * 1

# ---- Band index frequency range (for proxy display) ----
BAND_FMIN = 2.0
BAND_FMAX = 30.0  # <= Nyquist

# ---- Heatmap display range ----
HEATMAP_FMIN_SHOW = BAND_FMIN
HEATMAP_FMAX_SHOW = BAND_FMAX

# ---- Inversion frequency band (IMPORTANT: Luong model typical <= 20 Hz) ----
INV_FMIN = 2.0
INV_FMAX = 30.0

# ---- Luong Welch parameters ----
NPERSEG_LUONG = 2**14
STEP = NPERSEG_LUONG // 2  # 50% overlap => step = 8192 samples = 81.92 s for fs=100

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
OUT_CSV_QB_1MIN      = "BedloadFlux_qb_Strict1min.csv"


# =============================================================================
# 1) 自定义函数
# =============================================================================

def detrend_demean_stream(st_in):
    st = st_in.copy()
    try:
        st.merge(method=1, fill_value="interpolate")
    except Exception:
        pass
    st.detrend("linear")
    st.detrend("demean")
    return st


def remove_response_to_velocity(st_in, paz):
    st = st_in.copy()
    fs = st[0].stats.sampling_rate
    fN = fs / 2.0
    pre_filt = (0.2, 0.5, 0.7 * fN, 0.9 * fN)
    for tr in st:
        tr.simulate(paz_remove=paz, remove_sensitivity=True, pre_filt=pre_filt)
    return st, pre_filt


def sliding_welch_psd(x, fs, nperseg, step):
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

        psd_list.append(Pxx)
        times_center[i] = np.datetime64("1970-01-01")  # placeholder

    psd_arr = np.vstack(psd_list)
    return f, psd_arr, times_center


def minute_median_psd(f, psd_arr, times_center_dt64):
    df = pd.DataFrame(psd_arr)
    # 关键：强制 UTC（tz-aware），与 time_river 统一
    df["minute"] = pd.to_datetime(times_center_dt64, utc=True).floor("min")
    psd_min = df.groupby("minute").median()
    psd_min.index.name = "minute_utc"
    psd_min.columns = f
    return psd_min


def band_index_from_psd(psd_minute_df, fmin, fmax):
    f = psd_minute_df.columns.to_numpy(dtype=float)
    idx = (f >= fmin) & (f <= fmax)
    if not np.any(idx):
        raise ValueError(f"[ERROR] Band {fmin}-{fmax} Hz has no freq bins.")

    band_mean = psd_minute_df.loc[:, idx].mean(axis=1)
    band_int = psd_minute_df.loc[:, idx].apply(lambda r: np.trapezoid(r.values, f[idx]), axis=1)

    out = pd.DataFrame({
        f"PSD_{fmin:g}_{fmax:g}_mean": band_mean,
        f"PSD_{fmin:g}_{fmax:g}_int": band_int
    })
    out.index.name = "minute_utc"
    return out


def align_river_to_minutes(psd_minutes_index, time_river, depth_river):
    """Interpolate river depth to PSD minute grid (strict 1-min time axis)."""
    river = pd.Series(depth_river, index=time_river).sort_index()
    
    river_1min = river.resample("1min").mean()
    river_1min = river_1min.interpolate(
        method="time",
        limit=180,                 # 允许 3 小时插值
        limit_direction="both"
    )
    
    river_on_psd = river_1min.reindex(psd_minutes_index)
    river_on_psd = river_on_psd.interpolate(
        method="time",
        limit=180,
        limit_direction="both"
    )
    return river_on_psd


def safe_db(psd_linear, floor=1e-30):
    psd_linear = np.asarray(psd_linear, dtype=float)
    psd_linear = np.maximum(psd_linear, floor)
    return 10.0 * np.log10(psd_linear)


def quick_plot_full_trace_minmax(tr, max_bins=200_000, title="Full trace (min-max quicklook)"):
    """
    Fast overview using min-max per bin (preserves spikes).
    max_bins: number of bins along time axis (each bin contributes 2 points).
    """
    x = np.asarray(tr.data)
    n = x.size
    if n == 0:
        print("[WARN] Empty trace.")
        return

    # each bin -> 2 points (min,max), so bins ~ max_bins
    bin_size = max(1, n // max_bins)
    nb = n // bin_size

    x2 = x[:nb * bin_size].reshape(nb, bin_size)
    xmin = np.nanmin(x2, axis=1)
    xmax = np.nanmax(x2, axis=1)

    # interleave min/max to draw envelope
    y = np.empty(nb * 2, dtype=float)
    y[0::2] = xmin
    y[1::2] = xmax

    fs = tr.stats.sampling_rate
    t = (np.arange(nb) * bin_size) / fs / 3600.0
    t2 = np.repeat(t, 2)

    plt.figure(figsize=(14, 4))
    plt.plot(t2, y, linewidth=0.6)
    plt.xlabel("Time (hours from start)")
    plt.ylabel("Amplitude")
    plt.title(f"{title} | n={n}, bin_size={bin_size}, bins={nb}, plotted={y.size}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    


# =============================================================================
# 2) 数据读取及预处理
# =============================================================================

st = read(DATA_FILE)
tr_full = st[0]
fs = tr_full.stats.sampling_rate
fN = fs / 2.0

print("\n[INFO] Trace header:")
print(tr_full)
print(f"[INFO] fs={fs} Hz, Nyquist={fN} Hz")

if BAND_FMAX > fN:
    raise ValueError(f"[ERROR] BAND_FMAX={BAND_FMAX} exceeds Nyquist={fN}.")

if USE_AUTO_TIME_RANGE:
    T_GLOBAL_START = tr_full.stats.starttime
    T_GLOBAL_END = tr_full.stats.endtime
    N_HOURS = int((T_GLOBAL_END - T_GLOBAL_START) // 3600)
    print(f"[INFO] Auto time range: {T_GLOBAL_START} -> {T_GLOBAL_END}")
    print(f"[INFO] Auto N_HOURS = {N_HOURS}")
else:
    if T_GLOBAL_START is None:
        raise ValueError("[ERROR] Set T_GLOBAL_START when USE_AUTO_TIME_RANGE=False.")
    T_GLOBAL_END = T_GLOBAL_START + N_HOURS * 3600

if T_GLOBAL_START < tr_full.stats.starttime or T_GLOBAL_END > tr_full.stats.endtime:
    raise ValueError(
        "[ERROR] Requested time range is out of data coverage.\n"
        f"Data: {tr_full.stats.starttime} ~ {tr_full.stats.endtime}\n"
        f"Req : {T_GLOBAL_START} ~ {T_GLOBAL_END}"
    )

# quick look full data
'''
# raw
tr_raw_all = st[0].copy()

# detrend + demean (Stream → Stream)
st_det_all = detrend_demean_stream(st.copy())
tr_det_all = st_det_all[0]

# remove response (Stream → Stream)
st_vel_all, _ = remove_response_to_velocity(st_det_all, PAZ)
tr_vel_all = st_vel_all[0]

st_qc = Stream([tr_raw_all.copy(),tr_det_all.copy(),tr_vel_all.copy()])

# rename channels for display
#st_qc[0].stats.channel = "RAW"
#st_qc[1].stats.channel = "VEL"
#st_qc[2].stats.channel = "DETREND"

st_qc.plot(
    equal_scale=False,
    type="normal",
    linewidth=0.6,
    size=(1400, 500),
    tick_format="%m-%d",
    title="OBS waveform QC: raw / detrended / velocity",
)

'''
# =============================================================================
# 3) 主循环：逐小时数据处理
# =============================================================================

all_minutes_band = []
all_minutes_fullspec = []
f_ref = None

for ih in range(N_HOURS):
    t1 = T_GLOBAL_START + ih * 3600
    t2 = t1 + 3600

    st_seg = st.copy()
    st_seg.trim(t1, t2)

    if len(st_seg) == 0 or st_seg[0].stats.npts < NPERSEG_LUONG:
        npts = st_seg[0].stats.npts if len(st_seg) > 0 else 0
        print(f"[WARN] Empty/too short segment: {t1} - {t2}, npts={npts}, skip.")
        continue

    st_seg = detrend_demean_stream(st_seg)
    st_vel, _ = remove_response_to_velocity(st_seg, PAZ)

    trv = st_vel[0]
    x = trv.data.astype(np.float64)
    x -= np.mean(x)

    f, psd_arr, times_center = sliding_welch_psd(x, fs, NPERSEG_LUONG, STEP)
    if f is None:
        print(f"[WARN] nwin<=0: {t1} - {t2}, skip.")
        continue

    # fill correct center times
    nwin = psd_arr.shape[0]
    t0 = trv.stats.starttime
    for i in range(nwin):
        i0 = i * STEP
        tc = t0 + (i0 + NPERSEG_LUONG / 2) / fs
        times_center[i] = np.datetime64(tc.datetime)

    psd_minute = minute_median_psd(f, psd_arr, times_center)

    if f_ref is None:
        f_ref = psd_minute.columns.to_numpy(dtype=float)
    else:
        f_now = psd_minute.columns.to_numpy(dtype=float)
        if len(f_now) != len(f_ref) or np.max(np.abs(f_now - f_ref)) > 1e-12:
            raise ValueError("[ERROR] Frequency axis changed between hours. Check fs/Welch settings.")

    band_df = band_index_from_psd(psd_minute, BAND_FMIN, BAND_FMAX)

    all_minutes_fullspec.append(psd_minute)
    all_minutes_band.append(band_df)

    print(f"[INFO] {t1} - {t2}: Luong-minute points = {band_df.shape[0]}")

if len(all_minutes_band) == 0:
    raise RuntimeError("[ERROR] No valid hourly segments processed. Check data continuity/time range.")


# =============================================================================
# 4) 合并数据，求解PSD
# =============================================================================

out_band = pd.concat(all_minutes_band).sort_index()
out_band = out_band.groupby(out_band.index).median()
out_band_1min = out_band.resample("1min").median()

out_band.to_csv(OUT_CSV_LUONG_MINUTE)
out_band_1min.to_csv(OUT_CSV_STRICT_1MIN)

print("\n[INFO] Saved:")
print(" -", OUT_CSV_LUONG_MINUTE, "(Luong-minute)")
print(" -", OUT_CSV_STRICT_1MIN, "(strict 1-min)")

psd_all = pd.concat(all_minutes_fullspec).sort_index()
psd_all = psd_all.groupby(psd_all.index).median()
psd_all_1min = psd_all.resample("1min").median()
psd_all_1min_plot = psd_all_1min.interpolate(limit=10, limit_direction="both")


# =============================================================================
# 5) 流深数据与PSD时间网格对齐（严格1分钟）
# =============================================================================

river_on_minute = align_river_to_minutes(psd_all_1min.index, time_river, deepth_river)

# =============================================================================
# 6) 推移质反演 (seismic_bedload SaltationModel)
# =============================================================================
# IMPORTANT:
# - The model expects PSD_obs in dB (it converts back internally by 10**(dB/10)).
# - Use frequency band <= 20 Hz for inversion, consistent with example usage.

f_all = psd_all_1min.columns.to_numpy(dtype=float)

inv_band = (f_all >= INV_FMIN) & (f_all <= INV_FMAX)
if not np.any(inv_band):
    raise ValueError(f"[ERROR] Inversion band {INV_FMIN}-{INV_FMAX} Hz has no bins. Check nperseg/fs.")

# PSD_obs_dB: median over inversion band for each minute
P_lin = psd_all_1min.to_numpy(dtype=float)
PSD_dB = safe_db(P_lin, floor=1e-30)
PSD_obs_dB = np.nanmedian(PSD_dB[:, inv_band], axis=1)  # (ntime,)

# Prepare H (flow depth) aligned
H = river_on_minute.values.astype(float)

mask = np.isfinite(PSD_obs_dB) & np.isfinite(H)
PSD_obs_dB_use = PSD_obs_dB[mask]
H_use = H[mask]

# Frequencies for inversion (must match PSD_obs definition)
f_inv = f_all[inv_band]

print("\n[INFO] Inversion inputs:")
print("[INFO] PSD_obs_dB range:", np.nanmin(PSD_obs_dB_use), np.nanmax(PSD_obs_dB_use))
print("[INFO] H range (m):", np.nanmin(H_use), np.nanmax(H_use))
print("[INFO] f_inv range:", f_inv[0], "to", f_inv[-1], "Hz, nfreq=", f_inv.size)

# ---- Model parameters (you can tune later) ----
D = 0.075  # 单一粒径代表值
D50 = 0.075  # 中值粒径
sigma = 0.52  # σ 越大 → 粒径分布越“宽”（粗细颗粒并存）
mu = 0.15  # 控制粒径分布“峰值”在粒径轴上的位置。
s = sigma / np.sqrt(1/3 - 2/np.pi**2)  # raised-cosine 分布的尺度参数
pD = log_raised_cosine_pdf(D, mu, s) / D  # 粒径分布的 概率密度函数（pdf）
W = 164-19.1  # 河宽 m，175-14.8
theta = np.tan(1.4*np.pi/180)  # 河床坡度（无量纲1.4°）
r0 = 600.0  # 从“激发源区（颗粒撞击/床载活动）”到传感器的有效传播距离尺度（m）
rho_s = 2650.0  # 推移质密度kg/m3
qb0 = 10/rho_s  # 初始猜测的床载输沙率（m²/s）
tau_c50 = 0.045  # 对 D50 颗粒的临界无量纲切应力，越大 → 越难起动

model = SaltationModel()

qb_use = model.inverse_bedload(
    PSD_obs_dB_use, f_inv, D, H_use, W, theta, r0, qb0,
    D50=D50, tau_c50=tau_c50, pdf=pD
)

# Put back to full 1-min timeline
qb_1min = pd.Series(np.nan, index=psd_all_1min.index, name="qb_m2s")
qb_1min.loc[psd_all_1min.index[mask]] = qb_use

qb_df = pd.DataFrame(index=qb_1min.index)

qb_df["qb_vol_m2s"] = qb_1min                     # m^2/s,体积通量（单位河宽）
qb_df["qb_mass_kgms"] = qb_df["qb_vol_m2s"] * rho_s   # kg/(m·s),质量通量（单位河宽）
qb_df["Qb_mass_kgps"] = qb_df["qb_mass_kgms"] * W     # kg/s，全河宽床载通量

#qb_df.to_csv(OUT_CSV_QB_1MIN)

print("[INFO] Saved:", OUT_CSV_QB_1MIN)
print("[INFO] qb_vol range (m^2/s):", np.nanmin(qb_df["qb_vol_m2s"]), np.nanmax(qb_df["qb_vol_m2s"]))
print("[INFO] Qb_mass range (kg/s):", np.nanmin(qb_df["Qb_mass_kgps"]), np.nanmax(qb_df["Qb_mass_kgps"]))

# =============================================================================
# 7) Plot: PSD heatmap (dB) + river depth (black) + qb (optional)
# =============================================================================

# Heatmap data in dB (freq x time)
P_hm = psd_all_1min_plot.values.T  # (nfreq, ntime)
P_db = safe_db(P_hm, floor=1e-30)

# Limit frequency display
f_show = psd_all_1min_plot.columns.to_numpy(dtype=float)
freq_mask_show = (f_show >= HEATMAP_FMIN_SHOW) & (f_show <= HEATMAP_FMAX_SHOW)

P_db_show = P_db[freq_mask_show, :]
f_show_use = f_show[freq_mask_show]

# Robust color scale by percentiles (avoid huge colorbar range)
vmin = np.nanpercentile(P_db_show, 5)
vmax = np.nanpercentile(P_db_show, 95)

#vmin = -225
#vmax = -200

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
    cmap="viridis"
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

# ---- Overlay river depth (black) on right axis ----
ax_river = ax_hm.twinx()
ax_river.plot(psd_all_1min.index, river_on_minute.values, color="black", linewidth=2.0, label="River depth", zorder=10)
ax_river.set_ylabel("River depth (m)", color="black")
ax_river.tick_params(axis="y", labelcolor="black")
ax_river.legend(loc="upper left", frameon=False)

plt.tight_layout()


# =============================================================================
# 8) Plot: qb time series (optional)
# =============================================================================

fig_qb, ax_qb = plt.subplots(figsize=(14, 4.2))
ax_qb.plot(qb_df.index, qb_df["Qb_mass_kgps"].values, linewidth=0.8)
#ax_qb.set_yscale("log")
ax_qb.set_xlabel("Time (UTC)")
ax_qb.set_ylabel(r"$q_b$ (kg m$^{-1}$s$^{-1}$)")
ax_qb.set_title("Inverted bedload flux (SaltationModel, strict 1-min)")
ax_qb.grid(True, which="both", alpha=0.25)
fig_qb.autofmt_xdate()

print("[DEBUG] psd tz:", psd_all_1min.index.tz, "river tz:", time_river.tz)

plt.tight_layout()
plt.show()
