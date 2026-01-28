from obspy import read, UTCDateTime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import welch

import pandas as pd


# 读取地震波形数据
# PZ水文站obs设备4通道数据
pz_st1_1 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250627\\65B693D8.159.BHE"
pz_st1_2 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250627\\65B693D8.159.BHN"
pz_st1_3 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250627\\65B693D8.159.BHZ"
pz_st1_4 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250627\\65B693D8.159.HYD"

pz_st2_1 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250630\\65BC1C12.158.BHE"
pz_st2_2 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250630\\65BC1C12.158.BHN"
pz_st2_3 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250630\\65BC1C12.158.BHZ"
pz_st2_4 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250630\\65BC1C12.158.HYD"

pz_st3_1 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250719\\65E6239E.159.BHE"
pz_st3_2 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250719\\65E6239E.159.BHN"
pz_st3_3 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250719\\65E6239E.159.BHZ"
pz_st3_4 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250719\\65E6239E.159.HYD"

pz_st4_1 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250720\\65E87904.157.BHE"
pz_st4_2 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250720\\65E87904.157.BHN"
pz_st4_3 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250720\\65E87904.157.BHZ"
pz_st4_4 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250720\\65E87904.157.HYD"

pz_st5_1 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250721\\65EA9107.157.BHE"
pz_st5_2 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250721\\65EA9107.157.BHN"
pz_st5_3 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250721\\65EA9107.157.BHZ"
pz_st5_4 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250721\\65EA9107.157.HYD"

pz_st6_1 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250807\\660E6BB7.159.BHE"
pz_st6_2 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250807\\660E6BB7.159.BHN"
pz_st6_3 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250807\\660E6BB7.159.BHZ"
pz_st6_4 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250807\\660E6BB7.159.HYD"

pz_st7_1 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250808\\661169B4.158.BHE"
pz_st7_2 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250808\\661169B4.158.BHN"
pz_st7_3 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250808\\661169B4.158.BHZ"
pz_st7_4 = "F:\\YaJiang_OBS_data\\PZ_102_192_168_12_12\\C-00002_250808\\661169B4.158.HYD"


# MT水文站ob4通道数据
mt_st1_1 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250616\\65A063C7.17C.BHE"
mt_st1_2 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250616\\65A063C7.17C.BHN"
mt_st1_3 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250616\\65A063C7.17C.BHZ"
mt_st1_4 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250616\\65A063C7.17C.HYD"

mt_st2_1 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250618\\65A42E04.15A.BHE"
mt_st2_2 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250618\\65A42E04.15A.BHN"
mt_st2_3 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250618\\65A42E04.15A.BHZ"
mt_st2_4 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250618\\65A42E04.15A.HYD"

mt_st3_1 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250626\\65B44BE1.15A.BHE"
mt_st3_2 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250626\\65B44BE1.15A.BHN"
mt_st3_3 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250626\\65B44BE1.15A.BHZ"
mt_st3_4 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250626\\65B44BE1.15A.HYD"

mt_st4_1 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250630\\65BC19EF.158.BHE"
mt_st4_2 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250630\\65BC19EF.158.BHN"
mt_st4_3 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250630\\65BC19EF.158.BHZ"
mt_st4_4 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250630\\65BC19EF.158.HYD"

mt_st5_1 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250807\\660E6AC4.158.BHE"
mt_st5_2 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250807\\660E6AC4.158.BHN"
mt_st5_3 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250807\\660E6AC4.158.BHZ"
mt_st5_4 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250807\\660E6AC4.158.HYD"

mt_st6_1 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250808\\6610BBF3.158.BHE"
mt_st6_2 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250808\\6610BBF3.158.BHN"
mt_st6_3 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250808\\6610BBF3.158.BHZ"
mt_st6_4 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250808\\6610BBF3.158.HYD"

mt_st7_1 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250809\\661207A8.160.BHE"
mt_st7_2 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250809\\661207A8.160.BHN"
mt_st7_3 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250809\\661207A8.160.BHZ"
mt_st7_4 = "F:\\YaJiang_OBS_data\\MT_101_192_168_11_11\\C-00001_250809\\661207A8.160.HYD"

pz_data_BHE = "E:\\项目3-YJ项目-推移质监测\\雅江YaJiang-推移质OBS监测数据\\102\\C-00002_250630\\65BC1C12.158.BHE"
pz_data_BHN = "E:\\项目3-YJ项目-推移质监测\\雅江YaJiang-推移质OBS监测数据\\102\\C-00002_250630\\65BC1C12.158.BHN"
pz_data_BHZ = "E:\\雅江OBS数据\\65BC1C12.158.BHZ"
pz_data_HYD = "E:\\项目3-YJ项目-推移质监测\\雅江YaJiang-推移质OBS监测数据\\102\\C-00002_250630\\65BC1C12.158.HYD"

# 选择数据
st = read(pz_data_BHZ)

# 打印头部信息
print(st[0].stats)
print(st[0])
# 绘制OBS数据
fig = st.plot(handle=True, color='k', equal_scale=False, linewidth=0.5)
fig.set_size_inches(12, 4)
fig.subplots_adjust(top=0.855, bottom=0.2, left=0.1, right=0.95, hspace=0.0, wspace=0.2)

# 设置时间刻度
ax = fig.axes[0]
ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))  # 主刻度：每12小时
ax.xaxis.set_minor_locator(mdates.HourLocator(interval=2))  # 次刻度：每2小时
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))

fig.autofmt_xdate()
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Amplitude")


# =========================
# 去除仪器响应（counts -> m/s）
# =========================
# 1小时数据
t_start_utc = UTCDateTime("2025-07-05T03:00:00")  # <<< 改这里
t_end_utc   = t_start_utc + 3600
st_1h = st.copy()
st_1h.trim(t_start_utc, t_end_utc)

# 1) 合并/去趋势（长记录建议merge后再处理，避免间断影响）
st_1h.merge(method=1, fill_value="interpolate")
st_1h.detrend("linear")
st_1h.detrend("demean")

# 2) 你的检波器传递函数 + ADC（poles/zeros 以 rad/s 输入）
ADC_S = 1.6777e6  # counts/V

paz = {
    "poles": [-30.61, -15.8, -0.1693, -0.09732, -0.03333],
    "zeros": [0j, 0j, 0j, -30.21, -16.15],
    "gain": 1021.9,                     # V/(m/s)
    "sensitivity": 1021.9 * ADC_S       # counts/(m/s)
}

# 3) pre_filt：根据采样率自动设置上限（Nyquist=fs/2）
fs = st_1h[0].stats.sampling_rate
fN = fs / 2.0
pre_filt = (0.2, 0.5, 0.7 * fN, 0.9 * fN)  # (0.2,0.5,35,45) for fs=100

# 4) 对每条 Trace 去响应，输出地面速度（m/s）
for tr in st_1h:
    tr.simulate(paz_remove=paz, remove_sensitivity=True, pre_filt=pre_filt)

# 5) 自检（可选）
print(f"[INFO] Removed response -> velocity. fs={fs} Hz, Nyquist={fN} Hz, pre_filt={pre_filt}")
print("[INFO] finite:", np.isfinite(st_1h[0].data).all(), "st_1hd(m/s)=", np.std(st_1h[0].data))

# 重新画一次：去响应后的速度波形
fig2 = st_1h.plot(handle=True, color='k', equal_scale=False, linewidth=0.5)
fig2.set_size_inches(12, 4)
fig2.subplots_adjust(top=0.855, bottom=0.2, left=0.1, right=0.95, hspace=0.0, wspace=0.2)

ax2 = fig2.axes[0]
ax2.xaxis.set_major_locator(mdates.HourLocator(interval=12))
ax2.xaxis.set_minor_locator(mdates.HourLocator(interval=2))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
fig2.autofmt_xdate()
ax2.set_xlabel("Time (UTC)")
ax2.set_ylabel("Velocity (m/s)")

'''
# 时频图绘制
# 定义时间窗的开始和结束时间
start_time = st[0].stats.starttime + 5.999*24*60*60  # 起始时间：10秒
end_time = st[0].stats.starttime + 6.001*24*60*60    # 结束时间：20秒
st.trim(starttime=start_time, endtime=end_time)
st.spectrogram(log=False, title='BW.RJOB ' + str(st[0].stats.starttime))
'''

# ===========================================================================
# 取 1 小时数据（去响应后的速度 st 已经是 m/s）
# ===========================================================================
tr1h = st_1h[0]
fs = tr1h.stats.sampling_rate
x = tr1h.data.astype(np.float64)

print("[INFO] 1h segment:", tr1h.stats.starttime, "to", tr1h.stats.endtime,
      "npts=", tr1h.stats.npts, "fs=", fs)

# 可选：再做一次轻微去均值，避免窗口边界偏置
x = x - np.mean(x)

# ===========================================================================
# Welch PSD（探索阶段建议 4096 或 8192）
# ===========================================================================
nperseg = 8192           # 81.92 s
noverlap = nperseg // 2  # 50% overlap

f, Pxx = welch(
    x, fs=fs, window="hann",
    nperseg=nperseg, noverlap=noverlap,
    detrend=False, scaling="density"
)

# ===========================================================================
# 提取 30–45 Hz 指标（fs=100Hz 情况下）
# ===========================================================================
fmin, fmax = 30.0, 45.0
idx = (f >= fmin) & (f <= fmax)

# 频带平均（也可用积分 np.trapezoid(Pxx[idx], f[idx])）
psd_band_mean = np.mean(Pxx[idx])
psd_band_int  = np.trapezoid(Pxx[idx], f[idx])

print(f"[INFO] PSD band {fmin}-{fmax} Hz mean = {psd_band_mean:.3e} (m/s)^2/Hz")
print(f"[INFO] PSD band {fmin}-{fmax} Hz integral = {psd_band_int:.3e} (m/s)^2")

# ===========================================================================
# 绘图：1小时波形 + PSD（loglog）
# ===========================================================================
# 1小时波形
figw, axw = plt.subplots(figsize=(12, 3))
t = np.arange(tr1h.stats.npts) / fs
axw.plot(t, x, linewidth=0.4)
axw.set_xlabel("Time (s)")
axw.set_ylabel("Velocity (m/s)")
axw.set_title(f"Velocity waveform (1h) {t_start_utc} - {t_end_utc}")

# PSD
figp, axp = plt.subplots(figsize=(6.5, 4))
axp.loglog(f, Pxx, linewidth=0.8)
axp.set_xlabel("Frequency (Hz)")
axp.set_ylabel(r"PSD ((m/s)$^2$/Hz)")
axp.set_title("Welch PSD (1h segment)")

# 标出 30–45 Hz
axp.axvline(fmin, linestyle="--", linewidth=0.8)
axp.axvline(fmax, linestyle="--", linewidth=0.8)

# ==========================================================================================
# 下一步（Luong 风格）：1 小时数据 -> 滑动 Welch(nperseg=2^14, 50% overlap) -> 1-min median PSD
# 并输出每分钟的 30–45 Hz 频带指标（受 fs=100 Hz 限制）
# ==========================================================================================

# 输入仍然是去响应后的 1 小时速度数据
tr1h = st_1h[0]
fs = tr1h.stats.sampling_rate
x = tr1h.data.astype(np.float64)
x = x - np.mean(x)

# Luong Welch 参数
nperseg_luong = 2**14
noverlap_luong = nperseg_luong // 2
step = nperseg_luong - noverlap_luong

if len(x) < nperseg_luong:
    raise ValueError(
        f"[ERROR] 1h data too short for Luong nperseg=2^14 ({nperseg_luong} samples). "
        f"Need >= {nperseg_luong}, but got {len(x)}."
    )

nwin = (len(x) - nperseg_luong) // step + 1
print(f"[INFO] Luong-Welch windows = {nwin}, win_len={nperseg_luong/fs:.2f}s, step={step/fs:.2f}s")

# 逐窗 PSD + 记录窗口中心时间（用于分钟分组）
t0 = tr1h.stats.starttime
times_center = np.empty(nwin, dtype="datetime64[ns]")
psd_list = []

for i in range(nwin):
    i0 = i * step
    seg = x[i0:i0 + nperseg_luong]

    # 单段 Welch（noverlap=0），窗间 overlap 通过 step 实现
    f_l, Pxx_l = welch(
        seg, fs=fs, window="hann",
        nperseg=nperseg_luong, noverlap=0,
        detrend=False, scaling="density"
    )

    tc = t0 + (i0 + nperseg_luong / 2) / fs
    times_center[i] = np.datetime64(tc.datetime)
    psd_list.append(Pxx_l)

psd_arr = np.vstack(psd_list)  # shape: (nwin, nfreq)

# 按分钟聚合：每个频点取 median（Luong 的 minute median PSD）
df_psd = pd.DataFrame(psd_arr)
df_psd["minute"] = pd.to_datetime(times_center).floor("min")

psd_minute = df_psd.groupby("minute").median()
psd_minute.index.name = "minute_utc"

print("[INFO] Minute-median PSD shape:", psd_minute.shape, "(n_minutes, n_freq)")

# 频带指标（fs=100 -> Nyquist=50；推荐 30–45 Hz）
fmin, fmax = 30.0, 45.0
idx = (f_l >= fmin) & (f_l <= fmax)
cols = np.where(idx)[0]

PSD_30_45_mean = psd_minute.iloc[:, cols].mean(axis=1)
PSD_30_45_int  = psd_minute.iloc[:, cols].apply(lambda row: np.trapezoid(row.values, f_l[idx]), axis=1)

out_minute = pd.DataFrame({
    "PSD_30_45_mean": PSD_30_45_mean,
    "PSD_30_45_int": PSD_30_45_int
})
print("[INFO] Minute PSD (30–45 Hz) preview:")
print(out_minute.head())

# 保存：分钟级指标（反演用）+ 分钟级完整谱（检查用）
out_csv = "PSD_1h_1min_PSD30_45Hz_Luong.csv"
full_csv = "PSD_1h_1min_full_spectrum_Luong.csv"
out_minute.to_csv(out_csv)
psd_minute.to_csv(full_csv)

print("[INFO] Saved:")
print(" -", out_csv)
print(" -", full_csv)

# （可选）快速画一下 30–45 Hz 的分钟序列（log y 更清楚）
figm, axm = plt.subplots(figsize=(10, 3))
axm.plot(out_minute.index.astype("datetime64[ns]"), out_minute["PSD_30_45_mean"].values, linewidth=1.0)
axm.set_yscale("log")
axm.set_xlabel("Time (UTC, minute)")
axm.set_ylabel(r"Median PSD mean (30–45 Hz) ((m/s)$^2$/Hz)")
axm.set_title("1-min median PSD (Luong-style, 30–45 Hz)")
figm.autofmt_xdate()


# ==========================================================================================
# 下一步：扩展到 24 小时（或多天）——按小时循环计算 Luong-style 1-min median PSD 指标
# 输出：每分钟 PSD_30_45_mean / PSD_30_45_int 的连续时间序列
# ==========================================================================================

# --------- 你只改这两行 ----------
t_global_start = UTCDateTime("2025-07-05T03:00:00")  # 起点（UTC）
n_hours = 24*7                                         # 连续小时数（24=1天；要多天就 24*天数）
# ---------------------------------

# 预先准备：PAZ 与 pre_filt（对所有小时都一样）
ADC_S = 1.6777e6
paz = {
    "poles": [-30.61, -15.8, -0.1693, -0.09732, -0.03333],
    "zeros": [0j, 0j, 0j, -30.21, -16.15],
    "gain": 1021.9,
    "sensitivity": 1021.9 * ADC_S
}

# Luong Welch 参数
nperseg_luong = 2**14
noverlap_luong = nperseg_luong // 2
step = nperseg_luong - noverlap_luong

# 频带（受 fs=100Hz 限制）
fmin, fmax = 10.0, 45.0

all_minutes = []  # 用来存每小时结果（DataFrame）

for ih in range(n_hours):
    t1 = t_global_start + ih * 3600
    t2 = t1 + 3600

    # 1) 从原始整段 st 中裁剪出 1 小时（注意：这里 st 还是 counts）
    st_seg = st.copy()
    st_seg.trim(t1, t2)

    if len(st_seg) == 0 or st_seg[0].stats.npts < 1000:
        print(f"[WARN] Empty/too short segment: {t1} - {t2}, skip.")
        continue

    # 2) 合并/去趋势
    st_seg.merge(method=1, fill_value="interpolate")
    st_seg.detrend("linear")
    st_seg.detrend("demean")

    # 3) 去仪器响应 -> 速度
    fs = st_seg[0].stats.sampling_rate
    fN = fs / 2.0
    pre_filt = (0.2, 0.5, 0.7 * fN, 0.9 * fN)

    for tr in st_seg:
        tr.simulate(paz_remove=paz, remove_sensitivity=True, pre_filt=pre_filt)

    trv = st_seg[0]
    x = trv.data.astype(np.float64)
    x = x - np.mean(x)

    # 如果该小时数据长度不足以做 2^14 Welch，就跳过
    if len(x) < nperseg_luong:
        print(f"[WARN] Segment too short for Luong nperseg: {t1} - {t2}, npts={len(x)}")
        continue

    # 4) 滑动 Welch（2^14, 50% overlap）+ 记录中心时刻
    nwin = (len(x) - nperseg_luong) // step + 1
    times_center = np.empty(nwin, dtype="datetime64[ns]")
    psd_list = []

    t0 = trv.stats.starttime

    for i in range(nwin):
        i0 = i * step
        seg = x[i0:i0 + nperseg_luong]

        f_l, Pxx_l = welch(
            seg, fs=fs, window="hann",
            nperseg=nperseg_luong, noverlap=0,
            detrend=False, scaling="density"
        )

        tc = t0 + (i0 + nperseg_luong / 2) / fs
        times_center[i] = np.datetime64(tc.datetime)
        psd_list.append(Pxx_l)

    psd_arr = np.vstack(psd_list)

    # 5) minute median PSD
    df_psd = pd.DataFrame(psd_arr)
    df_psd["minute"] = pd.to_datetime(times_center).floor("min")
    psd_minute = df_psd.groupby("minute").median()
    psd_minute.index.name = "minute_utc"

    # 6) 30–45 Hz 指标
    idx = (f_l >= fmin) & (f_l <= fmax)
    cols = np.where(idx)[0]

    out_minute = pd.DataFrame({
        "PSD_30_45_mean": psd_minute.iloc[:, cols].mean(axis=1),
        "PSD_30_45_int":  psd_minute.iloc[:, cols].apply(lambda row: np.trapezoid(row.values, f_l[idx]), axis=1)
    })
    out_minute.index.name = "minute_utc"

    all_minutes.append(out_minute)

    print(f"[INFO] {t1} - {t2}: minutes={out_minute.shape[0]} (saved in memory)")

# 7) 合并所有小时结果并保存
if len(all_minutes) > 0:
    out_all = pd.concat(all_minutes).sort_index()

    # 去重：如果相邻小时因为窗口中心落点导致分钟重复，取 median（或 mean 都行）
    out_all = out_all.groupby(out_all.index).median()
    # ✅ 关键：重采样成严格 1-min
    out_all_1min = out_all.resample("1min").median()

    # 保存 Luong-style minute（不严格 60s）
    out_all.to_csv("PSD_LuongMinute_PSD30_45Hz.csv")
    
    # 保存严格 1-min（工程/水文对齐）
    out_all_1min.to_csv("PSD_1min_PSD30_45Hz.csv")
    
    print("[INFO] Luong-minute points:", out_all.shape[0])
    print("[INFO] Strict 1-min points:", out_all_1min.shape[0])

    print("[INFO] Total minutes:", out_all.shape[0])
    print("[INFO] minute PSD is defined by Welch-window center time (not strict 60s bins)")

    # 可选：快速画一下
    fig_all, ax_all = plt.subplots(figsize=(12, 3))
    ax_all.plot(out_all_1min.index, out_all_1min["PSD_30_45_mean"])
    ax_all.set_yscale("log")
    ax_all.set_xlabel("Time (UTC, minute)")
    ax_all.set_ylabel(r"Median PSD mean (30–45 Hz) ((m/s)$^2$/Hz)")
    ax_all.set_title("Luong-style 1-min median PSD (30–45 Hz), 24h")
    fig_all.autofmt_xdate()
    plt.show()
else:
    print("[WARN] No valid hourly segments processed.")

plt.show()