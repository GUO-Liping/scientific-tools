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

from obspy import read, UTCDateTime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import welch

# ---- seismic-bedload ----
from seismic_bedload import SaltationModel
from seismic_bedload.utils import log_raised_cosine_pdf


# =============================================================================
# 0) USER CONFIG (edit here)
# =============================================================================

# ---- Input BHZ SAC file ----
DATA_FILE = r"E:\\项目3-YJ项目-推移质监测\\雅江YaJiang-推移质OBS监测数据\\101MT\\C-00001_250618\\65A42E04.15A.BHZ"

# ---- Water level/depth series ----
time_river_str = [
    "2025-06-18 00:00", "2025-06-18 01:00", "2025-06-18 02:00", "2025-06-18 03:00",
    # ... (保持你原来的列表不变，这里略) ...
    "2025-06-26 23:00"
]
height_river = np.array([
    729.70, 729.82, 729.81, 729.84,
    # ... (保持你原来的数组不变，这里略) ...
    730.26
])

time_river = pd.to_datetime(time_river_str, utc=True)  # UTC
deepth_river = height_river - 710.6  # m

# ---- Processing time range (UTC) ----
USE_AUTO_TIME_RANGE = True
T_GLOBAL_START = None
N_HOURS = 24 * 1

# ---- Band index frequency range (for proxy display) ----
BAND_FMIN = 2.0
BAND_FMAX = 30.0  # <= Nyquist

# ---- Heatmap display range ----
HEATMAP_FMIN_SHOW = BAND_FMIN
HEATMAP_FMAX_SHOW = BAND_FMAX

# ---- Inversion frequency band (IMPORTANT: Luong model typical <= 20 Hz) ----
INV_FMIN = 5.0
INV_FMAX = 15.0

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
# 1) Helpers (procedural)
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
    df["minute"] = pd.to_datetime(times_center_dt64).floor("min")
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


# =============================================================================
# 2) Read data + determine processing range
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


# =============================================================================
# 3) Main loop: hourly processing
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
# 4) Merge + export PSD products
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
# 5) River depth align to PSD minute grid (strict 1-min)
# =============================================================================

river_on_minute = align_river_to_minutes(psd_all_1min.index, time_river, deepth_river)


# =============================================================================
# 6) Bedload flux inversion (seismic_bedload SaltationModel)
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
D = 0.3
D50 = 0.4
sigma = 0.52
mu = 0.15
s = sigma / np.sqrt(1/3 - 2/np.pi**2)
pD = log_raised_cosine_pdf(D, mu, s) / D

W = 50.0
theta = np.tan(1.4*np.pi/180)
r0 = 600.0

qb0 = 1e-3
tau_c50 = 0.045  # typical; you may calibrate

model = SaltationModel()

qb_use = model.inverse_bedload(
    PSD_obs_dB_use, f_inv, D, H_use, W, theta, r0, qb0,
    D50=D50, tau_c50=tau_c50, pdf=pD
)

# Put back to full 1-min timeline
qb_1min = pd.Series(np.nan, index=psd_all_1min.index, name="qb_m2s")
qb_1min.loc[psd_all_1min.index[mask]] = qb_use

qb_df = pd.DataFrame({"qb_m2s": qb_1min})
qb_df.to_csv(OUT_CSV_QB_1MIN)

print("[INFO] Saved:", OUT_CSV_QB_1MIN)
print("[INFO] qb range (m^2/s):", np.nanmin(qb_use), np.nanmax(qb_use))


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

cbar_hm = fig_hm.colorbar(im_hm, ax=ax_hm, pad=0.01)
cbar_hm.set_label(r"PSD (dB rel. (m/s)$^2$/Hz)")

ax_hm.set_title("Minute-median PSD heatmap (strict 1-min grid)")

# ---- Overlay river depth (black) on right axis ----
ax_river = ax_hm.twinx()
ax_river.plot(psd_all_1min.index, river_on_minute.values, color="black", linewidth=2.0, label="River depth", zorder=10)
ax_river.set_ylabel("River depth (m)", color="black")
ax_river.tick_params(axis="y", labelcolor="black")
ax_river.legend(loc="upper right", frameon=False)

plt.tight_layout()
plt.show()


# =============================================================================
# 8) Plot: qb time series (optional)
# =============================================================================

fig_qb, ax_qb = plt.subplots(figsize=(12, 3))
ax_qb.plot(qb_df.index, qb_df["qb_m2s"].values, linewidth=0.8)
ax_qb.set_yscale("log")
ax_qb.set_xlabel("Time (UTC)")
ax_qb.set_ylabel(r"$q_b$ (m$^2$/s)")
ax_qb.set_title("Inverted bedload flux (SaltationModel, strict 1-min)")
ax_qb.grid(True, which="both", alpha=0.25)
fig_qb.autofmt_xdate()
plt.tight_layout()
plt.show()
