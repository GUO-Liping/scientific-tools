# -*- coding: utf-8 -*-
"""
OBS bedload PSD processing (Luong-style) + paper-grade visualization
Procedural script (no OOP)

Paper figure (like your screenshot):
- Panel (a): time-frequency PSD heatmap in dB rel. (m/s)^2/Hz, with TOP horizontal colorbar
- Panel (b): water depth (cm) vs time since flood started (hours)
- Shared x-axis: "Time since flood started (hours)"

Key technical fix:
- pcolormesh needs TIME EDGES (bin boundaries), while time series uses CENTER times.
  This script constructs proper edges to avoid time-axis mismatch.

Outputs:
1) Multi-hour / multi-day Luong-style minute-median PSD time series (band proxy)
   - "Luong-minute" points (Welch-window center times, NOT strict 60 s)
   - Strict 1-min resampled series (for hydrology/engineering alignment)
2) Paper-grade figures:
   - Raw counts waveform (paper window)
   - Detrended counts waveform (paper window)
   - Velocity waveform (paper window)
   - Paper 2-panel figure: PSD heatmap (dB) + water depth curve (if provided)
3) Optional: heatmap for the full multi-hour span (minute × frequency)

Notes:
- Your data: fs=100 Hz => Nyquist = 50 Hz (so FREQ_MAX must <= 50)
- Luong-style window: nperseg=2^14=163.84 s, 50% overlap => step=81.92 s
  => ~42 windows per hour => floor("min") will give ~42 minute points/hour (NOT 60). Expected.
"""

from obspy import read, UTCDateTime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from scipy.signal import welch


# =============================================================================
# 0) USER CONFIG (edit here only)
# =============================================================================

# ---- Input file (single SAC trace, BHZ) ----
DATA_FILE = r"E:\\雅江OBS数据\\65BC1C12.158.BHZ"

# ---- Main processing: multi-hour span (UTC) ----
T_GLOBAL_START = UTCDateTime("2025-07-05T03:00:00")  # UTC
N_HOURS = 24 * 1  # e.g., 10 days = 24*10

# ---- Target frequency band (bedload proxy) ----
FREQ_MIN = 5.0
FREQ_MAX = 45.0  # must <= fs/2 = 50 for fs=100

# ---- Luong Welch parameters (fixed by method) ----
NPERSEG_LUONG = 2**14
NOVERLAP_LUONG = NPERSEG_LUONG // 2
STEP = NPERSEG_LUONG - NOVERLAP_LUONG

# ---- Instrument response removal settings ----
ADC_SENS = 1.6777e6  # counts/V
PAZ = {
    "poles": [-30.61, -15.8, -0.1693, -0.09732, -0.03333],
    "zeros": [0j, 0j, 0j, -30.21, -16.15],
    "gain": 1021.9,                    # V/(m/s)
    "sensitivity": 1021.9 * ADC_SENS   # counts/(m/s)
}

# ---- Paper-grade waveform visualization window (e.g., 10–60 min) ----
T_FIG_START = UTCDateTime("2025-07-05T03:10:00")
T_FIG_END   = T_FIG_START + 3600  # 60 minutes

# ---- Flood start time (t0) for the paper 2-panel figure x-axis ----
FLOOD_START_UTC = UTCDateTime("2025-07-05T03:00:00")  # <<< 改成洪水开始时刻（UTC）

# ---- Water depth data (optional) ----
# If you don't have it yet, set WATER_DEPTH_CSV = None
WATER_DEPTH_CSV = None  # e.g. r"E:\\水位数据\\water_depth_1min.csv"
WATER_DEPTH_TIME_COL = "time_utc"   # datetime column name
WATER_DEPTH_VALUE_COL = "water_depth_cm"  # depth column name (cm)

# ---- Save figures ----
SAVE_FIG = False
FIG_DPI = 300

# ---- CSV outputs ----
OUT_CSV_LUONG_MINUTE = "PSD_LuongMinute_Band.csv"
OUT_CSV_STRICT_1MIN  = "PSD_1min_Band.csv"

# ---- Paper heatmap dB limits (match your example style) ----
DB_VMIN = None
DB_VMAX = None
DB_PREF = 1.0  # reference for "dB rel. (m/s)^2/Hz": 10*log10(PSD/Pref)


# =============================================================================
# 1) Small helper utilities (procedural style)
# =============================================================================

def safe_detrend_trace(tr):
    """Detrend + demean in-place on a Trace copy, return the copy."""
    tr2 = tr.copy()
    tr2.detrend("linear")
    tr2.detrend("demean")
    return tr2


def remove_response_to_velocity(tr, paz):
    """
    Remove instrument response to obtain ground velocity (m/s).
    Uses pre_filt based on fs.
    """
    tr2 = tr.copy()
    fs = tr2.stats.sampling_rate
    fN = fs / 2.0
    pre_filt = (0.2, 0.5, 0.7 * fN, 0.9 * fN)

    tr2.simulate(
        paz_remove=paz,
        remove_sensitivity=True,
        pre_filt=pre_filt
    )
    return tr2, pre_filt


def welch_psd(x, fs, nperseg, noverlap):
    """Welch PSD, density scaling => (unit^2/Hz)."""
    f, Pxx = welch(
        x, fs=fs, window="hann",
        nperseg=nperseg, noverlap=noverlap,
        detrend=False, scaling="density"
    )
    return f, Pxx


def band_metrics(f, Pxx, fmin, fmax):
    """Band mean PSD and band-integrated power."""
    idx = (f >= fmin) & (f <= fmax)
    if not np.any(idx):
        return np.nan, np.nan
    return float(np.mean(Pxx[idx])), float(np.trapezoid(Pxx[idx], f[idx]))


def datetime64_from_utcdatetime(utc):
    """UTCDateTime -> numpy datetime64[ns]"""
    return np.datetime64(utc.datetime)


def read_water_depth_csv(path, time_col, value_col):
    """
    Read water depth series from CSV.
    Expected: a datetime column + a numeric depth column (cm).
    Returns: (DatetimeIndex, ndarray depth_cm)
    """
    df = pd.read_csv(path)
    if time_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"[ERROR] Water-depth CSV must have columns: {time_col}, {value_col}")
    t = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    y = pd.to_numeric(df[value_col], errors="coerce")
    m = t.notna() & y.notna()
    t = t[m].dt.tz_convert(None)  # drop tz to keep plotting simple (UTC naive)
    y = y[m].to_numpy(dtype=float)
    return pd.DatetimeIndex(t), y


def plot_paper_waveforms(tr_raw_counts, tr_detr_counts, tr_vel, fs, pre_filt):
    """
    Paper-grade waveform figures:
    - raw counts
    - detrended counts
    - velocity (m/s)
    """
    t = np.arange(tr_raw_counts.stats.npts) / fs

    # --- Raw counts ---
    fig0, ax0 = plt.subplots(figsize=(12, 3))
    ax0.plot(t, tr_raw_counts.data, linewidth=0.8)
    ax0.set_title("Raw waveform (counts)")
    ax0.set_xlabel("Time since window start (s)")
    ax0.set_ylabel("Counts")
    ax0.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    if SAVE_FIG:
        fig0.savefig("Fig_window_raw_counts.png", dpi=FIG_DPI, bbox_inches="tight")

    # --- Detrended counts ---
    fig1, ax1 = plt.subplots(figsize=(12, 3))
    ax1.plot(t, tr_detr_counts.data, linewidth=0.8)
    ax1.set_title("Detrended + demeaned waveform (counts)")
    ax1.set_xlabel("Time since window start (s)")
    ax1.set_ylabel("Counts")
    ax1.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    if SAVE_FIG:
        fig1.savefig("Fig_window_detrended_counts.png", dpi=FIG_DPI, bbox_inches="tight")

    # --- Velocity ---
    fig2, ax2 = plt.subplots(figsize=(12, 3))
    ax2.plot(t, tr_vel.data, linewidth=0.8)
    ax2.set_title(f"Velocity after response removal (pre_filt={pre_filt})")
    ax2.set_xlabel("Time since window start (s)")
    ax2.set_ylabel("Velocity (m/s)")
    ax2.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    if SAVE_FIG:
        fig2.savefig("Fig_window_velocity.png", dpi=FIG_DPI, bbox_inches="tight")

    plt.show()


def make_time_edges_from_minute_centers(minute_index):
    """
    Build pcolormesh time edges from minute-center timestamps.

    Why?
    - Luong-minute series timestamps are centers defined by Welch-window centers floored to minute.
    - For pcolormesh, we need bin edges (boundaries). We approximate edges by
      taking each center floored to minute, then appending last+1min.

    Returns:
    - t_edges (DatetimeIndex) length = len(minute_index)+1
    """
    t_cent = pd.DatetimeIndex(minute_index)
    # Use minute grid as edges (00:01, 00:02, ...). This keeps alignment consistent.
    t_edges = t_cent.floor("min")
    t_edges = t_edges.append(pd.Index([t_edges[-1] + pd.Timedelta(minutes=1)]))
    return pd.DatetimeIndex(t_edges)


def make_freq_edges(f):
    """
    Build frequency edges from frequency centers for pcolormesh.
    f: 1D array of frequency centers (Hz)
    returns f_edges length=len(f)+1
    """
    f = np.asarray(f, dtype=float)
    if f.size < 2:
        raise ValueError("[ERROR] frequency array too short for edges.")
    df = np.diff(f)
    # midpoint edges, endpoints extrapolated
    f_edges = np.r_[f[0] - df[0]/2, 0.5*(f[:-1] + f[1:]), f[-1] + df[-1]/2]
    return f_edges


def plot_paper_psd_heatmap_and_water_depth(
    psd_minute_df, f_l, flood_start_utc,
    water_depth_time=None, water_depth_cm=None,
    fmax_plot=None, db_vmin=-160, db_vmax=-110, pref=1.0,
    fig_name="Fig_paper_PSD_heatmap_plus_waterdepth.png"
):
    """
    Produce the paper-style 2-panel figure:
    (a) PSD heatmap (dB rel.) with TOP horizontal colorbar
    (b) Water depth curve
    Shared x-axis: hours since flood_start

    psd_minute_df: DataFrame (index: minute timestamps; columns: frequency bins PSD)
    f_l: frequency centers (Hz)
    """
    # ---- 1) Prepare time axis: centers + edges ----
    t_cent = pd.DatetimeIndex(psd_minute_df.index)
    t_edges = make_time_edges_from_minute_centers(t_cent)

    t0 = pd.to_datetime(flood_start_utc.datetime)
    # hours since flood start
    t_hours_edges = (t_edges - t0).total_seconds() / 3600.0

    # ---- 2) Frequency edges ----
    f_edges = make_freq_edges(f_l)

    # ---- 3) PSD to dB ----
    P = psd_minute_df.values.T  # (n_freq, n_time)
    P_db = 10.0 * np.log10(np.maximum(P / pref, 1e-300))

    # ---- 4) Plot layout ----
    fig = plt.figure(figsize=(8.4, 6.2))
    gs = GridSpec(2, 1, height_ratios=[1.05, 0.85], hspace=0.15)

    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.pcolormesh(
        t_hours_edges, f_edges, P_db,
        shading="auto",
        norm=Normalize(vmin=db_vmin, vmax=db_vmax),
        cmap="viridis"
    )

    # label "a)"
    ax0.text(0.02, 0.92, "a)", transform=ax0.transAxes, fontsize=16, fontweight="bold")
    ax0.set_ylabel("Frequency (Hz)")
    ax0.set_xlim(t_hours_edges[0], t_hours_edges[-1])

    # y-limit: default to Nyquist or user
    if fmax_plot is None:
        fmax_plot = float(np.max(f_l))
    ax0.set_ylim(0, fmax_plot)

    # TOP horizontal colorbar
    cbar = fig.colorbar(im, ax=ax0, orientation="horizontal", pad=0.15, fraction=0.08, aspect=40)
    cbar.set_label(r"Power [dB rel. (m/s)$^2$/Hz]")
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.xaxis.set_ticks_position('top')

    # ---- 5) Water depth panel ----
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax1.text(0.02, 0.92, "b)", transform=ax1.transAxes, fontsize=16, fontweight="bold")

    if water_depth_time is not None and water_depth_cm is not None:
        wd_t = pd.to_datetime(water_depth_time)
        wd_hours = (wd_t - t0).total_seconds() / 3600.0
        ax1.plot(wd_hours, water_depth_cm, linewidth=2.0)
        ax1.set_ylabel("Water depth (cm)")
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "No water-depth series provided", ha="center", va="center")
        ax1.set_ylabel("Water depth (cm)")

    ax1.set_xlabel("Time since flood started (hours)")

    # hide top x labels
    plt.setp(ax0.get_xticklabels(), visible=False)

    plt.tight_layout()
    if SAVE_FIG:
        fig.savefig(fig_name, dpi=FIG_DPI, bbox_inches="tight")


# =============================================================================
# 2) Read full trace
# =============================================================================

st = read(DATA_FILE)
tr_full = st[0]  # single Trace

print("\n[INFO] Full trace header:")
print(tr_full)
print("[INFO] fs =", tr_full.stats.sampling_rate, "Hz, Nyquist =", tr_full.stats.sampling_rate / 2.0, "Hz")

fs_full = float(tr_full.stats.sampling_rate)
if FREQ_MAX > fs_full / 2.0:
    raise ValueError(f"[ERROR] FREQ_MAX={FREQ_MAX} exceeds Nyquist={fs_full/2.0}. Reduce FREQ_MAX.")


# =============================================================================
# 3) Paper-grade waveform visualization (raw/detrended/velocity) for a chosen window
# =============================================================================

tr_fig_raw = tr_full.copy().trim(T_FIG_START, T_FIG_END, pad=True, fill_value=0)
tr_fig_detr = safe_detrend_trace(tr_fig_raw)
tr_fig_vel, pre_filt_fig = remove_response_to_velocity(tr_fig_detr, PAZ)

print("\n[INFO] Paper window:")
print("[INFO] window:", tr_fig_raw.stats.starttime, "to", tr_fig_raw.stats.endtime, "npts=", tr_fig_raw.stats.npts)
print("[INFO] pre_filt used:", pre_filt_fig)

plot_paper_waveforms(tr_fig_raw, tr_fig_detr, tr_fig_vel, fs_full, pre_filt_fig)


# =============================================================================
# 4) Main processing: multi-hour Luong-style minute-median PSD
# =============================================================================

all_minutes = []

# Optional: also store minute-median FULL SPECTRUM for the whole run
# (This can be big. For 10 days it may be large. Set to False if memory pressure.)
STORE_FULL_PSD_MINUTE = True
ALL_PSD_MINUTE_STACK = []  # list of DataFrame (minute × freq) for each hour

for ih in range(N_HOURS):
    t1 = T_GLOBAL_START + ih * 3600
    t2 = t1 + 3600

    # ---- Slice hour (counts) ----
    tr_hour_raw = tr_full.copy().trim(t1, t2, pad=False)

    if tr_hour_raw.stats.npts < NPERSEG_LUONG:
        print(f"[WARN] {t1} - {t2}: too short (npts={tr_hour_raw.stats.npts}), skip")
        continue

    # ---- Detrend/demean ----
    tr_hour_detr = safe_detrend_trace(tr_hour_raw)

    # ---- Remove response -> velocity ----
    tr_hour_vel, pre_filt_hour = remove_response_to_velocity(tr_hour_detr, PAZ)

    x = tr_hour_vel.data.astype(np.float64)
    x -= np.mean(x)

    if len(x) < NPERSEG_LUONG:
        print(f"[WARN] {t1} - {t2}: after processing too short, skip")
        continue

    # ---- Sliding Welch (Luong-style) ----
    nwin = (len(x) - NPERSEG_LUONG) // STEP + 1
    if nwin <= 0:
        print(f"[WARN] {t1} - {t2}: nwin<=0, skip")
        continue

    psd_list = []
    times_center = []

    t0_hour = tr_hour_vel.stats.starttime
    for i in range(nwin):
        i0 = i * STEP
        seg = x[i0:i0 + NPERSEG_LUONG]

        f_l, Pxx_l = welch_psd(seg, fs_full, NPERSEG_LUONG, 0)

        # center time of the Welch window
        tc = t0_hour + (i0 + NPERSEG_LUONG / 2) / fs_full
        times_center.append(datetime64_from_utcdatetime(tc))
        psd_list.append(Pxx_l)

    psd_arr = np.vstack(psd_list)

    # ---- Minute binning (robust) ----
    df_psd = pd.DataFrame(psd_arr)
    df_psd["minute"] = pd.to_datetime(times_center).floor("min")
    psd_minute = df_psd.groupby("minute").median()
    psd_minute.index.name = "minute_utc"

    # ---- Band metrics per minute ----
    idx = (f_l >= FREQ_MIN) & (f_l <= FREQ_MAX)
    cols = np.where(idx)[0]
    if cols.size == 0:
        raise ValueError("[ERROR] Band indices empty; check FREQ_MIN/FREQ_MAX vs fs.")

    out_minute = pd.DataFrame({
        "PSD_band_mean": psd_minute.iloc[:, cols].mean(axis=1),
        "PSD_band_int":  psd_minute.iloc[:, cols].apply(lambda r: np.trapezoid(r.values, f_l[idx]), axis=1)
    })
    out_minute.index.name = "minute_utc"

    all_minutes.append(out_minute)

    if STORE_FULL_PSD_MINUTE:
        ALL_PSD_MINUTE_STACK.append(psd_minute)

    print(f"[INFO] {t1} - {t2}: minute_points={out_minute.shape[0]} (Luong-minute)")


# ---- Merge + export ----
if len(all_minutes) == 0:
    raise RuntimeError("[ERROR] No valid hourly segments processed. Check time range or data continuity.")

out_all = pd.concat(all_minutes).sort_index()

# remove duplicates
out_all = out_all.groupby(out_all.index).median()

# strict 1-min resample
out_all_1min = out_all.resample("1min").median()

# Save
out_all.to_csv(OUT_CSV_LUONG_MINUTE)
out_all_1min.to_csv(OUT_CSV_STRICT_1MIN)

print("\n[INFO] Saved outputs:")
print(" -", OUT_CSV_LUONG_MINUTE, " (Luong-minute points, Welch-window center time)")
print(" -", OUT_CSV_STRICT_1MIN,  " (strict 1-min bins, may contain NaNs)")
print("[INFO] Luong-minute points:", out_all.shape[0])
print("[INFO] Strict 1-min points:", out_all_1min.shape[0])


# =============================================================================
# 5) Plot: multi-day strict 1-min band proxy time series
# =============================================================================

fig_ts, ax_ts = plt.subplots(figsize=(12, 3))
ax_ts.plot(out_all_1min.index, out_all_1min["PSD_band_mean"], linewidth=0.8)
ax_ts.set_yscale("log")
ax_ts.set_title(f"Strict 1-min PSD mean ({FREQ_MIN}-{FREQ_MAX} Hz)")
ax_ts.set_xlabel("Time (UTC)")
ax_ts.set_ylabel(r"Median PSD mean ((m/s)$^2$/Hz)")
ax_ts.grid(True, which="both", alpha=0.25)
fig_ts.autofmt_xdate()
plt.tight_layout()
if SAVE_FIG:
    fig_ts.savefig("Fig_multiday_strict_1min_PSD_series.png", dpi=FIG_DPI, bbox_inches="tight")


# =============================================================================
# 6) Paper figure: PSD heatmap (dB) + water depth (like your screenshot)
# =============================================================================

# --- Build a PSD minute-median heatmap dataset for the whole multi-hour span ---
# Warning: may be big for many days. You can disable STORE_FULL_PSD_MINUTE to skip.
if STORE_FULL_PSD_MINUTE and len(ALL_PSD_MINUTE_STACK) > 0:
    psd_minute_all = pd.concat(ALL_PSD_MINUTE_STACK).sort_index()
    # if duplicates exist, keep median
    psd_minute_all = psd_minute_all.groupby(psd_minute_all.index).median()
else:
    # fallback: use last computed hour only (for quick check)
    psd_minute_all = psd_minute

# Optional: limit frequency axis for plotting (fs=100Hz => <=50Hz)
FMAX_PLOT = min(50.0, float(np.max(f_l)))

# Read water depth if provided
wd_time, wd_cm = None, None
if WATER_DEPTH_CSV is not None:
    try:
        wd_time, wd_cm = read_water_depth_csv(WATER_DEPTH_CSV, WATER_DEPTH_TIME_COL, WATER_DEPTH_VALUE_COL)
        print(f"[INFO] Loaded water depth: n={len(wd_cm)}")
    except Exception as e:
        print("[WARN] Failed to read water depth CSV:", e)
        wd_time, wd_cm = None, None

# Make paper plot
plot_paper_psd_heatmap_and_water_depth(
    psd_minute_df=psd_minute_all,
    f_l=f_l,
    flood_start_utc=FLOOD_START_UTC,
    water_depth_time=wd_time,
    water_depth_cm=wd_cm,
    fmax_plot=FMAX_PLOT,
    db_vmin=DB_VMIN,
    db_vmax=DB_VMAX,
    pref=DB_PREF,
    fig_name="Fig_paper_PSDheatmap_plus_waterdepth.png"
)


# =============================================================================
# 7) Optional: quick PSD heatmap in linear units (sanity check)
# =============================================================================

fig_hm, ax_hm = plt.subplots(figsize=(12, 4))
im = ax_hm.pcolormesh(
    psd_minute_all.index,
    f_l,
    psd_minute_all.values.T,
    shading="auto"
)
ax_hm.set_ylim(0, FMAX_PLOT)
ax_hm.set_ylabel("Frequency (Hz)")
ax_hm.set_xlabel("Time (UTC)")
ax_hm.set_title("Minute-median PSD (Luong-style, linear units)")
cbar = fig_hm.colorbar(im, ax=ax_hm)
cbar.set_label(r"Median PSD ((m/s)$^2$/Hz)")
fig_hm.autofmt_xdate()
plt.tight_layout()

plt.show()
