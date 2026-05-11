from obspy import read, UTCDateTime, Stream
import time
import glob

# =============================================================================
# 2) 地震信号处理函数（去趋势/去响应/PSD）
# =============================================================================
def detrend_demean_stream(st_in: Stream) -> Stream:
    """合并间断（尽量插值填补）→ 线性去趋势 → 去均值。"""
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
    """
    st = st_in.copy()
    fs = st[0].stats.sampling_rate
    fN = fs / 2.0
    pre_filt = (0.2, 0.5, 0.7 * fN, 0.9 * fN)
    for tr in st:
        tr.simulate(paz_remove=paz, remove_sensitivity=True, pre_filt=pre_filt)
    return st, pre_filt


def remove_response_to_velocity(st_in: Stream, paz: dict):
    """
    去仪器响应：counts → m/s
    pre_filt 用于稳定去响应（高低频端做缓冲滤波）。
    """
    st = st_in.copy()
    fs = st[0].stats.sampling_rate
    fN = fs / 2.0
    pre_filt = (0.2, 0.5, 0.7 * fN, 0.9 * fN)
    for tr in st:
        tr.simulate(paz_remove=paz, remove_sensitivity=True, pre_filt=pre_filt)
    return st, pre_filt
# 从服务器读取波形数据
DATA_FILES = sorted(glob.glob("*.MSD"))   # 读入多个MSD/BHZ文件（指定路径）

# ---- 仪器参数 ----
ADC_SENS = 1.6777e6
PAZ = {
    "poles": [-30.61, -15.8, -0.1693, -0.09732, -0.03333],
    "zeros": [0j, 0j, 0j, -30.21, -16.15],
    "gain": 1021.9,
    "sensitivity": 1021.9 * ADC_SENS,
}

st = Stream()
time_plot_start = time.time()

for f in DATA_FILES:
    try:
        print("loading .MSD File:", f)
        st_tmp = read(f)
        tr_tmp = st_tmp[0]
        print(tr_tmp)
        st += st_tmp
    except Exception as e:
        print(f"[SKIP] {f} -> {e}")
        continue

st.merge(method=1, fill_value="interpolate")
tr_full = st[0]

T_GLOBAL_START = tr_full.stats.starttime
T_GLOBAL_END = tr_full.stats.endtime
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