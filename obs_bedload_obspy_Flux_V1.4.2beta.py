# -*- coding: utf-8 -*-
"""
OBS 推移质地震反演全流程集成脚本 (v1.3 - 极速严谨版)
理论来源:
  [1] Tsai et al., 2012 (基岩跳跃模型)
  [2] Luong et al., 2024 (冲积河道多模式模型)
  [3] Luong et al., 2026 (地震-水力半经验耦合模型)

更新说明 (v1.3):
  1. 性能革命：全面使用 NumPy 矩阵运算重构了三大模型的核心物理积分，彻底消除了计算密集型的嵌套 for 循环。
  2. 预计算优化：将与时间无关的地震波传播因子 (vc, vu, chi) 提取至全局循环外，避免了数十万次的冗余计算。
  3. 保持严谨：100% 保留了 v1.2 的精确理论积分与对数正态/升余弦概率分布计算，且包含可视化与频带输出。
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import welch
from scipy.special import gamma
from obspy import read, UTCDateTime, Stream

# =============================================================================
# 0) 全局参数与环境配置
# =============================================================================
OUT_DIR = "output_files_v1.3"
os.makedirs(OUT_DIR, exist_ok=True)

RIVER_XLSX = r"MT_river_height.xlsx"
DATA_FILES = sorted(glob.glob("*.BHZ"))

BED_BOTTOM = 710.6              
BAND_FMIN, BAND_FMAX = 2.0, 20.0    
INV_FMIN, INV_FMAX = 20.0, 50.0     
HEATMAP_FMIN_SHOW, HEATMAP_FMAX_SHOW = 2.0, 50.0 

USE_AUTO_TIME_RANGE = False
N_days_process = 7  # 支持处理1个月的数据
NPERSEG_LUONG = 2**14

ADC_SENS = 1.6777e6
PAZ = {
    "poles": [-30.61, -15.8, -0.1693, -0.09732, -0.03333],
    "zeros": [0j, 0j, 0j, -30.21, -16.15],
    "gain": 1021.9,
    "sensitivity": 1021.9 * ADC_SENS,
}

# =============================================================================
# 1) 核心物理与地震学参数
# =============================================================================
RHO_S, RHO_F = 2550.0, 1000.0       
G, NU = 9.81, 1e-6            
R_SUB = (RHO_S - RHO_F) / RHO_F

V0, Z0, F0 = 2206.0, 1000.0, 1.0 
A_PARAM, Q0 = 0.272, 20.0            

W = 160 - 27.5       
THETA = np.tan(0.4 * np.pi / 180)  
R0_DIST = 500.0      

D50, D_MEAN, SIGMA_D = 0.02, 0.08, 0.45       

# 离散积分网格
D_ARRAY = np.logspace(np.log10(0.002), np.log10(0.2), 30) 
T_HOP_ARRAY = np.logspace(np.log10(0.01), np.log10(0.5), 20) 

# =============================================================================
# 2) 基础数学与波传播引擎 (支持向量化)
# =============================================================================
def log_raised_cosine_pdf(D_arr, mu, s):
    ln_D = np.log(D_arr)
    ln_mu = np.log(mu)
    res = np.where(np.abs(ln_D - ln_mu) < s, 
                   0.5 / s * (1 + np.cos(np.pi * (ln_D - ln_mu) / s)), 0.0)
    integral = np.trapezoid(res, D_arr)
    return res / integral if integral > 0 else res

def log_normal_hop_time_pdf(t_arr, mu_t=np.log(0.1), sigma_t=0.5):
    res = (1.0 / (t_arr * sigma_t * np.sqrt(2 * np.pi))) * np.exp(-((np.log(t_arr) - mu_t)**2) / (2 * sigma_t**2))
    integral = np.trapezoid(res, t_arr)
    return res / integral if integral > 0 else res

def calc_tau_star(H, theta, D):
    return (H * theta) / (R_SUB * D)

def calc_critical_tau(theta):
    return 0.15 * (theta ** 0.25)

def calc_seismic_wave_properties(f, r0):
    zeta = A_PARAM / (1 - A_PARAM)
    vc0 = (V0 * gamma(1 + A_PARAM) / (2 * np.pi * Z0 * F0)**A_PARAM)**(1 / (1 - A_PARAM))
    vc = vc0 * (f / F0)**(-zeta)
    vu = vc / (1 + zeta)
    beta = (2 * np.pi * r0 * (1 + zeta) * f**(1 + zeta) / (vc0 * Q0 * F0**zeta))
    chi = (2 * np.log(1 + 1/beta) * np.exp(-2 * beta) + 
           (1 - np.exp(-beta)) * np.exp(-beta) * np.sqrt(2 * np.pi / beta))
    return vc, vu, chi

def calc_drag_coefficient(D):
    D_star = np.log10((R_SUB * G * D**3) / NU**2)
    R1 = -3.76715 + 1.92944*D_star - 0.09815*(D_star**2) - 0.00575*(D_star**3) + 0.00056*(D_star**4)
    R2 = (np.log10(1 - (0.2 / 0.85)) - 0.2**2.3 * np.tanh(D_star - 4.6) + 
          0.3 * (-0.3) * 0.2**2 * (D_star - 4.6))
    R3 = (0.65 - ((0.8 / 2.83) * np.tanh(D_star - 4.6)))**(1 + (0 / 2.5))
    W_star = R3 * 10**(R1 + R2)
    w_s = (W_star * R_SUB * G * NU)**(1/3)
    return (4/3) * (R_SUB * G * D) / (w_s**2)

# =============================================================================
# 3) 三大核心模型 (彻底矩阵化重构)
# =============================================================================
# 预计算积分网格的概率分布
s_param = SIGMA_D / np.sqrt(1/3 - 2/np.pi**2)
PDF_D = log_raised_cosine_pdf(D_ARRAY, D_MEAN, s_param)
PDF_T = log_normal_hop_time_pdf(T_HOP_ARRAY)
VP_ARRAY = np.pi * D_ARRAY**3 / 6
M_ARRAY = RHO_S * VP_ARRAY
CD_ARRAY = calc_drag_coefficient(D_ARRAY)

def model_tsai_2012_vectorized(PSD_obs_linear, term_f, H):
    """【极速模型一】Tsai 2012 纯跳跃"""
    tau_c_50 = calc_critical_tau(THETA)
    tau_c = tau_c_50 * (D_ARRAY / D50)**(-0.9)
    tau_star = calc_tau_star(H, THETA, D_ARRAY)
    
    valid_mask = tau_star > tau_c
    if not np.any(valid_mask): return 0.0
    
    stage = np.zeros_like(D_ARRAY)
    stage[valid_mask] = tau_star[valid_mask] / tau_c[valid_mask]
    
    Ub = np.zeros_like(D_ARRAY)
    Hb = np.zeros_like(D_ARRAY)
    Ub[valid_mask] = 1.56 * np.sqrt(R_SUB * G * D_ARRAY[valid_mask]) * (stage[valid_mask])**0.56
    Hb[valid_mask] = 1.44 * D_ARRAY[valid_mask] * (stage[valid_mask])**0.50
    
    wst = np.sqrt(4 * R_SUB * G * D_ARRAY / (3 * CD_ARRAY))
    
    Hb_c = np.zeros_like(D_ARRAY)
    Hb_c[valid_mask] = (3 * CD_ARRAY[valid_mask] * RHO_F * Hb[valid_mask]) / (2 * RHO_S * D_ARRAY[valid_mask] * np.cos(np.arctan(THETA)))
    
    wi = np.zeros_like(D_ARRAY)
    ws = np.zeros_like(D_ARRAY)
    valid_Hb_c = Hb_c > 0
    wi[valid_Hb_c] = wst[valid_Hb_c] * np.cos(np.arctan(THETA)) * np.sqrt(1 - np.exp(-Hb_c[valid_Hb_c]))
    ws[valid_Hb_c] = (Hb_c[valid_Hb_c] * wst[valid_Hb_c] * np.cos(np.arctan(THETA))) / (2 * np.log(np.exp(Hb_c[valid_Hb_c]/2) + np.sqrt(np.exp(Hb_c[valid_Hb_c]) - 1)))
    
    rate = np.zeros_like(D_ARRAY)
    denom = VP_ARRAY * Ub * Hb
    valid_denom = denom > 0
    rate[valid_denom] = (2/3) * W * 1.0 * ws[valid_denom] / denom[valid_denom]
    
    # 构建粒径矩阵 D
    term_D = rate * (np.pi**2 * M_ARRAY**2 * wi**2) / RHO_S**2
    # 交叉相乘生成 [频率, 粒径] 二维矩阵，并积分
    psd_matrix = term_f[:, np.newaxis] * term_D[np.newaxis, :]
    PSD_star_f = np.trapezoid(psd_matrix * PDF_D[np.newaxis, :], D_ARRAY, axis=1)
    
    median_psd_star = np.median(PSD_star_f)
    return 0.0 if median_psd_star == 0 else PSD_obs_linear / median_psd_star

def model_luong_2024_vectorized(PSD_obs_linear, term_f, H):
    """【极速模型二】Luong 2024 多模式模型 (利用矩阵外积避免循环)"""
    tau_c_50 = calc_critical_tau(THETA)
    tau_c = tau_c_50 * (D_ARRAY / D50)**(-0.9)
    tau_star = calc_tau_star(H, THETA, D_ARRAY)
    
    valid_mask = tau_star > tau_c
    if not np.any(valid_mask): return 0.0
    
    stage = np.zeros_like(D_ARRAY)
    stage[valid_mask] = tau_star[valid_mask] / tau_c[valid_mask]
    
    ks = 3 * D_ARRAY
    u_shear = np.sqrt(G * H * THETA)
    U_max = 8.1 * u_shear * (H / ks)**1.6
    
    Ub = np.zeros_like(D_ARRAY)
    if H < 0.4:
        Ub[valid_mask] = 30.5 * tau_star[valid_mask] * np.sqrt(R_SUB * G * D_ARRAY[valid_mask]) * (D_ARRAY[valid_mask]/ks[valid_mask])**0.583
    else:
        stage_gt_1 = valid_mask & (stage > 1)
        Ub[stage_gt_1] = 1.56 * np.sqrt(R_SUB * G * D_ARRAY[stage_gt_1]) * (stage[stage_gt_1] - 1)**0.56
        
    Ub = np.minimum(np.maximum(Ub, 0.0), U_max)
    if not np.any(Ub > 0): return 0.0
    
    I_total = (np.abs((0.539 * (1 + 0.5) * Ub) * 0.352))**2 
    
    # 构造 [粒径, 时间] 矩阵
    s_len = Ub[:, np.newaxis] * T_HOP_ARRAY[np.newaxis, :]
    rate = np.zeros_like(s_len)
    s_mask = s_len > 0
    rate[s_mask] = W * 1.0 / (VP_ARRAY[:, np.newaxis] * s_len)[s_mask]
    
    term_DT = rate * (M_ARRAY[:, np.newaxis]**2 * np.pi**2 * I_total[:, np.newaxis]) / (RHO_S**2 * 4)
    # 沿时间轴积分，降维回一维粒径数组
    psd_D = np.trapezoid(term_DT * PDF_T[np.newaxis, :], T_HOP_ARRAY, axis=1)
    
    # 再与频率矩阵交叉乘积，沿粒径轴积分
    psd_matrix = term_f[:, np.newaxis] * psd_D[np.newaxis, :]
    PSD_star_f = np.trapezoid(psd_matrix * PDF_D[np.newaxis, :], D_ARRAY, axis=1)

    median_psd_star = np.median(PSD_star_f)
    return 0.0 if median_psd_star == 0 else PSD_obs_linear / median_psd_star

def model_luong_2026(PSD_obs_dB, H):
    k, m_exp, n_exp = 5.2, 0.25, 1.36
    A_pre, B_pre = 29.0, -127.0
    
    tau_c = calc_critical_tau(THETA)
    tau_star = calc_tau_star(H, THETA, D50)
    if tau_star <= tau_c: return 0.0
    
    PSD_re_dB = A_pre * np.log10(tau_star) + B_pre
    ratio_P_Pre = 10 ** ((PSD_obs_dB - PSD_re_dB) / 10.0)
    qb_star = k * (ratio_P_Pre ** m_exp) * ((tau_star - tau_c) ** n_exp)
    
    conversion = np.sqrt(R_SUB * G * D50**3) * RHO_S
    return qb_star * conversion

# =============================================================================
# 4) 数据读取与处理工具
# =============================================================================
def read_river_from_excel(xlsx, time_col="time", height_col="height", bed_elev=0.0):
    df = pd.read_excel(xlsx, sheet_name=0)[[time_col, height_col]]
    df.columns = ["time", "height"]
    t = pd.to_datetime(df["time"], errors="coerce")
    
    if getattr(t.dt, "tz", None) is None:
        t = t.dt.tz_localize("Asia/Shanghai").dt.tz_convert("UTC")
    else:
        t = t.dt.tz_convert("UTC")
        
    depth = pd.to_numeric(df["height"], errors="coerce").to_numpy(dtype=float) - bed_elev
    mask = t.notna() & np.isfinite(depth)
    order = np.argsort(t[mask].values)
    return pd.to_datetime(t[mask].values[order], utc=True), depth[mask][order]

def remove_response_to_velocity(st_in: Stream, paz: dict):
    st = st_in.copy()
    fN = st[0].stats.sampling_rate / 2.0
    pre_filt = (0.2, 0.5, 0.7 * fN, 0.9 * fN)
    for tr in st:
        tr.simulate(paz_remove=paz, remove_sensitivity=True, pre_filt=pre_filt)
    return st

def strict_1min_psd_from_trace(tr_vel, nperseg):
    fs = tr_vel.stats.sampling_rate
    t0 = tr_vel.stats.starttime
    npts = tr_vel.stats.npts
    
    idx = pd.date_range(pd.Timestamp(t0.datetime, tz="UTC").floor("min"),
                        pd.Timestamp((t0 + (npts - 1) / fs).datetime, tz="UTC").ceil("min"),
                        freq="1min", tz="UTC", inclusive="left")
    
    x = tr_vel.data.astype(np.float64)
    x -= np.mean(x)
    psd_list, valid_idx = [], []
    f_axis = None
    
    for minute in idx:
        tc = UTCDateTime(minute.to_pydatetime()) + 30.0
        ic = int(round((tc - t0) * fs))
        i1, i2 = ic - nperseg//2, ic + nperseg//2
        if i1 >= 0 and i2 <= npts:
            f, Pxx = welch(x[i1:i2], fs=fs, window="hann", nperseg=nperseg, noverlap=0, detrend=False, scaling="density")
            psd_list.append(Pxx)
            valid_idx.append(minute)
            if f_axis is None: f_axis = f
            
    if not psd_list: return pd.DataFrame()
    return pd.DataFrame(psd_list, index=valid_idx, columns=f_axis)

def band_index_from_psd(psd_df, fmin, fmax):
    f = psd_df.columns.to_numpy(dtype=float)
    idx = (f >= fmin) & (f <= fmax)
    band_mean = psd_df.loc[:, idx].mean(axis=1)
    band_int = psd_df.loc[:, idx].apply(lambda r: np.trapezoid(r.values, f[idx]), axis=1)
    return pd.DataFrame({f"PSD_{fmin:g}_{fmax:g}_mean": band_mean, f"PSD_{fmin:g}_{fmax:g}_int": band_int})

# =============================================================================
# 5) 主执行流程 (Fail-Fast + 高速集成)
# =============================================================================
if __name__ == "__main__":
    print("=== OBS 推移质反演多模型集成系统 (v1.3 极速严谨版) ===")

    if not os.path.exists(RIVER_XLSX):
        raise FileNotFoundError(f"[致命错误] 找不到水位文件: {os.path.abspath(RIVER_XLSX)}")
    if len(DATA_FILES) == 0:
        raise FileNotFoundError("[致命错误] 未发现任何 .BHZ 文件！")

    print(f"-> 1/5 正在读取并处理水位数据...")
    time_river, depth_river = read_river_from_excel(RIVER_XLSX, bed_elev=BED_BOTTOM)
    river_series = pd.Series(depth_river, index=time_river).resample("1min").mean().interpolate(method="time")
    
    print("-> 2/5 正在加载地震波形数据 (可能耗时较长)...")
    st = Stream()
    for file in DATA_FILES:
        try: st += read(file)
        except Exception as e: print(f"   [跳过] {file}: {e}")
            
    st.merge(method=1, fill_value="interpolate")
    tr_full = st[0]
    
    T_START = tr_full.stats.starttime
    T_END = tr_full.stats.endtime if USE_AUTO_TIME_RANGE else T_START + N_days_process * 24 * 3600
    total_hours = int((T_END - T_START) // 3600)

    print(f"-> 3/5 计算滑动 Welch PSD (总计 {total_hours} 小时)...")
    all_psd_1min = []
    
    for ih in range(total_hours):
        if ih % 12 == 0 and ih > 0: print(f"   已处理 {ih}/{total_hours} 小时...")
        t1 = T_START + ih * 3600
        st_seg = st.slice(t1, t1 + 3600)
        if len(st_seg) == 0 or st_seg[0].stats.npts < NPERSEG_LUONG: continue
        
        st_seg.detrend("linear").detrend("demean")
        st_vel = remove_response_to_velocity(st_seg, PAZ)
        
        psd_df = strict_1min_psd_from_trace(st_vel[0], NPERSEG_LUONG)
        if not psd_df.empty: all_psd_1min.append(psd_df)

    psd_all_1min = pd.concat(all_psd_1min).groupby(level=0).median()
    river_on_minute = river_series.reindex(psd_all_1min.index).interpolate(method="time", limit_direction="both")

    band_df = band_index_from_psd(psd_all_1min, BAND_FMIN, BAND_FMAX)

    f_all = psd_all_1min.columns.to_numpy(dtype=float)
    inv_band = (f_all >= INV_FMIN) & (f_all <= INV_FMAX)
    f_inv = f_all[inv_band]

    P_lin = psd_all_1min.to_numpy(dtype=float)
    PSD_dB = 10.0 * np.log10(np.maximum(P_lin, 1e-30))
    PSD_obs_dB = np.nanmedian(PSD_dB[:, inv_band], axis=1)
    
    H_arr = river_on_minute.values.astype(float)
    
    mask = np.isfinite(PSD_obs_dB) & np.isfinite(H_arr)
    time_valid = psd_all_1min.index[mask]
    PSD_obs_dB_use = PSD_obs_dB[mask]
    H_use = H_arr[mask]

    print(f"-> 4/5 启动高精度矩阵化反演引擎，有效样本: {len(H_use)} ...")
    qb_tsai_arr = np.zeros_like(H_use)
    qb_luong24_arr = np.zeros_like(H_use)
    qb_luong26_arr = np.zeros_like(H_use)

    # 预计算不变的频率波属性项 (提速核心秘诀)
    vc_inv, vu_inv, chi_inv = calc_seismic_wave_properties(f_inv, R0_DIST)
    TERM_F = (f_inv**3) / (vc_inv**3 * vu_inv**2) * chi_inv

    print('len(H_use)=',len(H_use))
    for i in range(len(H_use)):
        print('i=',i)
        h = H_use[i]
        p_db = PSD_obs_dB_use[i]
        p_lin = 10**(p_db / 10.0)
        
        # 调用的已经是完全矩阵化后的函数，内部不再有任何 for 循环
        qb_tsai_arr[i] = model_tsai_2012_vectorized(p_lin, TERM_F, h)
        qb_luong24_arr[i] = model_luong_2024_vectorized(p_lin, TERM_F, h)
        qb_luong26_arr[i] = model_luong_2026(p_db, h)

    qb_df = pd.DataFrame(index=time_valid)
    qb_df["Qb_Tsai2012 (kg/s)"] = qb_tsai_arr * RHO_S * W
    qb_df["Qb_Luong2024 (kg/s)"] = qb_luong24_arr * RHO_S * W
    qb_df["Qb_Luong2026 (kg/s)"] = qb_luong26_arr * W

    # =============================================================================
    # 6) 可视化绘图与输出
    # =============================================================================
    print("-> 5/5 正在生成可视化热力图与保存数据...")
    
    psd_plot = psd_all_1min.interpolate(limit=10, limit_direction="both")
    f_show_mask = (f_all >= HEATMAP_FMIN_SHOW) & (f_all <= HEATMAP_FMAX_SHOW)
    P_hm = 10.0 * np.log10(np.maximum(psd_plot.values.T[f_show_mask, :], 1e-30))
    f_show = f_all[f_show_mask]
    
    fig_hm, ax_hm = plt.subplots(figsize=(14, 5))
    times = psd_plot.index.to_pydatetime()
    extent = [mdates.date2num(times[0]), mdates.date2num(times[-1]), f_show[0], f_show[-1]]
    
    im = ax_hm.imshow(P_hm, aspect="auto", origin="lower", extent=extent, 
                      vmin=np.nanpercentile(P_hm, 5), vmax=np.nanpercentile(P_hm, 95), cmap="viridis")
    ax_hm.set_ylabel("Frequency (Hz)")
    ax_hm.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    fig_hm.colorbar(im, ax=ax_hm, pad=0.05).set_label("PSD (dB)")
    
    ax_riv = ax_hm.twinx()
    ax_riv.plot(psd_all_1min.index, river_on_minute.values, color="red", lw=2, label="River Depth")
    ax_riv.set_ylabel("Depth (m)", color="red")
    ax_riv.tick_params(axis="y", labelcolor="red")
    
    plt.title("PSD Heatmap & Flow Depth")
    fig_hm.autofmt_xdate()
    plt.tight_layout()
    fig_hm.savefig(os.path.join(OUT_DIR, "PSD_Heatmap_v1.3.png"), dpi=300)
    plt.close()

    fig_qb, ax_qb = plt.subplots(figsize=(14, 5))
    ax_qb.plot(qb_df.index, qb_df["Qb_Tsai2012 (kg/s)"].rolling(10).mean(), label="Tsai 2012", alpha=0.7)
    ax_qb.plot(qb_df.index, qb_df["Qb_Luong2024 (kg/s)"].rolling(10).mean(), label="Luong 2024", alpha=0.7)
    ax_qb.plot(qb_df.index, qb_df["Qb_Luong2026 (kg/s)"].rolling(10).mean(), label="Luong 2026", color="black", lw=1.5)
    
    ax_qb.set_ylabel("Bedload Flux (kg/s)")
    ax_qb.legend()
    ax_qb.grid(alpha=0.3)
    ax_qb.set_title("Bedload Flux Inversion (10-min smoothed)")
    fig_qb.autofmt_xdate()
    plt.tight_layout()
    fig_qb.savefig(os.path.join(OUT_DIR, "Bedload_Flux_v1.3.png"), dpi=300)
    plt.close()

    with pd.ExcelWriter(os.path.join(OUT_DIR, "OBS_Results_MultiModel_v1.3.xlsx")) as writer:
        river_on_minute.to_frame("River_Depth").tz_localize(None).to_excel(writer, sheet_name="Depth")
        band_df.tz_localize(None).to_excel(writer, sheet_name="PSD_Band_Indices")
        qb_df.tz_localize(None).to_excel(writer, sheet_name="Inversion_Results")

    print(f"\n[任务完毕] 运算全程完毕，结果已保存至: {os.path.abspath(OUT_DIR)}")