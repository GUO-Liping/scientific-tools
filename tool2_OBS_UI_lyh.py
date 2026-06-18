import os
import numpy as np
import pandas as pd
import warnings

from seismic_bedload import SaltationModel
from seismic_bedload import log_raised_cosine_pdf

print("1. 正在加载 5200139 台站已对齐的数据表...")
# 🟢 注意：路径已经更新为 139 台站的文件
INPUT_EXCEL = r"D:\02 丽江地震数据\01北京时间\去仪器响应\PSD数据_NPZ\5200139_30_80Hz_5min_LinearEnergy_with_Depth.xlsx"
OUTPUT_EXCEL = r"D:\02 丽江地震数据\01北京时间\去仪器响应\PSD数据_NPZ\5200139_30_80Hz_5min_LinearEnergy_with_Flux_0829_0904.xlsx"

df_full = pd.read_excel(INPUT_EXCEL)
df_full = df_full.dropna(subset=['Linear_Energy_30_80Hz', 'Water_Depth_m'])
df_full['Time'] = pd.to_datetime(df_full['Time'])

# ==========================================
# 2. 时间切片：0829 - 0904
# ==========================================
start_date = '2025-08-29 00:00:00'
end_date   = '2025-09-04 23:59:59'

print(f"正在截取 {start_date[:10]} 至 {end_date[:10]} 的数据...")
mask = (df_full['Time'] >= start_date) & (df_full['Time'] <= end_date)
df = df_full.loc[mask].copy()

if df.empty:
    print("❌ 错误：在此时间段内没有找到数据！")
    exit()

N = len(df)
print(f"✅ 截取成功！共有 {N} 个时间点需要计算。")

observe_linear = df['Linear_Energy_30_80Hz'].values
H = df['Water_Depth_m'].values

# ==========================================
# 3. 初始化物理参数与正演模型
# ==========================================
D = np.linspace(0.002, 0.07, 100)
sigma_g = 0.85
mu = 0.009
s = sigma_g / np.sqrt(1 / 3 - 2 / np.pi ** 2)
pD = log_raised_cosine_pdf(D, mu, s) / D

model = SaltationModel()
freqs = np.linspace(30, 80, 10)
ave_freqs = np.zeros(N)

# ⚠️ 场地参数：请确认 139 台站距离河道的真实距离！
W = 50       # 河宽 (m)
theta = np.tan(0.7 * np.pi / 180)  # 坡度 (-)
r0 = 20      # 🟢 震源到 5200139 台站的距离 (m)，若与 137 台站不同请修改此处！
qb_unit = 1  # 单位体积通量 m^2/s

print(f"3. 正在计算地震反演模型 (总共 {N} 步)...")

for j in range(N):
    PSD_estimated_tmp = np.zeros(len(freqs))

    if H[j] < 0.05:
        ave_freqs[j] = np.nan
        continue

    for k in range(len(freqs)):
        res = np.zeros(len(D))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(len(D)):
                PSD = model.forward_psd(freqs[k], D[i], H[j], W, theta, r0, qb_unit)
                res[i] = PSD

        res_clean = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
        PSD_estimated_tmp[k] = np.trapezoid(res_clean * pD, x=D)

    ave_freqs[j] = np.median(PSD_estimated_tmp)

    if (j + 1) % 500 == 0:
        print(f"   已完成 {j + 1} / {N} 个时间点...")

# ==========================================
# 4. 反演计算与 MPM 水力学理论计算
# ==========================================
print("4. 正在计算推移质通量与水力学理论值...")

# A. 地震反演推移质通量
qbd = observe_linear / ave_freqs
bedload_flux_seismic = qbd * 2700

df['地震理论单位能量'] = ave_freqs
df['反演推移质通量_kg_ms'] = bedload_flux_seismic

# B. MPM 水力学理论通量
rho_w, rho_s, g, D50, tau_c_star = 1000, 2700, 9.8, 0.009, 0.045
tau_b = rho_w * g * df['Water_Depth_m'] * theta
tau_star = tau_b / ((rho_s - rho_w) * g * D50)

mpm_flux = np.where(
    tau_star > tau_c_star,
    8 * (tau_star - tau_c_star)**1.5 * np.sqrt((rho_s/rho_w - 1) * g * D50**3) * rho_s,
    0
)

df['水力理论推移质通量_MPM_kg_ms'] = mpm_flux

# ==========================================
# 5. 保存结果
# ==========================================
df.to_excel(OUTPUT_EXCEL, index=False)
print("-" * 50)
print(f"🎉 计算完成！台站 5200139 的最终数据保存在: \n{OUTPUT_EXCEL}")