# 将每一行都添加详细注释的完整 Python 代码保存为 .py 文件

# Tsai et al. (2012) 地震噪声模型 —— 每行含注释的 Python 实现
# ===============================================================

import numpy as np                      # 导入数值计算库 numpy
import matplotlib.pyplot as plt        # 导入绘图库 matplotlib
from scipy.special import gamma        # 从 scipy.special 导入 gamma 函数（用于剪切波速表达式）
from scipy.integrate import quad       # 从 scipy.integrate 导入积分函数 quad（用于对粒径积分）

# ------------------------
# 基本物理常数（来自文中参数设定）
# ------------------------
rho_s = 2700                           # 颗粒密度，单位 kg/m³
rho_f = 1000                           # 水密度，单位 kg/m³
g = 9.81                               # 重力加速度，单位 m/s²
C1 = 2 / 3                             # 冲击频率修正因子（公式 13）
Q0 = 20                                # 品质因子 Q（用于 Rayleigh 波衰减，公式 3）
f0 = 1                                 # 参考频率，单位 Hz
v0 = 2206                              # 剪切波速度参考值，公式 (4)
z0 = 1000                              # 剪切波速度参考深度，单位 m，公式 (4)
a = 0.272                              # 剪切波速度深度依赖幂次，公式 (4)
x = a / (1 - a)                        # 频率依赖幂次，用于公式 (5)
vc0 = ((2 * np.pi * z0 * f0)**-a * v0 * gamma(1 + a))**(1 / (1 - a))  # Rayleigh 波相速度归一化系数
vu_ratio = 1 / (1 + x)                 # 群速度与相速度之比，公式 (5)

# ------------------------
# 拖曳系数与沉速计算（公式 16）
# ------------------------
def drag_coefficient(D):
    Re = w_s*D/v    # 使用粒径计算 Reynolds 数（估算）
    C_d = (rho_s - rho)*g*V / (rho*w_s**2*A)
    if C_d < 0.5 or C_d > 1.4:
        raise ValueError
    return C_d  # 返回简化的 Dietrich 拖曳系数

def terminal_velocity(D):
    Cd = drag_coefficient(D)           # 获取拖曳系数
    R = (rho_s - rho_f) / rho_f        # 颗粒与水的相对密度
    return np.sqrt(4 * R * g * D / (3 * Cd))  # 计算终端沉速 w_st，公式 (16)

# ------------------------
# Rayleigh 波传播模型（公式 3、5、8、9）
# ------------------------
def rayleigh_velocity(f):
    return vc0 * (f / f0) ** -x        # Rayleigh 波相速度 v_c，公式 (5)

def group_velocity(vc):
    return vc * vu_ratio               # 群速度 v_u = v_c / (1 + x)

def attenuation_term(f, r):
    return 2 * np.pi * r * (1 + x) * f**(1 + x) / (vc0 * Q0 * f0**x)  # 衰减参数 β，公式 (8)

def attenuation_integral(beta):
    return 2 * np.log(1 + 1 / beta) * np.exp(-2 * beta) + (1 - np.exp(-beta)) * np.exp(-beta) * np.sqrt(2 * np.pi / beta)  # c(β) 衰减函数，公式 (9)

# ------------------------
# 粒径分布函数：log-raised cosine 分布（公式 20）
# ------------------------
def log_raised_cosine(D, D50, sg):
    s = sg / np.sqrt(1 / 3 - 2 / np.pi**2)        # 将 sg 转为 raised cosine 分布区间宽度
    logD = np.log(D)                              # 取对数
    logD50 = np.log(D50)                          # 中值粒径的对数
    mask = np.abs(logD - logD50) < s              # 只在有效区间内计算概率密度
    p = np.zeros_like(D)                          # 初始化为零
    p[mask] = (1 + np.cos(np.pi * (logD[mask] - logD50) / s)) / (2 * s * D[mask])  # 分布公式 (20)
    return p                                       # 返回概率密度值（未归一化）

# ------------------------
# 主函数：计算频率 f 下总功率谱密度（公式 10 + 7）
# ------------------------
def total_psd(f, H, theta, qb, W, r, D50, sg, tau_star_c=0.045):
    def integrand(D):
        Vp = np.pi * D**3 / 6                      # 颗粒体积
        m = rho_s * Vp                             # 颗粒质量
        Cd = drag_coefficient(D)                   # 拖曳系数
        print('Cd = ', Cd)
        w_st = terminal_velocity(D)                # 终端沉速
        u_star = np.sqrt(g * H * np.sin(theta))    # 床面剪切速度
        R = (rho_s - rho_f) / rho_f                # 相对密度
        tau_star = u_star**2 / (R * g * D)         # Shields 应力
        Hb = 1.44 * D * (tau_star / tau_star_c)**0.5  # 动沙层厚度，公式 (15)
        Ub = 1.56 * np.sqrt(R * g * D) * (tau_star / tau_star_c)**0.56  # 水平速度，公式 (14)
        H_hat = (3 * Cd * rho_f * Hb) / (2 * rho_s * D * np.cos(theta)) # 无量纲床层厚度
        wi = w_st * np.cos(theta) * np.sqrt(1 - np.exp(-H_hat))         # 垂直冲击速度，公式 (16)
        ws_mean = H_hat * w_st * np.cos(theta) / (2 * np.log(np.exp(H_hat / 2) + np.sqrt(np.exp(H_hat) - 1)))  # 平均沉速，公式 (17)
        t_i = Hb / (C1 * ws_mean)                # 冲击间隔时间，公式 (18)
        pD = log_raised_cosine(np.array([D]), D50, sg)[0]  # 获取粒径 D 的概率密度
        qbD = pD * qb                            # 单粒径通量，q_{bD}
        n_ti = C1 * W * qbD / (Vp * Ub * Hb)     # 冲击率，公式 (13)
        vc = rayleigh_velocity(f)                # 相速度
        vu = group_velocity(vc)                  # 群速度
        beta = attenuation_term(f, r)            # 衰减参数 β
        c_beta = attenuation_integral(beta)      # 衰减函数 c(β)
        PSD = n_ti * (np.pi**2 * f**3 * m**2 * wi**2) / (rho_s**2 * vc**3 * vu**2) * c_beta  # PSD 单粒径，公式 (7)
        return PSD                                # 返回当前粒径的 PSD 贡献

    PSD_total, _ = quad(integrand, D50 * 0.1, D50 * 3, limit=200)  # 粒径积分，公式 (10)
    return PSD_total

# ------------------------
# 可视化：绘制频率 vs PSD 曲线
# ------------------------
def plot_total_psd():
    H = 4.0                                   # 水深 (m)
    theta = np.radians(1.4)                   # 河床坡度 (单位：弧度)
    qb = 1e-3                                 # 总床载通量 (m²/s)
    W = 50                                    # 河道宽度 (m)
    r = 600                                   # 测站距河流距离 (m)
    D50 = 0.15                                # 中值粒径 (m)
    sg = 0.52                                 # 粒径分布宽度（对数标准差）
    #freqs = np.logspace(0.1, 1.3, 50)           # 频率范围：1 Hz 到 100 Hz，对数分布
    freqs = np.linspace(0.1, 20, 100)  # 避免从0开始导致 log(0) 错误
    
    # 计算对应 PSD 值
    psd_values = [total_psd(f, H, theta, qb, W, r, D50, sg) for f in freqs]
    
    # 绘图
    plt.figure(figsize=(8, 5))
    plt.plot(freqs, 10 * np.log10(psd_values), label=f'r = {r} m')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Total PSD (dB rel. velocity power)")
    plt.title("Predicted PSD from Sediment Transport\n(Tsai et al., 2012 model)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
                         # 显示图像

# ------------------------
# 执行主函数
# ------------------------
if __name__ == "__main__":
    plot_total_psd()

'''
# 写入文件
file_path = "/mnt/data/tsai_seismic_psd_model_commented.py"
with open(file_path, "w") as f:
    f.write(detailed_code)

file_path  # 提供下载链接
'''

