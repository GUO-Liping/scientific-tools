import matplotlib 
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 实测数据
x_center = 0.75
x_radius = 0.001
field_x = np.linspace(x_center - x_radius, x_center + x_radius, 8)

field_n = np.array([19, 65, 60, 38, 20, 16, 8, 7])
upper, lower = max(field_x), min(field_x)
dx = (upper - lower) / (len(field_x)-1)
field_pdf = field_n / (dx * field_n.sum())

# 展开数据用于拟合
data_expanded = np.repeat(field_x, field_n)
# 绘图用连续x轴
x_cont = np.linspace(lower, upper, 500)

# 1. 截断均匀分布 (uniform 在 [lower, upper] 上天然归一化)
uniform_dist = stats.uniform(loc=lower, scale=upper - lower)
pdf_uniform = uniform_dist.pdf(x_cont)
print(f"截断均匀分布 Uniform:\t loc={uniform_dist.kwds['loc']:.4f}, scale={uniform_dist.kwds['scale']:.4f}")

# 2. 截断正态分布
mu = (upper + lower) / 2
sigma = np.std(data_expanded, ddof=1)  # 无偏估计
a, b = (lower - mu) / sigma, (upper - mu) / sigma
trunc_norm = stats.truncnorm(a, b, loc=mu, scale=sigma)
pdf_normal = trunc_norm.pdf(x_cont)/(trunc_norm.cdf(upper) - trunc_norm.cdf(lower))
print(f"截断正态分布 TruncNorm:\t mu={mu:.4f}, sigma={sigma:.4f}, a={a:.4f}, b={b:.4f}")

# 3. 截断 Weibull 右偏
c_wei_r, loc_wei_r, scale_wei_r = stats.weibull_min.fit(data_expanded)
a_r, b_r = (lower - loc_wei_r) / scale_wei_r, (upper - loc_wei_r) / scale_wei_r
trunc_weibull_r = stats.truncweibull_min(c_wei_r, a_r, b_r, loc=loc_wei_r, scale=scale_wei_r)
pdf_weibull_r = trunc_weibull_r.pdf(x_cont)/(trunc_weibull_r.cdf(upper) - trunc_weibull_r.cdf(lower))
print(f"截断Weibull右偏 TruncWeibullMin_r:\t c={c_wei_r:.4f}, loc={loc_wei_r:.4f}, scale={scale_wei_r:.4f}, a={a_r:.4f}, b={b_r:.4f}")

# 4. 截断 Weibull 左偏（用反向数据拟合）
c_wei_l, loc_wei_l, scale_wei_l = stats.weibull_min.fit(np.repeat(field_x, field_n[::-1]))
a_l, b_l = (lower - loc_wei_l) / scale_wei_l, (upper - loc_wei_l) / scale_wei_l
trunc_weibull_l = stats.truncweibull_min(c_wei_l, a_l, b_l, loc=loc_wei_l, scale=scale_wei_l)
pdf_weibull_l = trunc_weibull_l.pdf(x_cont)/(trunc_weibull_l.cdf(upper) - trunc_weibull_l.cdf(lower))
print(f"截断Weibull左偏 TruncWeibullMin_l:\t c={c_wei_l:.4f}, loc={loc_wei_l:.4f}, scale={scale_wei_l:.4f}, a={a_l:.4f}, b={b_l:.4f}")

# 5. 截断指数分布
b_fit, loc_fit, scale_fit = stats.truncexpon.fit(data_expanded)
trunc_expon = stats.truncexpon(b_fit, loc=loc_fit, scale=scale_fit)
pdf_expon = trunc_expon.pdf(x_cont)/(trunc_expon.cdf(upper) - trunc_expon.cdf(lower))
print(f"截断指数分布 TruncExpon:\t b={b_fit:.4f}, loc={loc_fit:.4f}, scale={scale_fit:.4f}")

# 绘图
plt.figure(figsize=(10, 6))
plt.bar(field_x, field_pdf, width=dx * 0.8, alpha=0.6, color='gray', label='实测数据')
plt.plot(x_cont, pdf_uniform, label='截断均匀分布')
plt.plot(x_cont, pdf_normal, label='截断正态分布')
plt.plot(x_cont, pdf_weibull_r, label='截断Weibull右偏')
plt.plot(x_cont, pdf_weibull_l, label='截断Weibull左偏')
plt.plot(x_cont, pdf_expon, label='截断指数分布')
plt.xlabel('颗粒直径 (m)')
plt.ylabel('概率密度')
plt.legend()
plt.title('颗粒直径分布截断拟合')
plt.show()
