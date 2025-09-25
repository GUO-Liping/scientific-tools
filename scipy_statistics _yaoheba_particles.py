
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 实测数据
field_x = np.array([0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4])/2
field_n = np.array([19, 65, 60, 38, 20, 16, 8, 7])
dx = 0.3 / 2
field_pdf = field_n / (dx * field_n.sum())

# 展开数据用于拟合
data_expanded = np.repeat(field_x, field_n)

# 绘图用连续x轴
x_cont = np.linspace(min(field_x), max(field_x), 500)

# 1. 均匀分布拟合
uni_a, uni_b = min(field_x), max(field_x)
pdf_uniform = stats.uniform.pdf(x_cont, loc=uni_a, scale=uni_b - uni_a)

# 2. 正态分布拟合
mu, sigma = stats.norm.fit(data_expanded)
pdf_normal = stats.norm.pdf(x_cont, mu, sigma)

# 3. Weibull 分布拟合（右偏）
c, loc, scale = stats.weibull_min.fit(data_expanded, floc=0)
print('c, loc, scale=',c, loc, scale)
pdf_weibull = stats.weibull_min.pdf(x_cont, c, loc=loc, scale=scale)


# 3. Weibull 分布拟合（左偏）
shift = max(data_expanded) + 1e-6
#data_shifted = -data_expanded + shift
data_shifted = data_expanded[::-1]
c_weibull, loc_weibull, scale_weibull = stats.weibull_min.fit(data_shifted, floc=0)
print('c_weibull, loc_weibull, scale_weibull=',c_weibull, loc_weibull, scale_weibull)
pdf_weibull_left = stats.weibull_min.pdf(x_cont[::-1], c_weibull, loc=loc_weibull, scale=scale_weibull)


# 4. 左偏分布拟合（反向 Gamma）
# 平移并取负，使数据为正数
shift = max(data_expanded) + 1e-6
data_shifted = -data_expanded + shift
# 拟合 Gamma
a_gamma, loc_gamma, scale_gamma = stats.gamma.fit(data_shifted, floc=0)
print('a_gamma, loc_gamma, scale_gamma =',a_gamma, loc_gamma, scale_gamma)
# 计算反向 Gamma PDF
pdf_gamma_left = stats.gamma.pdf(shift - x_cont, 1.26, loc=0, scale=scale_gamma)

# 5. 指数分布拟合
lambda_exp = 1 / np.mean(data_expanded)
pdf_expon = stats.expon.pdf(x_cont, scale=1/lambda_exp)

# 绘图
plt.figure(figsize=(10,6))
plt.bar(field_x, field_pdf, width=dx*0.8, alpha=0.4, color='gray', label='实测数据')
plt.plot(x_cont, pdf_uniform, label='均匀分布拟合')
plt.plot(x_cont, pdf_normal, label='正态分布拟合')
plt.plot(x_cont, pdf_weibull, label='Weibull 分布拟合')
plt.plot(x_cont, pdf_weibull_left, label='Weibull 逆分布拟合')
#plt.plot(x_cont, pdf_gamma_left, label='左偏 反向 Gamma 拟合')
plt.plot(x_cont, pdf_expon, label='指数分布拟合')
plt.xlabel('颗粒直径 (m)')
plt.ylabel('概率密度')
plt.legend()
plt.title('颗粒直径分布拟合')
plt.show()
