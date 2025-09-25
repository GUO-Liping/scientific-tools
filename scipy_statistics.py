import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, truncnorm, truncexpon, lognorm, weibull_min, truncweibull_min

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 区间范围
lower, upper = 0.15, 1.2
x_arr = np.linspace(lower, upper, 50)

# 实测数据
field_x = np.array([0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4])/2
field_n = np.array([19, 65, 60, 38, 20, 16, 8, 7])
field_pdf = field_n / (0.3/2 * field_n.sum())  # 概率密度

# 创建图形
plt.figure(figsize=(8, 5))
plt.plot(field_x, field_pdf, '-x', label="实测数据", color='blue', zorder=5)
plt.xlabel("粒径 r")
plt.ylabel("概率密度")

# 1. Uniform 分布
uniform_dist = uniform(loc=lower, scale=upper - lower)

# 2. Truncated Normal 分布
mu, sigma = (lower+upper)/2, 0.25
truncated_normal_dist = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

# 3. Truncated Exponential 分布
exp_scale = 0.5
truncated_exp_dist = truncexpon((upper - lower) / exp_scale, loc=lower, scale=exp_scale)

# 4. Truncated LogNormal 分布
s_right = 0.3
scale_right = 0.8
lognormal_dist = lognorm(s_right, scale=scale_right)
lognormal_pdf = lambda xx: lognormal_dist.pdf(xx) / (lognormal_dist.cdf(upper) - lognormal_dist.cdf(lower))

# 5. Truncated Weibull 分布
c_wei = 2.1
loc_wei, scale_wei = 0.0, 0.6
weibull_dist = truncweibull_min(c_wei, (lower - loc_wei) / scale_wei, (upper - loc_wei) / scale_wei, loc=loc_wei, scale=scale_wei)

# 6. Standard Weibull 分布 (裁剪)
weibull2_dist = weibull_min(2.5, loc=0, scale=0.6)
weibull2_pdf = lambda xx: weibull2_dist.pdf(xx) / (weibull2_dist.cdf(upper) - weibull2_dist.cdf(lower))

# --- 绘图 ---
plt.bar(field_x, field_pdf, label="field survey", width=0.1)

plt.plot(x_arr, uniform_dist.pdf(x_arr), label="Uniform [0.3, 1.2]")
plt.plot(x_arr, truncated_normal_dist.pdf(x_arr), label="Truncated Normal")
plt.plot(x_arr, truncated_exp_dist.pdf(x_arr), label="Truncated Exponential")
#plt.plot(x_arr, lognormal_pdf(x_arr), label="Truncated LogNormal")
plt.plot(x_arr, weibull_dist.pdf(x_arr), 'o-', label="Truncated Weibull")
#plt.plot(x_arr, weibull2_pdf(x_arr), label="Standard Weibull (Truncated)")

plt.xlabel("粒径 r")
plt.ylabel("概率密度")
plt.legend()
plt.title("常见5种截断分布（区间 [0.3, 1.2] 内积分=1）")
plt.grid(True)
plt.show()
