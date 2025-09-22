import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
from scipy.integrate import quad

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

from scipy.stats import uniform, truncnorm, truncexpon, lognorm, weibull_min, truncweibull_min

# 区间范围
a, b = 0.3, 1.2
x = np.linspace(a, b, 500)

# 1. Uniform 分布 [a,b]
dist_uniform = uniform(loc=a, scale=b-a)
print(dist_uniform.support())

# 2. Truncated Normal 分布
mu, sigma = 0.75, 0.15
a_, b_ = (a - mu) / sigma, (b - mu) / sigma
dist_norm = truncnorm(a_, b_, loc=mu, scale=sigma)
print(dist_norm.support())

# 3. Truncated Exponential 分布
scale_exp = 0.4
dist_expon = truncexpon((b - a) / scale_exp, loc=a, scale=scale_exp)
print(dist_expon.support())

# 4. Truncated LogNormal 分布
# lognormal 默认定义在 (0, ∞)，所以必须裁剪到 [a,b]
s, scale_logn = 0.4, np.exp(np.log(0.7))   # 形状和尺度参数
dist_logn = lognorm(s, loc=0, scale=scale_logn)
# 注意：lognorm 没有直接的 trunc 版本，需要手动裁剪或近似
pdf_logn = lambda xx: dist_logn.pdf(xx) / (dist_logn.cdf(b) - dist_logn.cdf(a))
#print(pdf_logn.support())


# 5. Truncated Weibull 分布
c = 2.0   # 形状参数
scale_weib = 0.6
dist_weib = weibull_min(c, loc=0, scale=scale_weib)
pdf_weib = lambda xx: dist_weib.pdf(xx) / (dist_weib.cdf(b) - dist_weib.cdf(a))
#print(pdf_weib.support())

# 6. Truncated Weibull 分布
c2 = 2.0   # 形状参数
scale_weib2 = 0.6
loc2 = 0.0
a_standard = (0.3-loc2)/scale_weib2
b_standard = (1.2-loc2)/scale_weib2
dist_weib2 = truncweibull_min(c, a_standard, b_standard)
print(dist_weib2.support())


# --- 绘图 ---
plt.figure(figsize=(8,5))
plt.plot(x, dist_uniform.pdf(x), label="Uniform [0.3,1.2]")
plt.plot(x, dist_norm.pdf(x), label="Truncated Normal")
plt.plot(x, dist_expon.pdf(x), label="Truncated Exponential")
plt.plot(x, pdf_logn(x), label="Truncated LogNormal")
plt.plot(x, pdf_weib(x), label="Truncated Weibull")
plt.plot(x, dist_weib2.pdf(x), '-o', label="Truncated Weibull 22")

plt.xlabel("粒径 r")
plt.ylabel("概率密度")
plt.legend()
plt.title("常见5种截断分布（区间 [0.3,1.2] 内积分=1）")
plt.grid(True)
plt.show()
