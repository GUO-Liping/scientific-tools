import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置字体、大小、风格
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'lines.linewidth': 1,
    'lines.markersize': 5
})

# 使用 Seaborn 风格
plt.style.use('seaborn-v0_8-whitegrid')  # 更简洁的科研风格

# 数据
x_arr = np.array([0.45, 0.75, 1.05])
x_err1 = 0.01
x_err2 = 0.05
x_err3 = 0.15

y_uniform1 = np.array([2336.922, 6490.768, 12721.535])
y_uniform2 = np.array([2346.153, 6499.998, 12730.766])
y_uniform3 = np.array([2423.076, 6576.921, 12807.689])

y_normal1 = np.array([2336.922, 6490.768, 12721.535])
y_normal2 = np.array([2346.153, 6499.998, 12730.766])
y_normal3 = np.array([2423.076, 6576.921, 12807.689])

y_weibull_r1 = np.array([2301.529, 6431.798, 12638.983])
y_weibull_r2 = np.array([2168.675, 6204.637, 12317.495])
y_weibull_r3 = np.array([1886.773, 5686.973, 11564.063])

y_weibull_l1 = np.array([2370.88, 6547.419, 12800.904])
y_weibull_l2 = np.array([2515.416, 6781.795, 13126.013])
y_weibull_l3 = np.array([2918.307, 7412.461, 13983.563])

y_exponential1 = np.array([2319.749, 6459.632, 12683.12])
y_exponential2 = np.array([2251.083, 6352.695, 12529.621])
y_exponential3 = np.array([2175.638, 6145.718, 12182.775])

# 配色和标记
# 配色1：'#32037D', '#7C1A97', '#C94E65'，蓝色、紫色、红色
# 配色2：'#1f77b4', '#ff7f0e', '#2ca02c'，蓝色、橙色、绿色
# 配色3：'#015493', '#019092', '#999999'，蓝色、绿色、灰色
colors = [ '#999999', '#019092', '#015493']  
markers = ['o', 's', 'D']                   # 圆形、方形、菱形
labels = ['Narrow \nGrading', 'Medium \nGrading', 'Wide \nGrading']
x_errs = [x_err1, x_err2, x_err3]
linestyles = ['-','--','-.']

# 绘制每个子图
fig, axs = plt.subplots(1, 5, figsize=(16/2.54, 9/2.54), sharey=True)

# 绘制统一函数
def plot_error_bars(ax, x, ys, x_errs, title, ylabel=False):
    for y, color, marker, err, label, ls in zip(ys, colors, markers, x_errs, labels,linestyles):
        ax.errorbar(x, y, xerr=err, fmt=marker, color=color, label=label,linestyle=ls,
                    capsize=5, elinewidth=1, markersize=4)

    ax.set_title(title, fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.3)

# 绘制每个子图
plot_error_bars(axs[0], x_arr, [y_uniform1, y_uniform2, y_uniform3], x_errs, "Uniform", ylabel=True)
axs[0].set_ylabel('Expected impact force (kN)', fontsize=10)
plot_error_bars(axs[1], x_arr, [y_normal1, y_normal2, y_normal3], x_errs, "Normal")
plot_error_bars(axs[2], x_arr, [y_weibull_r1, y_weibull_r2, y_weibull_r3], x_errs, "Right-skewed Weibull")
axs[2].set_xlabel('Particle radius (m)', fontsize=10)
plot_error_bars(axs[3], x_arr, [y_weibull_l1, y_weibull_l2, y_weibull_l3], x_errs, "Left-skewed Weibull")
plot_error_bars(axs[4], x_arr, [y_exponential1, y_exponential2, y_exponential3], x_errs, "Exponential")

# 设置全局图例
fig.legend(labels, fontsize=8, bbox_to_anchor=(0.88, 0.8), loc=2, borderaxespad=0)

# 调整布局和图例位置
plt.tight_layout()
plt.subplots_adjust(bottom=0.12, top=0.85, left=0.1, right=0.88)  # 给图例留空间

# 显示图表
# Save the plot as EPS
# plt.savefig('C:\\Users\\HY\\Desktop\\python_plot.eps', format='eps', dpi=300)
plt.show()
