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
x_data = np.array([0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7])
y_data = np.array([5962.3,5962.3,5962.3,5962.3,5962.3,5962.3,6447.5,7220.7,7844.9])



# 配色和标记
# 配色1：'#32037D', '#7C1A97', '#C94E65'，蓝色、紫色、红色
# 配色2：'#1f77b4', '#ff7f0e', '#2ca02c'，蓝色、橙色、绿色
# 配色3：'#015493', '#019092', '#999999'，蓝色、绿色、灰色
colors = [ '#999999', '#019092', '#015493']  
markers = ['o', 's', 'D']                   # 圆形、方形、菱形
labels = ['Narrow Grading', 'Medium Grading', 'Wide Grading']
x_errs = [x_err1, x_err2, x_err3]
y_uni_errs = [y_uni_err1, y_uni_err2, y_uni_err3]
y_nor_errs = [y_nor_err1, y_nor_err2, y_nor_err3]
y_wei_r_errs = [y_wei_r_err1, y_wei_r_err2, y_wei_r_err3]
y_wei_l_errs = [y_wei_l_err1, y_wei_l_err2, y_wei_l_err3]
y_exp_errs = [y_exp_err1, y_exp_err2, y_exp_err3]

linestyles = ['-','--','-.']

# 绘制每个子图
fig, axs = plt.subplots(1, 5, figsize=(16/2.54, 9/2.54), sharey=True)

# 绘制统一函数
def plot_error_bars(ax, x, ys, x_errs, title, ylabel=False):
    for y, color, marker, xerr, label, ls in zip(ys, colors, markers, x_errs, labels,linestyles):
        ax.errorbar(x, y, xerr=xerr, fmt=marker, color=color, label=label,linestyle=ls,
                    capsize=3, elinewidth=1, markersize=4)

    ax.set_title(title, fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.3)

# 绘制每个子图
plot_error_bars(axs[0], x_arr, [y_uniform1, y_uniform2, y_uniform3], x_errs, "Uniform", ylabel=True)
plot_error_bars(axs[1], x_arr, [y_normal1, y_normal2, y_normal3], x_errs, "Normal")
plot_error_bars(axs[2], x_arr, [y_weibull_r1, y_weibull_r2, y_weibull_r3], x_errs, "Right-skewed Weibull")
plot_error_bars(axs[3], x_arr, [y_weibull_l1, y_weibull_l2, y_weibull_l3], x_errs, "Left-skewed Weibull")
plot_error_bars(axs[4], x_arr, [y_exponential1, y_exponential2, y_exponential3], x_errs, "Exponential")


# 在每个子图添加scatter点和延长线
for ax in axs:
    ax.scatter(single_particle_x, single_particle_y, 
               color='black', marker='^', s=30, label='Single Particle')
    ## 纵向延长线
    #ax.vlines(single_particle_x, ymin=1500, ymax=single_particle_y, 
    #          color='black', linestyle=':', linewidth=0.8)
    ## 横向延长线
    #ax.hlines(single_particle_y, xmin=0.25, xmax=single_particle_x, 
    #          color='black', linestyle=':', linewidth=0.8)


# 设置全局图例
fig.legend(fontsize=8, bbox_to_anchor=(0.16, 0.06), ncol = 4, loc=2, borderaxespad=0)

# 调整布局和图例位置
plt.tight_layout()
plt.subplots_adjust(bottom=0.15, top=0.95, left=0.1, right=0.95)  # 给图例留空间

# 显示图表
# Save the plot as EPS
# plt.savefig('C:\\Users\\HY\\Desktop\\python_plot.eps', format='eps', dpi=300)
plt.show()
