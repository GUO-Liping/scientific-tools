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

DEM_Volumn = np.array([ 1000.,  3250.,  5500.,  7750., 10000.])
flow_rate = np.array([173.7, 362.1, 550.6, 739. , 927.4])
EF_max = np.array([[4684., 4684., 4684., 4684., 4684.]])

fig, axs = plt.subplots(1, 5, figsize=(16/2.54, 9/2.54), sharey=True)

# 数据1： 三角形脉冲叠加
total_force_060 = np.array([ 3678.,  3991.,  6402.,  8033., 10135.])  # total_force
total_force_064 = np.array([ 3678.,  4420.,  6681.,  8695., 10656.])  # total_force
total_force_068 = np.array([ 3678.,  4800.,  6946.,  9234., 11177.])  # total_force
axs[0].plot(flow_rate,total_force_060)
axs[0].plot(flow_rate,total_force_064)
axs[0].plot(flow_rate,total_force_068)

# 数据2： 正弦波脉冲叠加
total_force_060 = np.array([ 3678.,  5373.,  7720., 10162., 12637.])  # total_force
total_force_064 = np.array([ 3678.,  5598.,  8080., 10747., 13490.])  # total_force
total_force_068 = np.array([ 3678.,  5787.,  8569., 11471., 14348.])  # total_force
axs[1].plot(flow_rate,total_force_060)
axs[1].plot(flow_rate,total_force_064)
axs[1].plot(flow_rate,total_force_068)
axs[1].scatter(flow_rate,EF_max)

# 数据3： 矩形波脉冲叠加
total_force_060 = np.array([ 7357., 11035., 14714., 18392., 22071.])  # total_force
total_force_064 = np.array([ 7357., 11035., 14714., 18392., 22071.])  # total_force
total_force_068 = np.array([ 7357., 11035., 14714., 18392., 25749.])  # total_force


# 数据3： 梯形波脉冲叠加
total_force_060 = np.array([ 3678.,  7357.,  8977., 11716., 14567.])  # total_force
total_force_064 = np.array([ 3678.,  7357.,  9689., 12823., 16163.])  # total_force
total_force_068 = np.array([ 3678.,  7357., 10220., 13542., 17288.])  # total_force
axs[2].plot(flow_rate,total_force_060)
axs[2].plot(flow_rate,total_force_064)
axs[2].plot(flow_rate,total_force_068)

# 数据5： 高斯波脉冲叠加
total_force_060 = np.array([3678., 3754., 4691., 6214., 7793.])  # total_force
total_force_064 = np.array([3678., 3808., 4956., 6637., 8316.])  # total_force
total_force_068 = np.array([3678., 3881., 5250., 7025., 8833.])  # total_force
axs[3].plot(flow_rate,total_force_060)
axs[3].plot(flow_rate,total_force_064)
axs[3].plot(flow_rate,total_force_068)

# 数据4： 指数波脉冲叠加
total_force_060 = np.array([3753., 4331., 5156., 6062., 7001.])  # total_force
total_force_064 = np.array([3773., 4422., 5312., 6294., 7292.])  # total_force
total_force_068 = np.array([3797., 4513., 5478., 6504., 7649.])  # total_force
axs[4].plot(flow_rate,total_force_060)
axs[4].plot(flow_rate,total_force_064)
axs[4].plot(flow_rate,total_force_068)


plt.plot(flow_rate, total_force_060)
plt.plot(flow_rate, total_force_064)
plt.plot(flow_rate, total_force_068)

# 数据
x_arr = np.array([0.45, 0.75, 1.05])
x_err1 = 0.01
x_err2 = 0.05
x_err3 = 0.15

y_min1 = np.array([ 2233.8, 6318.5, 12480.0])
y_min2 = np.array([ 1846.2, 5653.8, 11538.5])
y_min3 = np.array([ 1038.5, 4153.8, 9346.2])

#y_min1 = np.array([ 2233.8, 1846.2, 1038.5])
#y_min2 = np.array([ 6318.5, 5653.8, 4153.8])
#y_min3 = np.array([ 12480. , 11538.5,  9346.2])

y_max1 = np.array([ 2441.5, 6664.6, 12964.6])
y_max2 = np.array([ 2884.6, 7384.6, 13961.5])
y_max3 = np.array([ 4153.8, 9346.2, 16615.4])

#y_max1 = np.array([ 2441.5, 2884.6, 4153.8])
#y_max2 = np.array([ 6664.6, 7384.6, 9346.2])
#y_max3 = np.array([ 12964.6, 13961.5, 16615.4])

y_uniform1 = np.array([2336.922, 6490.768, 12721.535])
y_uniform2 = np.array([2346.153, 6499.998, 12730.766])
y_uniform3 = np.array([2423.076, 6576.921, 12807.689])
y_uni_err1 = np.vstack((y_uniform1-y_min1, y_max1-y_uniform1))
y_uni_err2 = np.vstack((y_uniform2-y_min2, y_max2-y_uniform2))
y_uni_err3 = np.vstack((y_uniform3-y_min3, y_max3-y_uniform3))

y_normal1  = np.array([2336.922, 6490.768, 12721.535])
y_normal2  = np.array([2346.153, 6499.998, 12730.766])
y_normal3  = np.array([2423.076, 6576.921, 12807.689])
y_nor_err1 = np.vstack((y_normal1-y_min1, y_max1-y_normal1))
y_nor_err2 = np.vstack((y_normal2-y_min2, y_max2-y_normal2))
y_nor_err3 = np.vstack((y_normal3-y_min3, y_max3-y_normal3))

y_weibull_r1 = np.array([2301.529, 6431.798, 12638.983])
y_weibull_r2 = np.array([2168.675, 6204.637, 12317.495])
y_weibull_r3 = np.array([1886.773, 5686.973, 11564.063])
y_wei_r_err1 = np.vstack((y_weibull_r1-y_min1, y_max1-y_weibull_r1))
y_wei_r_err2 = np.vstack((y_weibull_r2-y_min2, y_max2-y_weibull_r2))
y_wei_r_err3 = np.vstack((y_weibull_r3-y_min3, y_max3-y_weibull_r3))

y_weibull_l1 = np.array([2370.88, 6547.419, 12800.904])
y_weibull_l2 = np.array([2515.416, 6781.795, 13126.013])
y_weibull_l3 = np.array([2918.307, 7412.461, 13983.563])
y_wei_l_err1 = np.vstack((y_weibull_l1-y_min1, y_max1-y_weibull_l1))
y_wei_l_err2 = np.vstack((y_weibull_l2-y_min2, y_max2-y_weibull_l2))
y_wei_l_err3 = np.vstack((y_weibull_l3-y_min3, y_max3-y_weibull_l3))

y_exponential1 = np.array([2319.749, 6459.632, 12683.12])
y_exponential2 = np.array([2251.083, 6352.695, 12529.621])
y_exponential3 = np.array([2175.638, 6145.718, 12182.775])
y_exp_err1   = np.vstack((y_exponential1-y_min1, y_max1-y_exponential1))
y_exp_err2   = np.vstack((y_exponential2-y_min2, y_max2-y_exponential2))
y_exp_err3   = np.vstack((y_exponential3-y_min3, y_max3-y_exponential3))

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

single_particle_min = np.array([0.44, 0.4, 0.3, 0.74, 0.7, 0.6, 1.04, 1.0, 0.9])
single_particle_max = np.array([0.46, 0.5, 0.6, 0.76, 0.8, 0.9, 1.06, 1.1, 1.2])
single_force_min    = np.array([2233.8, 1846.2, 1038.5, 6318.5, 5653.8, 4153.8, 12480. , 11538.5,  9346.2])
single_force_max    = np.array([2441.5, 2884.6, 4153.8, 6664.6, 7384.6, 9346.2, 12964.6, 13961.5, 16615.4])

axs[0].set_ylabel('Expected impact force (kN)', fontsize=10)
axs[2].set_xlabel('Particle radius (m)', fontsize=10)

single_particle_x = np.array([0.45,0.75,1.05])
single_particle_y = np.array([2336,6490,12721])

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

