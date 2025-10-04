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
colors = [ '#999999', '#019092', '#015493']  


DEM_Volumn = np.array([ 1000.,  3250.,  5500.,  7750., 10000.])
flow_rate = np.array([173.7, 362.1, 550.6, 739. , 927.4])
EF_max = np.array([4684., 4684., 4684., 4684., 4684.])

fig, axs = plt.subplots(1, 5, figsize=(16/2.54, 9/2.54), sharey=True)

# 数据1： 三角形脉冲叠加
total_force_060 = np.array([ 3678.,  3991.,  6402.,  8033., 10135.])  # total_force
total_force_064 = np.array([ 3678.,  4420.,  6681.,  8695., 10656.])  # total_force
total_force_068 = np.array([ 3678.,  4800.,  6946.,  9234., 11177.])  # total_force
axs[0].plot(flow_rate,total_force_060, 'o-', color=colors[0], label=r'$\alpha_s = 0.60$')
axs[0].plot(flow_rate,total_force_064, 's-', color=colors[1], label=r'$\alpha_s = 0.64$')
axs[0].plot(flow_rate,total_force_068, '^-', color=colors[2], label=r'$\alpha_s = 0.68$')

# 数据2： 正弦波脉冲叠加
total_force_060 = np.array([ 3678.,  5373.,  7720., 10162., 12637.])  # total_force
total_force_064 = np.array([ 3678.,  5598.,  8080., 10747., 13490.])  # total_force
total_force_068 = np.array([ 3678.,  5787.,  8569., 11471., 14348.])  # total_force
axs[1].plot(flow_rate,total_force_060, 'o-', color=colors[0])
axs[1].plot(flow_rate,total_force_064, 's-', color=colors[1])
axs[1].plot(flow_rate,total_force_068, '^-', color=colors[2])


# 数据3： 梯形波脉冲叠加
total_force_060 = np.array([ 3678.,  7357.,  8977., 11716., 14567.])  # total_force
total_force_064 = np.array([ 3678.,  7357.,  9689., 12823., 16163.])  # total_force
total_force_068 = np.array([ 3678.,  7357., 10220., 13542., 17288.])  # total_force
axs[2].plot(flow_rate,total_force_060, 'o-', color=colors[0])
axs[2].plot(flow_rate,total_force_064, 's-', color=colors[1])
axs[2].plot(flow_rate,total_force_068, '^-', color=colors[2])

# 数据5： 高斯波脉冲叠加
total_force_060 = np.array([3678., 3754., 4691., 6214., 7793.])  # total_force
total_force_064 = np.array([3678., 3808., 4956., 6637., 8316.])  # total_force
total_force_068 = np.array([3678., 3881., 5250., 7025., 8833.])  # total_force
axs[3].plot(flow_rate,total_force_060, 'o-', color=colors[0])
axs[3].plot(flow_rate,total_force_064, 's-', color=colors[1])
axs[3].plot(flow_rate,total_force_068, '^-', color=colors[2])

# 数据4： 指数波脉冲叠加
total_force_060 = np.array([3753., 4331., 5156., 6062., 7001.])  # total_force
total_force_064 = np.array([3773., 4422., 5312., 6294., 7292.])  # total_force
total_force_068 = np.array([3797., 4513., 5478., 6504., 7649.])  # total_force
axs[4].plot(flow_rate,total_force_060, 'o-', color=colors[0])
axs[4].plot(flow_rate,total_force_064, 's-', color=colors[1])
axs[4].plot(flow_rate,total_force_068, '^-', color=colors[2])

# 配色和标记
# 配色1：'#32037D', '#7C1A97', '#C94E65'，蓝色、紫色、红色
# 配色2：'#1f77b4', '#ff7f0e', '#2ca02c'，蓝色、橙色、绿色
# 配色3：'#015493', '#019092', '#999999'，蓝色、绿色、灰色
colors = [ '#999999', '#019092', '#015493']  
markers = ['o', 's', 'D']                   # 圆形、方形、菱形
labels = ['Narrow Grading', 'Medium Grading', 'Wide Grading']


# 在每个子图添加scatter点和延长线
for ax in axs:
    ax.plot(flow_rate, EF_max, color='black', marker=None, linestyle='-.', label='Expected force')
    ## 纵向延长线
    #ax.vlines(single_particle_x, ymin=1500, ymax=single_particle_y, 
    #          color='black', linestyle=':', linewidth=0.8)
    ## 横向延长线
    #ax.hlines(single_particle_y, xmin=0.25, xmax=single_particle_x, 
    #          color='black', linestyle=':', linewidth=0.8)


# 设置全局图例
fig.legend(fontsize=8, bbox_to_anchor=(0.16, 0.06), ncol = 3, loc=2, borderaxespad=0)

# 调整布局和图例位置
plt.tight_layout()
plt.subplots_adjust(bottom=0.15, top=0.95, left=0.1, right=0.95)  # 给图例留空间

# 显示图表
# Save the plot as EPS
# plt.savefig('C:\\Users\\HY\\Desktop\\python_plot.eps', format='eps', dpi=300)
plt.show()

