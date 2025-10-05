import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置字体、大小、风格（期刊风格）
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'lines.linewidth': 1.2,
    'lines.markersize': 5,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

# 配色：Nature风格（蓝、橙、灰）
colors = ['#1f77b4', '#ff7f0e', '#7f7f7f']
markers = ['o', 's', 'D']        # 圆、方、菱形
linestyles = ['-', '--', '-.']   # 实线、虚线、点划线
labels = [r'$\alpha_s = 0.60$', r'$\alpha_s = 0.64$', r'$\alpha_s = 0.68$']

# 横轴 & 参考线
DEM_impact_rate = np.array([59, 123, 187, 251, 315])
EF_max = np.array([4684.] * 5)

# 各图数据
data_sets = [
    # 三角脉冲
    [np.array([3678., 3991., 6402., 8033., 10135.]),
     np.array([3678., 4420., 6681., 8695., 10656.]),
     np.array([3678., 4800., 6946., 9234., 11177.])],
    # 正弦波脉冲
    [np.array([3678., 5373., 7720., 10162., 12637.]),
     np.array([3678., 5598., 8080., 10747., 13490.]),
     np.array([3678., 5787., 8569., 11471., 14348.])],
    # 梯形波脉冲
    [np.array([3678., 7357., 8977., 11716., 14567.]),
     np.array([3678., 7357., 9689., 12823., 16163.]),
     np.array([3678., 7357., 10220., 13542., 17288.])],
    # 指数波脉冲
    [np.array([3753., 4331., 5156., 6062., 7001.]),
     np.array([3773., 4422., 5312., 6294., 7292.]),
     np.array([3797., 4513., 5478., 6504., 7649.])],
    # 高斯波脉冲
    [np.array([3678., 3754., 4691., 6214., 7793.]),
     np.array([3678., 3808., 4956., 6637., 8316.]),
     np.array([3678., 3881., 5250., 7025., 8833.])]
]

# 子图标题（可选）
titles = ['Triangle', 'Sine', 'Trapezoid', 'Exponential', 'Gaussian']

# 创建子图
fig, axs = plt.subplots(1, 5, figsize=(16/2.54, 9/2.54), sharey=True)

# 绘图循环
for idx, ax in enumerate(axs):
    for i in range(3):
        ax.plot(
            DEM_impact_rate,
            data_sets[idx][i],
            marker=markers[i],
            linestyle=linestyles[i],
            color=colors[i],
            markerfacecolor='white',
            markeredgecolor=colors[i],
            label=labels[i] if idx == 0 else None  # 只在第一个子图加label
        )
    # 添加参考线
    ax.plot(DEM_impact_rate, EF_max, color='black', linestyle=':', label='Expected force' if idx == 0 else None)
    # ✅ 添加水平网格线
    ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.7)
    # 设置子图标题（可选）
    ax.set_title(titles[idx], fontsize=9)
    ax.set_xlabel('Impact rate (s$^{-1}$)', fontsize=9)
    ax.tick_params(direction='in', length=3)

# 统一 y 轴标签
axs[0].set_ylabel('Total impact force (kN)', fontsize=9)

# 图例放底部（全局）
fig.legend(
    loc='lower center',
    ncol=4,
    bbox_to_anchor=(0.5, -0.02),
    frameon=False
)

# 布局调整
plt.tight_layout()
plt.subplots_adjust(bottom=0.18, top=0.93, left=0.10, right=0.98)

# 保存为高质量矢量图（推荐期刊投稿）
# plt.savefig('nature_ready_plot.pdf', dpi=600, bbox_inches='tight')

# 显示图形
plt.show()
