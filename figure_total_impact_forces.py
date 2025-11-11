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
# 使用 Seaborn 风格
plt.style.use('seaborn-v0_8-whitegrid')  # 更简洁的科研风格

#colors = ['#1f77b4', '#ff7f0e', '#7f7f7f']  # 配色：Nature风格（蓝、橙、灰）
#colors = ['#4189C8', '#F37E78', '#8D1755']  # 配色：Nature风格（蓝、红、褐）
colors = ['#4189C8', '#F37E78', '#7f7f7f']  # 配色：Nature风格（蓝、红、灰）


markers = ['o', 's', 'D']        # 圆、方、菱形
linestyles = ['-', '--', '-.']   # 实线、虚线、点划线
labels = ['Small particles dominant', 'Medium particles dominant', 'Large particles dominant']

# 横轴 & 参考线
DEM_impact_rate60 = np.array([ 59.,  77.,  96., 114., 132., 150., 169., 187., 205., 223., 242., 260., 278., 297., 315.])
DEM_impact_rate64 = np.array([ 63.,  82., 102., 121., 141., 160., 180., 199., 219., 238., 258., 277., 297., 316., 336.])
DEM_impact_rate68 = np.array([ 67.,  88., 108., 129., 150., 170., 191., 212., 233., 253., 274., 295., 315., 336., 357.])
EF_max_weibull_r = np.array([5177.] * 15)
EF_max_normal = np.array([6869.] * 15)
EF_max_weibull_l = np.array([8690.] * 15)

DEM_impact_rate = [DEM_impact_rate68, DEM_impact_rate68, DEM_impact_rate68]
EF_max = [EF_max_weibull_r, EF_max_normal, EF_max_weibull_l]
# 各图数据
data_sets = [
    # 三角脉冲
    [np.array([ 4066.,  4066.,  4066.,  4066.,  4378.,  5312.,  6083.,  6669.,  7154.,  7580.,  7922.,  8390.,  9200.,  9874., 10499.]),
     np.array([ 5387.,  5387.,  5387.,  5387.,  5801.,  7039.,  8061.,  8837.,  9479., 10044., 10497., 11118., 12191., 13084., 13912.]),
     np.array([ 6825.,  6825.,  6825.,  6825.,  7349.,  8917., 10211., 11194., 12008., 12724., 13297., 14084., 15444., 16574., 17623.])],
    # 正弦波脉冲
    [np.array([ 4066.,  4066.,  4069.,  5221.,  5921.,  6399.,  7157.,  7986.,  8635.,  9215., 10099., 10838., 11486., 12235., 13065.]),
     np.array([ 5387.,  5387.,  5392.,  6919.,  7847.,  8480.,  9484., 10582., 11442., 12211., 13383., 14362., 15220., 16213., 17313.]),
     np.array([ 6825.,  6825.,  6831.,  8765.,  9940., 10742., 12013., 13405., 14495., 15469., 16953., 18194., 19281., 20538., 21931.])],
    # 梯形波脉冲
    [np.array([ 4066.,  4066.,  5430.,  7223.,  8131.,  8131.,  8131.,  9277., 10247., 11100., 11784., 12545., 13626., 14525., 15358.]),
     np.array([ 5387.,  5387.,  7195.,  9572., 10775., 10775., 10775., 12293., 13579., 14709., 15615., 16624., 18056., 19247., 20351.]),
     np.array([ 6825.,  6825.,  9114., 12125., 13649., 13649., 13649., 15573., 17201., 18633., 19780., 21059., 22873., 24381., 25780.])],
    # 指数波脉冲
    [np.array([4066., 4230., 4366., 4527., 4780., 4990., 5215., 5426., 5741., 5991., 6226., 6545., 6809., 7056., 7311.]),
     np.array([5387., 5605., 5785., 5999., 6334., 6612., 6910., 7190., 7607., 7939., 8250., 8673., 9023., 9350., 9688.]),
     np.array([ 6825.,  7100.,  7328.,  7600.,  8024.,  8376.,  8754.,  9108.,  9636., 10057., 10450., 10987., 11430., 11844., 12273.])],
    # 高斯波脉冲
    [np.array([4066., 4066., 4066., 4066., 4146., 4292., 4548., 4873., 5255., 5689., 6129., 6590., 7066., 7519., 7994.]),
     np.array([ 5387.,  5387.,  5387.,  5387.,  5494.,  5687.,  6026.,  6457.,  6963.,  7538.,  8122.,  8732.,  9364.,  9964., 10593.]),
     np.array([ 6825.,  6825.,  6825.,  6825.,  6959.,  7204.,  7634.,  8179.,  8820.,  9549., 10289., 11062., 11862., 12622., 13419.])]
]

# 子图标题（可选）
titles = ['Triangle', 'Sine', 'Trapezoid', 'Exponential', 'Gaussian']

# 创建子图
fig, axs = plt.subplots(1, 5, figsize=(16/2.54, 5/2.54), sharey=True)

# 绘图循环
for idx, ax in enumerate(axs):
    for i in range(3):
        ax.plot(
            DEM_impact_rate[i],
            data_sets[idx][i],
            linestyle=linestyles[i],
            color=colors[i],
            label=labels[i] if idx == 0 else None  # 只在第一个子图加label
        )
        ax.scatter(
            DEM_impact_rate[i][::4],
            data_sets[idx][i][::4],
            marker=markers[i],
            color=colors[i],
            facecolors='white',         # 中空点
            #edgecolors='green',         # 绿色边框
            s=10,
            label=labels[i] if idx == 0 else None  # 只在第一个子图加label
        )
        # 添加参考线
        ax.plot(DEM_impact_rate[i], np.pi/4*EF_max[i], color='black', linestyle='-', label='Expected force' if idx == 0 else None)

    # ✅ 添加水平网格线
    ax.grid(axis='x', linestyle=':', linewidth=0.6, alpha=0.7)
    ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.7)
    # 设置子图标题（可选）
    ax.set_title(titles[idx], fontsize=8)
    ax.tick_params(direction='in', length=3)

# 统一x y 轴标签
axs[0].set_ylabel('Total impact force (kN)', fontsize=10)
axs[2].set_xlabel('Impact rate (s$^{-1}$)', fontsize=10)

# 图例放底部（全局）
fig.legend(
    loc='lower center',
    ncol=4,
    bbox_to_anchor=(0.5, -0.12),
    frameon=False
)

# 布局调整
#plt.tight_layout()
plt.subplots_adjust(bottom=0.28, top=0.9, left=0.10, right=0.98, hspace=0.2, wspace=0.2)

# 保存为高质量矢量图（推荐期刊投稿）
# plt.savefig('nature_ready_plot.pdf', dpi=600, bbox_inches='tight')

# 显示图形
plt.show()
