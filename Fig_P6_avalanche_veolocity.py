import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# ==========================================================
# ================= 1. 内置原始数据与深度解析 =================
# ==========================================================
raw_data = """试验编号  工况名称    I-I断面流深（初始时刻）   I-I初始速度（初始时刻）   I-I弗劳德数 II-II流深（冲击时刻）   II-II前缘冲击速度（初始时刻）   II-II弗劳德数
103 SL-M1500    0.047   2.67    3.97    0.037   1.60    2.68 
104 SL-M3000    0.060   3.952   5.20    0.057   1.84    2.49 
105 SL-M4500    0.100   4.13    4.21    0.097   3.16    3.27 
106 SP-M1500-D30    0.049   4.181   6.09    0.048   3.446   5.07 
107 SP-M3000-D30    0.065   4.486   5.67    0.051   3.158   4.51 
108 SP-M4500-D30    0.099   4.804   4.92    0.075   3.344   3.94 
109 SP-M1500-D50    0.059   5.264   6.99    0.042   3.581   5.63 
110 SP-M3000-D50    0.072   5.380   6.47    0.045   3.584   5.45 
111 SP-M4500-D50    0.085   5.819   6.44    0.053   3.824   5.36 
112 SP-M1500-D70    0.075   5.880   6.92    0.054   3.979   5.52 
113 SP-M3000-D70    0.076   5.949   6.96    0.054   4.086   5.67 
114 SP-M4500-D70    0.086   5.971   6.57    0.058   3.507   4.70 
115 SL-M1500-RA1    0.040   3.401   5.48    0.035   1.254   2.16 
116 SL-M3000-RA1    0.049   4.687   6.83    0.038   2.470   4.09 
117 SL-M4500-RA1    0.089   4.010   4.33    0.078   3.023   3.49 
118 SL-M1500-RA2    0.052   3.323   4.70    0.050   1.351   1.95 
119 SL-M3000-RA2    0.069   4.511   5.54    0.058   2.706   3.62 
120 SL-M4500-RA2    0.087   4.724   5.16    0.079   2.998   3.44 
121 SL-M1500-RA3    0.057   3.170   4.28    0.046   2.024   3.04 
122 SL-M3000-RA3    0.066   3.587   4.50    0.062   2.659   3.44 
123 SL-M4500-RA3    0.080   3.731   4.25    0.073   2.905   3.47 
"""
# 124-126行重要数据完好保留在注释矩阵中
#124 SL-M1500-RA4    0.046   3.432   5.16    0.046   0.094   0.14 
#125 SL-M3000-RA4    0.058   4.703   6.30    0.055   3.386   4.66 
#126 SL-M4500-RA4    0.089   4.658   5.04    0.078   3.483   4.02 

df = pd.read_csv(io.StringIO(raw_data), sep=r'\s+')
df.columns = ['ID', 'Condition', 'h1', 'v1', 'Fr1', 'h2', 'v2', 'Fr2']
df['ksi'] = (df['v1'] - df['v2']) / df['v1']

def extract_features_final(cond):
    parts = cond.split('-')
    raw_type = parts[0]
    mass = int(parts[1][1:])
    param = parts[2] if len(parts) > 2 else 'Smooth'
    if raw_type == 'SL' and 'RA' in param:
        final_type = 'RA-SL'
    else:
        final_type = raw_type
    return pd.Series([final_type, mass, param])

df[['Type', 'Mass', 'Param']] = df['Condition'].apply(extract_features_final)


# ==========================================================
# ================= 2. 🎛️ 核心参数分组字典 ====================
# ==========================================================
PANEL_TEXTS = {
    'ax_a': {'title': 'a  Mass effect on velocity transition',    'xlabel': 'Mass (g)', 'ylabel': 'Velocity $v$ (m s$^{-1}$)', 'leg_title': 'Material condition'},
    'ax_b': {'title': 'b  Particle size effect on transition',  'xlabel': 'Particle diameter', 'ylabel': 'Velocity $v$ (m s$^{-1}$)', 'leg_title': 'Mass (g)'},
    'ax_c': {'title': 'c  Slope roughness effect on transition', 'xlabel': 'Slope roughness level', 'ylabel': 'Velocity $v$ (m s$^{-1}$)', 'leg_title': 'Mass (g)'},
    'ax_d': {'title': 'd  Flow depth evolution',                 'xlabel': 'Impact depth $h_2$ (m)', 'ylabel': 'Initial depth $h_1$ (m)'},
    'ax_e': {'title': 'e  Froude number evolution',               'xlabel': 'Impact Froude number $Fr_2$', 'ylabel': 'Initial Froude number $Fr_1$'},
    'ax_f': {'title': 'f  Macro-aggregation of phase on loss',   'xlabel': 'Material type', 'ylabel': 'Momentum loss ratio $\\xi$'}
}

AXIS_LIMITS = {
    'ax_a': {'ylim': (0, 6.5)},
    'ax_b': {'ylim': (0, 6.5)},
    'ax_c': {'ylim': (0, 7.5)},
    'ax_d': {'ylim': (0, 0.12)},
    'ax_e': {'ylim_f': (1.5, 8.2)},
    'ax_f': {'ylim': (0, 1.0)},
}

LEGEND_CONFIGS = {
    'ax_a': {'loc': 'upper left', 'bbox': (0.15, 1.0)},
    'ax_b': {'loc': 'upper left'},
    'global_bottom': {'loc': 'lower center', 'bbox': (0.5, 0.01), 'ncol': 5, 'fontsize': 8, 'handletextpad': 0.2, 'columnspacing': 0.6} # 散点图大组图例移至大图最底部
}

COLOR_PALETTES = {
    'palette_a': {'D0 (SL)': '#F0E6E4', 'D30': '#E2BBB6', 'D50': '#CC8B83', 'D70': '#A44A3F'},
    'palette_b': {1500: '#A6CEE3', 3000: '#1F78B4', 4500: '#08306B'},
    'palette_c': {1500: '#E5E5E5', 3000: '#A6A6A6', 4500: '#595959'},
    'palette_g': {'SP': '#ABD9E9', 'SL': '#FDDBC7', 'RA-SL': '#FEE6CE'}
}

scatter_styles = {
    ('SP', 1500):    {'color': '#ABD9E9', 'size': 25, 'legend_size': 4.5}, 
    ('SP', 3000):    {'color': '#4575B4', 'size': 45, 'legend_size': 6.5}, 
    ('SP', 4500):    {'color': '#08519C', 'size': 65, 'legend_size': 8.5},
    ('SL', 1500):    {'color': '#FDDBC7', 'size': 25, 'legend_size': 4.5}, 
    ('SL', 3000):    {'color': '#F4A582', 'size': 45, 'legend_size': 6.5}, 
    ('SL', 4500):    {'color': '#B2182B', 'size': 65, 'legend_size': 8.5},
    ('RA-SL', 1500): {'color': '#FEE6CE', 'size': 25, 'legend_size': 4.5}, 
    ('RA-SL', 3000): {'color': '#FDAE6B', 'size': 45, 'legend_size': 6.5}, 
    ('RA-SL', 4500): {'color': '#E6550D', 'size': 65, 'legend_size': 8.5}  
}

bar_kws_uniform = {'edgecolor': 'black', 'linewidth': 0.9, 'width': 0.3, 'alpha': 0.92, 'dodge': True, 'errorbar': None}
markers_type = {'SP': 'o', 'SL': r'$\ast$', 'RA-SL': 'D'}


# ==========================================================
# ================= 3. 🎯 严格对齐 16cm 双栏排版规范 ============
# ==========================================================
plt.rcParams.update({
    'font.size': 8,               # 基础刻度/图例全部锁定 8pt
    'axes.labelsize': 10,         # 轴名严格锁定 10pt
    'axes.titlesize': 10,         # 子图标题严格锁定 10pt
    'axes.titleweight': 'bold',
    'xtick.labelsize': 8,         
    'ytick.labelsize': 8,         
    'legend.fontsize': 8,         
    'axes.linewidth': 1.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'mathtext.fontset': 'stixsans',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# 精准物理尺寸：总宽度设定为 16cm (16/2.54 英寸)
fig, axes = plt.subplots(2, 3, figsize=(16 / 2.54, 10.1 / 2.54), dpi=150)
ax_a, ax_b, ax_c = axes[0, 0], axes[0, 1], axes[0, 2]
ax_d, ax_e, ax_f = axes[1, 0], axes[1, 1], axes[1, 2] 

def apply_nature_spines_and_grid(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.35, zorder=0)

def draw_custom_scatter_engine(ax, x_col, y_col, txt_key):
    for (t_val, m_val), style in scatter_styles.items():
        sub_df = df[(df['Type'] == t_val) & (df['Mass'] == m_val)]
        if not sub_df.empty:
            ax.scatter(sub_df[x_col], sub_df[y_col], marker=markers_type[t_val], 
                       s=style['size'], c=style['color'], edgecolor='black', linewidths=0.7, alpha=0.9, zorder=3)
            
    min_val, max_val = min(df[x_col].min(), df[y_col].min()) * 0.92, max(df[x_col].max(), df[y_col].max()) * 1.08
    ax.plot([min_val, max_val], [min_val, max_val], color='#6b6b6b', linestyle='--', linewidth=0.5, zorder=1)
    
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel(PANEL_TEXTS[txt_key]['xlabel'])
    ax.set_ylabel(PANEL_TEXTS[txt_key]['ylabel'])
    #ax.set_title(PANEL_TEXTS[txt_key]['title'], loc='left', pad=10)
    apply_nature_spines_and_grid(ax)


# ==========================================================
# ================= 4. 子图 a: 质量效应 =====================
# ==========================================================
df_a_sp = df[df['Type'] == 'SP'].copy()
df_a_sl = df[(df['Type'] == 'SL') & (df['Param'] == 'Smooth')].copy()
df_a_sl['Param'] = 'D0 (SL)' 
df_a = pd.concat([df_a_sl, df_a_sp])
hue_order_a = ['D0 (SL)', 'D30', 'D50', 'D70']

df_a['Mass'] = pd.Categorical(df_a['Mass'].astype(str), categories=['1500', '3000', '4500'], ordered=True)

sns.barplot(data=df_a, x='Mass', y='v1', hue='Param', ax=ax_a, palette=COLOR_PALETTES['palette_a'], hue_order=hue_order_a, **bar_kws_uniform)
#ax_a.set_title(PANEL_TEXTS['ax_a']['title'], loc='left', pad=10)
ax_a.set_xlabel(PANEL_TEXTS['ax_a']['xlabel'])
ax_a.set_ylabel(PANEL_TEXTS['ax_a']['ylabel'])
ax_a.set_ylim(AXIS_LIMITS['ax_a']['ylim']) 
apply_nature_spines_and_grid(ax_a)

handles_a, labels_a = ax_a.get_legend_handles_labels()
# 🎯 截断设计：线段画11个像素自动急停无限留白，让向右实心箭头 '>' 绝不被实线刺穿
handles_a.append(
    Line2D([0], [0], color='#444444', linewidth=0.5, linestyle=(0, (11, 100)), 
           marker='v', markerfacecolor='#222222', markeredgecolor='#222222', markersize=5, label='Velocity Drop')
)
ax_a.legend(handles=handles_a, title=PANEL_TEXTS['ax_a']['leg_title'], frameon=False, 
            loc=LEGEND_CONFIGS['ax_a']['loc'], bbox_to_anchor=LEGEND_CONFIGS['ax_a']['bbox'], title_fontsize=8)

patches_a = ax_a.patches
idx_a = 0
for hue_val in hue_order_a:
    for x_val in ['1500', '3000', '4500']:
        p = patches_a[idx_a]
        bar_center = p.get_x() + p.get_width() / 2
        sub = df_a[(df_a['Mass'].astype(str) == x_val) & (df_a['Param'] == hue_val)]
        if not sub.empty:
            v1_val = sub['v1'].values[0]
            v2_val = sub['v2'].values[0]
            ax_a.plot([bar_center, bar_center], [v1_val, v2_val], color='#444444', linestyle='-', linewidth=1.0, alpha=0.85, zorder=4)
            # 🎯 图中落点同步完美联动修改为向右实心小箭头 '>'
            ax_a.scatter(bar_center, v2_val, marker='v', color='#222222', s=20, edgecolors='black', zorder=5)
        idx_a += 1


# ==========================================================
# ================= 5. 子图 b: 粒径效应 =====================
# ==========================================================
df_sp_only = df[df['Type'] == 'SP'].copy()
df_sl_control_b = df[(df['Type'] == 'SL') & (df['Param'] == 'Smooth')].copy()
df_sl_control_b['Param'] = 'D0 (SL)' 
df_b = pd.concat([df_sl_control_b, df_sp_only])
order_b = ['D0 (SL)', 'D30', 'D50', 'D70'] 

df_b['Param'] = pd.Categorical(df_b['Param'], categories=order_b, ordered=True)

sns.barplot(data=df_b, x='Param', y='v1', hue='Mass', ax=ax_b, palette=COLOR_PALETTES['palette_b'], hue_order=[1500, 3000, 4500], order=order_b, **bar_kws_uniform)
#ax_b.set_title(PANEL_TEXTS['ax_b']['title'], loc='left', pad=10)
ax_b.set_xlabel(PANEL_TEXTS['ax_b']['xlabel'])
ax_b.set_ylabel(PANEL_TEXTS['ax_b']['ylabel'])
ax_b.set_ylim(AXIS_LIMITS['ax_b']['ylim'])
apply_nature_spines_and_grid(ax_b)

handles_b, labels_b = ax_b.get_legend_handles_labels()
handles_b.append(
    Line2D([0], [0], color='#444444', linewidth=0.5, linestyle=(0, (11, 100)), 
           marker='v', markerfacecolor='#222222', markeredgecolor='#222222', markersize=5, label='Velocity Drop')
)
# 作为 b、c 两图共同的 Mass 组代言人
ax_b.legend(handles=handles_b, title=PANEL_TEXTS['ax_b']['leg_title'], frameon=False, loc=LEGEND_CONFIGS['ax_b']['loc'], title_fontsize=8)

patches_b = ax_b.patches
idx_b = 0
for hue_val in [1500, 3000, 4500]:
    for x_val in order_b:
        p = patches_b[idx_b]
        bar_center = p.get_x() + p.get_width() / 2
        sub = df_b[(df_b['Param'] == x_val) & (df_b['Mass'] == hue_val)]
        if not sub.empty:
            v1_val = sub['v1'].values[0]
            v2_val = sub['v2'].values[0]
            ax_b.plot([bar_center, bar_center], [v1_val, v2_val], color='#444444', linestyle='-', linewidth=1.0, alpha=0.85, zorder=4)
            ax_b.scatter(bar_center, v2_val, marker='v', color='#222222', s=20, edgecolors='black', zorder=5)
        idx_b += 1


# ==========================================================
# ================= 6. 子图 c: 粗糙度效应 (🎯合并同类项清理) ====
# ==========================================================
df_c = df[df['Type'].isin(['SL', 'RA-SL'])].copy()
df_c['Param'] = df_c['Param'].replace('Smooth', 'RA0 (SL)')
order_c = ['RA0 (SL)', 'RA1', 'RA2', 'RA3'] 

df_c['Param'] = pd.Categorical(df_c['Param'], categories=order_c, ordered=True)

sns.barplot(data=df_c, x='Param', y='v1', hue='Mass', ax=ax_c, palette=COLOR_PALETTES['palette_c'], hue_order=[1500, 3000, 4500], order=order_c, **bar_kws_uniform)
#ax_c.set_title(PANEL_TEXTS['ax_c']['title'], loc='left', pad=10)
ax_c.set_xlabel(PANEL_TEXTS['ax_c']['xlabel'])
ax_c.set_ylabel(PANEL_TEXTS['ax_c']['ylabel'])
ax_c.set_ylim(AXIS_LIMITS['ax_c']['ylim'])
apply_nature_spines_and_grid(ax_c)

# 🎯 顶刊排版优化：直接移去 c 的重复 Mass 图例，让其与 b 图共享同一含义，释放宝贵绘图空间
if ax_c.get_legend() is not None:
    ax_c.get_legend().remove()

patches_c = ax_c.patches
idx_c = 0
for hue_val in [1500, 3000, 4500]:
    for x_val in order_c:
        p = patches_c[idx_c]
        bar_center = p.get_x() + p.get_width() / 2
        sub = df_c[(df_c['Param'] == x_val) & (df_c['Mass'] == hue_val)]
        if not sub.empty:
            v1_val = sub['v1'].values[0]
            v2_val = sub['v2'].values[0]
            ax_c.plot([bar_center, bar_center], [v1_val, v2_val], color='#444444', linestyle='-', linewidth=1.0, alpha=0.85, zorder=4)
            ax_c.scatter(bar_center, v2_val, marker='v', color='#222222', s=20, edgecolors='black', zorder=5)
        idx_c += 1


# ==========================================================
# ================= 7. 下排基础散点图调用区 ===================
# ==========================================================
draw_custom_scatter_engine(ax_d, 'h2', 'h1', 'ax_d')
draw_custom_scatter_engine(ax_e, 'Fr2', 'Fr1', 'ax_e')


# ==========================================================
# ================= 8. 子图 f: 大类聚合箱线图 (🎯消除警告与冗余) =
# ==========================================================
# 🎯 完美去除警告：加入 hue='Type' 且 legend=False，严格契合新版 Seaborn 语法规范
sns.boxplot(data=df, x='Type', y='ksi', ax=ax_f, order=['SP', 'SL', 'RA-SL'], hue='Type', legend=False,
            palette=COLOR_PALETTES['palette_g'], width=0.4, fliersize=0, boxprops=dict(edgecolor='black', linewidth=1.0))

sns.stripplot(data=df, x='Type', y='ksi', ax=ax_f, order=['SP', 'SL', 'RA-SL'],
              color='#222222', size=3.5, jitter=0.15, alpha=0.7, linewidth=0.5, edgecolor='black', zorder=3)

ax_f.set_ylabel(PANEL_TEXTS['ax_f']['ylabel'])
ax_f.set_xlabel(PANEL_TEXTS['ax_f']['xlabel'])
#ax_f.set_title(PANEL_TEXTS['ax_f']['title'], loc='left', pad=10)
ax_f.set_ylim(AXIS_LIMITS['ax_f']['ylim'])
apply_nature_spines_and_grid(ax_f)

# 🎯 顶刊排版优化：横轴已自带高度清晰的 SP/SL/RA-SL 标签，图例在数学上完全冗余，不予生成以保纯净


# ==========================================================
# ================= 9. 全局联动图例生成系统 (🎯 底部多列优雅平铺) =
# ==========================================================
legend_elements_linked = []
for (t_val, m_val), style in scatter_styles.items():
    label_text = f"{t_val}-M{m_val}"
    leg_size = style.get('legend_size', style['size'] * 0.12)
    item = Line2D([0], [0], color='w', 
                  marker=markers_type[t_val], 
                  markerfacecolor=style['color'], 
                  markersize=leg_size, 
                  markeredgecolor='k', 
                  label=label_text)
    legend_elements_linked.append(item)

# 🎯 【终极美化】：利用 fig.legend 完美跨越边界，平铺在整幅大图的最下方中心，给下排散点图腾出 100% 画布
fig.legend(handles=legend_elements_linked, 
           loc=LEGEND_CONFIGS['global_bottom']['loc'], 
           bbox_to_anchor=LEGEND_CONFIGS['global_bottom']['bbox'], 
           ncol=LEGEND_CONFIGS['global_bottom']['ncol'], 
           handletextpad=LEGEND_CONFIGS['global_bottom']['handletextpad'], 
           columnspacing=LEGEND_CONFIGS['global_bottom']['columnspacing'], 
           fontsize=LEGEND_CONFIGS['global_bottom']['fontsize'],
           frameon=False)


# ==========================================================
# ================= 10. 整体排版缩进与渲染 ===================
# ==========================================================
# 🎯 关键留白：通过 rect=[0, 0.07, 1, 1] 专门为大图底部留出 7% 的黄金地带放置全局图例，绝不发生重叠
plt.tight_layout(rect=[0, 0.07, 1, 1], pad=1.2, w_pad=1.5, h_pad=2.0)
plt.show()