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

df = pd.read_csv(io.StringIO(raw_data), sep=r'\s+')
df.columns = ['ID', 'Condition', 'h1', 'v1', 'Fr1', 'h2', 'v2', 'Fr2']

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

def apply_nature_spines_and_grid(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.grid(True, linestyle=':', linewidth=0.6, alpha=0.35, zorder=0)

def draw_custom_scatter_engine(ax, x_col, y_col, txt_key):
    for (t_val, m_val), style in scatter_styles.items():
        sub_df = df[(df['Type'] == t_val) & (df['Mass'] == m_val)]
        if not sub_df.empty:
            ax.scatter(sub_df[x_col], sub_df[y_col], marker=markers_type[t_val], 
                       s=style['size'], c=style['color'], edgecolor='black', linewidths=0.85, alpha=0.9, zorder=3)
            
    min_val, max_val = min(df[x_col].min(), df[y_col].min()) * 0.92, max(df[x_col].max(), df[y_col].max()) * 1.08
    ax.plot([min_val, max_val], [min_val, max_val], color='#6b6b6b', linestyle='--', linewidth=1.4, zorder=1)
    
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel(PANEL_TEXTS[txt_key]['xlabel'])
    ax.set_ylabel(PANEL_TEXTS[txt_key]['ylabel'])
    ax.set_title(PANEL_TEXTS[txt_key]['title'], loc='left', pad=12)
    apply_nature_spines_and_grid(ax)

df[['Type', 'Mass', 'Param']] = df['Condition'].apply(extract_features_final)


# ==========================================================
# ================= 2. 🎛️ 核心参数分组字典 ====================
# ==========================================================
PANEL_TEXTS = {
    'ax_a': {'title': 'a  Mass effect on initial velocity', 'xlabel': 'Mass (g)', 'ylabel': 'Initial velocity (m s$^{-1}$)', 'leg_title': 'Material condition'},
    'ax_b': {'title': 'b  Particle size effect',            'xlabel': 'Particle diameter', 'ylabel': 'Initial velocity (m s$^{-1}$)', 'leg_title': 'Mass (g)'},
    'ax_c': {'title': 'c  Slope roughness effect (SL)',      'xlabel': 'Slope roughness level', 'ylabel': 'Initial velocity (m s$^{-1}$)', 'leg_title': 'Mass (g)'},
    'ax_d': {'title': 'd  Flow depth evolution',           'xlabel': 'Impact depth $h_2$ (m) [Sec II-II]', 'ylabel': 'Initial depth $h_1$ (m) [Sec I-I]'},
    'ax_e': {'title': 'e  Velocity evolution',             'xlabel': 'Impact velocity (m s$^{-1}$) [Sec II-II]', 'ylabel': 'Initial velocity $v_1$ (m s$^{-1}$) [Sec I-I]'},
    'ax_f': {'title': 'f  Froude number evolution',         'xlabel': 'Impact Froude number $Fr_2$ [Sec II-II]', 'ylabel': 'Initial Froude number $Fr_1$ [Sec I-I]'}
}

AXIS_LIMITS = {
    'ax_a': {'ylim': (0, 6.5)},
    'ax_b': {'ylim': (0, 6.5)},
    'ax_c': {'ylim': (0, 7.5)},
    'ax_f': {'ylim_f': (1.5, 8.2)} 
}

LEGEND_CONFIGS = {
    'ax_a': {'loc': 'upper center', 'bbox': (0.38, 1.0)},
    'ax_b': {'loc': 'upper left'},
    'ax_c': {'loc': 'upper left'},
    'ax_f_global': {'loc': 'lower right', 'ncol': 3, 'fontsize': 9.2, 'handletextpad': 0.2, 'columnspacing': 0.6} 
}

COLOR_PALETTES = {
    'palette_a': {'D0 (SL)': '#F0E6E4', 'D30': '#E2BBB6', 'D50': '#CC8B83', 'D70': '#A44A3F'},
    'palette_b': {1500:      '#A6CEE3',  3000: '#1F78B4',  4500: '#08306B'},
    'palette_c': {1500:      '#E5E5E5',  3000: '#A6A6A6',  4500: '#595959'}
}

# 【核心映射源】：此处的配置决定了下排所有散点在图中的【真实物理样式】
scatter_styles = {
    ('SP', 1500):    {'color': '#ABD9E9', 'size': 30, 'legend_size': 3}, ('SP', 3000):    {'color': '#4575B4', 'size': 60, 'legend_size': 6}, ('SP', 4500):    {'color': '#08519C', 'size': 90, 'legend_size': 9 },
    ('SL', 1500):    {'color': '#E5E5E5', 'size': 40, 'legend_size': 4}, ('SL', 3000):    {'color': '#A6A6A6', 'size': 80, 'legend_size': 8}, ('SL', 4500):    {'color': '#595959', 'size': 160, 'legend_size': 12},
    ('RA-SL', 1500): {'color': '#FEE6CE', 'size': 30, 'legend_size': 3}, ('RA-SL', 3000): {'color': '#FDAE6B', 'size': 60, 'legend_size': 6}, ('RA-SL', 4500): {'color': '#E6550D', 'size': 80, 'legend_size': 8 }  
}

bar_kws_uniform = {'edgecolor': 'black', 'linewidth': 1.1, 'width': 0.3, 'alpha': 0.92, 'dodge': True, 'errorbar': None}
markers_type = {'SP': 'o', 'SL': r'$\ast$', 'RA-SL': 'D'}


# ================= 3. 全局基本环境样式更新 =================
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12.5,
    'axes.titlesize': 13.5,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'axes.linewidth': 1.25,
    'xtick.major.width': 1.25,
    'ytick.major.width': 1.25,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'mathtext.fontset': 'stixsans',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.4))
ax_a, ax_b, ax_c = axes[0, 0], axes[0, 1], axes[0, 2]
ax_d, ax_e, ax_f = axes[1, 0], axes[1, 1], axes[1, 2]


# ==========================================================
# ================= 4. 子图 a: 质量效应 =====================
# ==========================================================
df_a_sp = df[df['Type'] == 'SP'].copy()
df_a_sl = df[(df['Type'] == 'SL') & (df['Param'] == 'Smooth')].copy()
df_a_sl['Param'] = 'D0 (SL)' 
df_a = pd.concat([df_a_sl, df_a_sp])

hue_order_a = ['D0 (SL)', 'D30', 'D50', 'D70']

sns.barplot(data=df_a, x='Mass', y='v1', hue='Param', ax=ax_a, 
            palette=COLOR_PALETTES['palette_a'], hue_order=hue_order_a, **bar_kws_uniform)

ax_a.set_title(PANEL_TEXTS['ax_a']['title'], loc='left', pad=12)
ax_a.set_xlabel(PANEL_TEXTS['ax_a']['xlabel'])
ax_a.set_ylabel(PANEL_TEXTS['ax_a']['ylabel'])
ax_a.set_ylim(AXIS_LIMITS['ax_a']['ylim']) 
ax_a.legend(title=PANEL_TEXTS['ax_a']['leg_title'], frameon=False, loc=LEGEND_CONFIGS['ax_a']['loc'], bbox_to_anchor=LEGEND_CONFIGS['ax_a']['bbox'])
apply_nature_spines_and_grid(ax_a)


# ==========================================================
# ================= 5. 子图 b: 粒径效应 =====================
# ==========================================================
df_sp_only = df[df['Type'] == 'SP'].copy()
df_sl_control_b = df[(df['Type'] == 'SL') & (df['Param'] == 'Smooth')].copy()
df_sl_control_b['Param'] = 'D0 (SL)' 
df_b = pd.concat([df_sl_control_b, df_sp_only])
order_b = ['D0 (SL)', 'D30', 'D50', 'D70'] 

sns.barplot(data=df_b, x='Param', y='v1', hue='Mass', ax=ax_b, palette=COLOR_PALETTES['palette_b'], hue_order=[1500, 3000, 4500], order=order_b, **bar_kws_uniform)

ax_b.set_title(PANEL_TEXTS['ax_b']['title'], loc='left', pad=12)
ax_b.set_xlabel(PANEL_TEXTS['ax_b']['xlabel'])
ax_b.set_ylabel(PANEL_TEXTS['ax_b']['ylabel'])
ax_b.set_ylim(AXIS_LIMITS['ax_b']['ylim'])
ax_b.legend(title=PANEL_TEXTS['ax_b']['leg_title'], frameon=False, loc=LEGEND_CONFIGS['ax_b']['loc'])
apply_nature_spines_and_grid(ax_b)


# ==========================================================
# ================= 6. 子图 c: 粗糙度效应 ===================
# ==========================================================
df_c = df[df['Type'].isin(['SL', 'RA-SL'])].copy()
df_c['Param'] = df_c['Param'].replace('Smooth', 'RA0 (SL)')
order_c = ['RA0 (SL)', 'RA1', 'RA2', 'RA3']

sns.barplot(data=df_c, x='Param', y='v1', hue='Mass', ax=ax_c, palette=COLOR_PALETTES['palette_c'], hue_order=[1500, 3000, 4500], order=order_c, **bar_kws_uniform)

ax_c.set_title(PANEL_TEXTS['ax_c']['title'], loc='left', pad=12)
ax_c.set_xlabel(PANEL_TEXTS['ax_c']['xlabel'])
ax_c.set_ylabel(PANEL_TEXTS['ax_c']['ylabel'])
ax_c.set_ylim(AXIS_LIMITS['ax_c']['ylim'])
ax_c.legend(title=PANEL_TEXTS['ax_c']['leg_title'], frameon=False, loc=LEGEND_CONFIGS['ax_c']['loc'])
apply_nature_spines_and_grid(ax_c)


# ==========================================================
# ================= 7. 下排散点图调用区 =====================
# ==========================================================
draw_custom_scatter_engine(ax_d, 'h2', 'h1', 'ax_d')
draw_custom_scatter_engine(ax_e, 'v2', 'v1', 'ax_e')
draw_custom_scatter_engine(ax_f, 'Fr2', 'Fr1', 'ax_f')


# ==========================================================
# ================= 8. 【物理联动】图例自动生成闭环引擎 ============
# ==========================================================
# 核心重构：彻底摒弃人工配置，直接遍历顶层核心控制字典，实现真正的 closed-loop 同步
legend_elements_linked = []

for (t_val, m_val), style in scatter_styles.items():
    # 动态拼接工况名称文本（如 SP-M1500）
    label_text = f"{t_val}-M{m_val}"
    
    # 动态映射底层完全一致的标记、缩放尺度和配色参数
    item = Line2D([0], [0], color='w', 
                  marker=markers_type[t_val], 
                  markerfacecolor=style['color'], 
                  markersize=style['legend_size'], # 缩放图例点大小使其在框内美观
                  markeredgecolor='k', 
                  label=label_text)
    
    legend_elements_linked.append(item)

# 将自动生成的联动图例按 3 列精准嵌入子图 f 的安全留白区
ax_f.set_ylim(AXIS_LIMITS['ax_f']['ylim_f'])
ax_f.legend(handles=legend_elements_linked, frameon=False, 
            loc=LEGEND_CONFIGS['ax_f_global']['loc'], 
            ncol=LEGEND_CONFIGS['ax_f_global']['ncol'], 
            handletextpad=LEGEND_CONFIGS['ax_f_global']['handletextpad'], 
            columnspacing=LEGEND_CONFIGS['ax_f_global']['columnspacing'], 
            fontsize=LEGEND_CONFIGS['ax_f_global']['fontsize'])


# ==========================================================
# ================= 9. 整体排版渲染 =========================
# ==========================================================
plt.tight_layout(pad=2.8, w_pad=3.2, h_pad=3.8)
plt.show()