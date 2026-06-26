import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 解决中文字体缺失及负号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Segoe UI']
plt.rcParams['axes.unicode_minus'] = False

def process_sliding_window_data(file_path):
    """
    全自动滑动窗口数据处理程序
    输入:
        file_path: Excel文件路径
    """
    # 1. 读取原始捕捉数据
    df = pd.read_excel(file_path, sheet_name='TrackData')
    
    # 自动提取并排序所有已追踪的有效帧
    frame_indices = sorted(df['Frame_Index (绝对帧数)'].dropna().unique().astype(int))
    
    if len(frame_indices) < 2:
        print(f"错误：数据不足，无法构建滑动窗口。")
        return None
        
    print(f"自动遍历区间: Frame{frame_indices[0]} -> Frame{frame_indices[-1]}")
    all_process_data = []

    # 2. 自动生成全量滑动窗口配对
    for i in range(len(frame_indices) - 1):
        idx1, idx2 = frame_indices[i], frame_indices[i+1]
        
        row1 = df[df['Frame_Index (绝对帧数)'] == idx1].iloc[0]
        row2 = df[df['Frame_Index (绝对帧数)'] == idx2].iloc[0]
        
        delta_t = row2['Absolute_Time (绝对时间-秒)'] - row1['Absolute_Time (绝对时间-秒)']
        if delta_t <= 0: continue

        # 3. 动态识别所有捕捉点列并计算
        x_cols = [c for c in df.columns if 'Physical_X_P' in c]
        for x_col in x_cols:
            y_col = x_col.replace('X', 'Y')
            p_label = x_col.split(' ')[0].replace('Physical_X_', '')
            
            dx = row2[x_col] - row1[x_col]
            dy = row2[y_col] - row1[y_col]
            
            vx = dx / delta_t
            vy = dy / delta_t
            
            disp_true = np.sqrt(dx**2 + dy**2)
            vel_true = np.sqrt(vx**2 + vy**2)
            
            all_process_data.append({
                'Time_Interval': f'Frame{idx1}-Frame{idx2}',
                'Start_Frame': idx1,  # 辅助绘图颜色映射与排序
                'Point_Label': p_label,
                'Height_Y (mm)': row1[y_col],  
                'Delta_T (s)': delta_t,
                'Delta_X (mm)': dx,
                'Velocity_X (mm/s)': vx,
                'Delta_Y (mm)': dy,
                'Velocity_Y (mm/s)': vy,
                'True_Displacement (mm)': disp_true,
                'True_Velocity (mm/s)': vel_true
            })

    # 4. 构建数据表并写入新 Sheet
    process_df = pd.DataFrame(all_process_data)
    
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        process_df.drop(columns=['Start_Frame']).to_excel(writer, sheet_name='ProcessData', index=False)
        
    print(f"处理成功！（共 {process_df['Time_Interval'].nunique()} 个时间段）已写入 [ProcessData] 工作表。")
    
    # 5. 调用优化后的多维度辨识绘图机制
    plot_complete_profiles(process_df)
    
    return process_df

def plot_complete_profiles(process_df):
    """ 基于形态与色彩多维度解耦的全量数据绘图预览 """
    # 按时间先后排序时间段
    process_df = process_df.sort_values(by='Start_Frame')
    intervals = process_df['Time_Interval'].unique()
    
    # 创建科学画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True)
    fig.suptitle('湿雪崩全量运动学剖面时空演化图 (多维高区分度样式)', fontsize=14, fontweight='bold')

    # 色彩梯度配置
    min_f = process_df['Start_Frame'].min()
    max_f = process_df['Start_Frame'].max()
    if min_f == max_f: max_f += 1
    
    norm = matplotlib.colors.Normalize(vmin=min_f, vmax=max_f)
    mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap='coolwarm')

    # --- 核心优化：定义高可辨识度的标记和线型循环池 ---
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X', 'h', 'd']  # 圆、方、上三角、菱、下三角、五边、星、叉等
    linestyles = ['-', '--', '-.', ':']                          # 实线、虚线、点划线、细点线

    # 开始绘制全量线条
    for idx, interval in enumerate(intervals):
        data = process_df[process_df['Time_Interval'] == interval].sort_values(by='Height_Y (mm)')
        start_frame_num = data['Start_Frame'].iloc[0]
        
        # 1. 动态获取时间颜色
        line_color = mapper.to_rgba(start_frame_num)
        
        # 2. 动态循环获取独立的形状和线型
        current_marker = markers[idx % len(markers)]
        current_linestyle = linestyles[idx % len(linestyles)]
        
        # 左图：合速度剖面
        ax1.plot(data['True_Velocity (mm/s)'], data['Height_Y (mm)'], 
                 linestyle=current_linestyle, 
                 marker=current_marker, 
                 color=line_color, 
                 alpha=0.9, 
                 linewidth=2,
                 markersize=6, 
                 markeredgecolor='#222222',  # 深色边缘防止重叠粘连
                 markeredgewidth=0.7,
                 label=interval)
                 
        # 右图：合位移剖面
        ax2.plot(data['True_Displacement (mm)'], data['Height_Y (mm)'], 
                 linestyle=current_linestyle, 
                 marker=current_marker, 
                 color=line_color, 
                 alpha=0.9, 
                 linewidth=2,
                 markersize=6, 
                 markeredgecolor='#222222', 
                 markeredgewidth=0.7,
                 label=interval)

    # 计算自适应图例列数 (大于 6 个区间时自动开启双列排版，避免挤压数据)
    legend_cols = 2 if len(intervals) > 6 else 1

    # 左面板修饰
    ax1.set_title('Velocity Profile (合速度剖面)', fontsize=12, fontweight='semibold')
    ax1.set_xlabel('真实合速度 True Velocity (mm/s)', fontsize=10)
    ax1.set_ylabel('初始高度 Initial Height Y (mm)', fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.legend(title="时间区间 (不同线型及形状)", fontsize=9, loc='best', frameon=True, shadow=True, ncol=legend_cols)

    # 右面板修饰
    ax2.set_title('Displacement Profile (合位移剖面)', fontsize=12, fontweight='semibold')
    ax2.set_xlabel('真实合位移 True Displacement (mm)', fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.legend(title="时间区间 (不同线型及形状)", fontsize=9, loc='best', frameon=True, shadow=True, ncol=legend_cols)

    plt.tight_layout()
    plt.show()

# ==========================================
# 自动化执行一键测试
# ==========================================
if __name__ == "__main__":
    target_file = '111-front_tracked_data.xlsx' 
    
    try:
        process_sliding_window_data(target_file)
    except FileNotFoundError:
        print(f"提示：未找到文件 '{target_file}'，请检查路径。")