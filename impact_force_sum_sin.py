import numpy as np
import matplotlib.pyplot as plt

# 参数设置
impact_duration = 5e-4
delta_t = 3e-4
num_impacts = 5
time_total = (num_impacts - 1) * delta_t + impact_duration
omega_sine = np.pi / impact_duration
T_sine = 2 * impact_duration
num_points = 100
time_step = impact_duration / num_points

# 计算总时间和样本数
time_values = np.arange(0, time_total, time_step)
num_total_points = len(time_values)
num_samples_per_impact = round(delta_t / impact_duration * num_points)

# 预分配空间，初始化总和
total_waveform = np.zeros(num_total_points)

# 生成波形并叠加
for i in range(num_impacts):
    # 计算当前冲击的时间点
    impact_time_values = np.linspace(i * delta_t, i * delta_t + T_sine / 2, num_points)
    
    # 生成当前冲击的波形
    impact_wave = np.sin(omega_sine * (impact_time_values - i * delta_t))
    
    # 对当前波形进行零填充，使其对齐到总时间轴
    padded_impact_wave = np.pad(impact_wave, (i * num_samples_per_impact, num_total_points - num_points - i * num_samples_per_impact), 'constant', constant_values=0)
    
    # 将当前波形叠加到总和中
    total_waveform += padded_impact_wave
    
    # 可视化每个冲击的波形
    plt.plot(time_values, padded_impact_wave, '-')

# 可视化最终的总和波形
plt.plot(time_values, total_waveform, '-o', linewidth=3)

# 显示图形
plt.xlabel('Time (s)')
plt.ylabel('Function value')
plt.title('Function value over time')
plt.show()
