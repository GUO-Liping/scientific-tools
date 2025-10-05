import numpy as np
import matplotlib.pyplot as plt

def generate_waveform(wave_type, num_points=100, amplitude=1.0):
    """生成基础单峰波形"""
    x = np.linspace(0, 1, num_points)

    if wave_type == 'sine':
        return amplitude * np.sin(np.pi * x)
    
    elif wave_type == 'triangle':
        half = round(num_points / 2)
        rise = np.linspace(0, amplitude, half)
        fall = np.linspace(amplitude, 0, half)
        return np.concatenate((rise, fall[1:],np.array([0])),axis=0)
    
    elif wave_type == 'square':
        return np.full(num_points, amplitude)

    elif wave_type == 'trapezoidal':
        num_rise = round(num_points / 8)
        rise = np.linspace(0, amplitude, num_rise)
        const = np.full(num_points - 2*num_rise, amplitude)
        fall = np.linspace(amplitude, 0, num_rise)
        return np.concatenate((rise, const, fall),axis=0)

    
    elif wave_type in ('exponential', 'shock'):
        b = -np.log(0.02)/1.0
        return amplitude * np.exp(-b*x)
    
    elif wave_type == 'sawtooth':
        return amplitude * x

    elif wave_type == 'gaussian':
        return amplitude * np.exp(-20 * (x - 0.5) ** 2)
    
    else:
        raise ValueError(f"Unsupported wave type: {wave_type}")

def generate_wave_sequence(wave_type='sine', num_waves=5, 
                           wave_duration=0.0005, delta_t=0.0003, 
                           amplitude=1.0, num_points=100):
    """生成波形序列并求叠加"""
    time_step = wave_duration / num_points
    shift_points = max(1, int(round(delta_t / time_step)))
    total_points = (num_waves - 1) * shift_points + num_points
    time_values = np.arange(total_points) * time_step

    base_wave = generate_waveform(wave_type, num_points, amplitude)
    total_wave = np.zeros(total_points)
    waveforms = []

    for i in range(num_waves):
        wave = np.zeros(total_points)
        start = i * shift_points
        end = min(start + num_points, total_points)
        length = end - start
        wave[start:end] = base_wave[:length]
        waveforms.append(wave)
        total_wave += wave

    return waveforms, total_wave, time_values

def plot_waveforms(waveforms, total_wave, time_values, wave_type):
    """绘制多个波形及其叠加图"""
    plt.figure(figsize=(10, 6))
    
    # 绘制前几条单个波形
    max_show = len(waveforms)
    for i in range(max_show):
        plt.plot(time_values, waveforms[i], alpha=0.6, label=f'Wave {i+1}')
    
    # 绘制叠加波形
    plt.plot(time_values, total_wave, 'r-', linewidth=2, label='Sum')
    
    plt.title(f'{wave_type.capitalize()} Wave Sequence ({len(waveforms)} waves)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ======================== 主程序 ========================
if __name__ == "__main__":
    wave_types = ['sine', 'triangle', 'square', 'sawtooth', 'gaussian', 'exponential', 'trapezoidal']
    t_contact = 0.0003  # 每个波的持续时间
    delta_t_DEMs = 0.0001       # 相邻波形间隔
    amplitude = 1.0
    num_waves = np.maximum(np.ceil(t_contact / delta_t_DEMs), 1).astype(int)


    for wave_type in wave_types:
        waveforms, total_wave, time_values = generate_wave_sequence(
            wave_type=wave_type,
            num_waves=num_waves,
            wave_duration=t_contact,
            delta_t=delta_t_DEMs,
            amplitude=amplitude
        )
        plot_waveforms(waveforms, total_wave, time_values, wave_type)

    wave_type = 'sine'
    if wave_type == 'sine':
        max_coefficient = np.max(generate_wave_sequence(
            wave_type=wave_type,
            num_waves=num_waves,
            wave_duration=t_contact,
            delta_t=delta_t_DEMs,
            amplitude=amplitude
        )[1])
    print('max_coefficient=', max_coefficient)
        