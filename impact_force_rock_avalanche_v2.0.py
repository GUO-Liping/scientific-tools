import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from scipy.integrate import quad
import math

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用 SimHei 字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def adjust_radius(radius_min, radius_max):
    """如果半径上下限相等，则微调radius_max，防止除零错误。"""
    if radius_max == radius_min:
        radius_min -= 1e-5
        radius_max += 1e-5
    return radius_min, radius_max

def compute_impact_duration(DEM_density, DEM_modulus, DEM_miu, DEM_radius, DEM_velocity, Pier_modulus, Pier_miu):
    R1 = DEM_radius
    R2 = float('inf')
    E1 = DEM_modulus
    E2 = Pier_modulus
    miu1 = DEM_miu
    miu2 = Pier_miu

    # Hertz弹性碰撞时长
    velocity_relative = DEM_velocity
    m1 = DEM_density*4*np.pi*R1**3/3
    k1 = (1-miu1**2)/(np.pi*E1)
    k2 = (1-miu2**2)/(np.pi*E2)
    n0 = 4/(3*np.pi*(k1+k2)) * np.sqrt(R1)
    n1 = 1 / m1

    alpha1 = (5*velocity_relative**2/(4*n0*n1))**(2/5)
    impact_duration_DEM_plane = 2.943 * alpha1/velocity_relative

    impact_duration_DEM_DEM = 2.943 * (5*np.sqrt(2)/4 * np.pi*DEM_density * (1-DEM_miu**2)/DEM_modulus)**(2/5) * DEM_radius / (velocity_relative**(1/5)) # s
    # impact_duration_DEM_plane2 = 2.943/(velocity_relative**0.2) * (15*np.pi*m1*(k1+k2)/(16*np.sqrt(R1)))**0.4
    return impact_duration_DEM_plane,impact_duration_DEM_DEM


def compute_elastoplastic_impact_duration(DEM_density, DEM_modulus, DEM_miu, DEM_radius, DEM_velocity, Pier_modulus, Pier_miu, sigma_y):
    # Johnson弹塑性碰撞时长
    sigma_yd = 1.0 * sigma_y  # pd/pm = 1.28 for steel material
    p_d = 3.0 * sigma_yd
    modulus_star = 1/((1-DEM_miu**2)/DEM_modulus + (1-Pier_miu**2)/Pier_modulus)
    radius_star = DEM_radius
    mass_star = DEM_density * 4/3 * np.pi * radius_star**3
    velocity_relative = DEM_velocity

    coeff1 = 3 * np.pi**(5/4) * 4**(3/4) / 10
    coeff2 = p_d / modulus_star
    coeff3 = (1/2 * mass_star * velocity_relative**2 / (p_d * radius_star**3))**(-1/4)
    coeff_re = np.sqrt(coeff1 * coeff2 * coeff3)
    if coeff_re > 1:
        coeff_re = 1

    # 当速度比较小时，采用下式coeff_re2计算的恢复系数大于1，且与coeff_re相差较大，这说明使用p_d≈3.0σy不准确，故不采用
    # coeff_re2 = 3.7432822830305064 * np.sqrt(sigma_yd/modulus_star) * ((1/2*mass_star*velocity_relative**2)/(sigma_yd*radius_star**3))**(-1/8)

    def integrand(x):
        return 1 / np.sqrt(1 - x**(5/2))
    
    int_result, int_error = quad(integrand, 0, 1)
    delta_z_star = ((15*mass_star*velocity_relative**2) / (16*np.sqrt(radius_star)*modulus_star))**(2/5)
    t_elastic_origin = 2 * (delta_z_star/velocity_relative) * int_result

    # eq.(11.24), 本程序采用该式  
    t_elastic = 2.868265699194853 * (mass_star**2 / (radius_star*modulus_star**2 * (coeff_re*velocity_relative)))**(1/5)
    t_plastic = np.sqrt((np.pi * mass_star) / (8*radius_star*p_d))
    
    t_elastic_eq11_47 = 1.2 * coeff_re * t_plastic  # eq.(11.47)

    t_elastoplastic = t_elastic + t_plastic
    return t_elastoplastic

def compute_effective_flow_rate(pier_width, DEM_depth, radius_max, DEM_velocity):
    """计算桥墩有效宽度，根据截面形状调整。"""
    pier_width_effect = pier_width + 4 * radius_max
    DEM_depth_effect = DEM_depth
    effective_flow_rate = pier_width_effect * DEM_depth_effect * DEM_velocity
    
    return effective_flow_rate

def compute_Hertz_contact_forces(radius_min, radius_max, modulus_eq, dem_density, dem_velocity):
    """计算Hertz弹性接触理论中的接触力。"""
    radius_equ = (1/3 * (radius_max**2 + radius_max*radius_min + radius_min**2)) ** 0.5
    factor = (4/3) * modulus_eq**(2/5) * (5*np.pi/4)**(3/5) * dem_density**(3/5) * dem_velocity**(6/5)
    force_min = factor * radius_min**2
    force_max = factor * radius_max**2
    force_equ = factor * radius_equ**2

    coeff_force = 4/9 * (radius_max**2 + radius_max*radius_min + radius_min**2)
    force_average = coeff_force * (5*np.pi/4)**(3/5) * modulus_eq**(2/5) * dem_density**(3/5) * dem_velocity**(6/5)

    return force_min, force_max, force_equ, force_average

def compute_Thornton_contact_force(radius_min, radius_max, modulus_eq, dem_density, dem_velocity, sigma_y):
    """计算Thornton弹性/理想塑性-接触力的期望。"""
    velocity_y = np.sqrt(np.pi**4/40) * (sigma_y**5 / (dem_density * modulus_eq**4))**0.5  # np.sqrt(np.pi**4/40)=1.560521475613219791
    Fy_r_max = sigma_y**3 * np.pi**3 * radius_max**2 / (6 * modulus_eq**2)
    Fy_r_min = sigma_y**3 * np.pi**3 * radius_min**2 / (6 * modulus_eq**2)

    if dem_velocity <= velocity_y:
        print('\tdem_velocity <= velocity_y !', 'dem_velocity=', np.round(dem_velocity,3), 'velocity_y=',np.round(velocity_y,6))
        F_single_min = (4/3) * modulus_eq**0.4 * (5 * dem_density * np.pi * dem_velocity**2 / 4)**0.6 * radius_min**2
        F_single_max = (4/3) * modulus_eq**0.4 * (5 * dem_density * np.pi * dem_velocity**2 / 4)**0.6 * radius_max**2
        E_Fmax = (4/9) * (radius_max**2 + radius_max*radius_min + radius_min**2) * modulus_eq**0.4 * (5 * dem_density * np.pi * dem_velocity**2 / 4)**0.6
    else:
        print('\tdem_velocity > velocity_y !', 'dem_velocity=', np.round(dem_velocity,3), 'velocity_y=',np.round(velocity_y,6))
        F_single_min = np.sqrt(Fy_r_min**2 +  np.pi * sigma_y * dem_density * (4/3) * np.pi * radius_min**3 * (dem_velocity**2 - velocity_y**2)*radius_min)
        F_single_max = np.sqrt(Fy_r_max**2 +  np.pi * sigma_y * dem_density * (4/3) * np.pi * radius_max**3 * (dem_velocity**2 - velocity_y**2)*radius_max)
        E_Fmax = 1/3 * (radius_max**2 + radius_max*radius_min + radius_min**2) * np.sqrt(sigma_y**6*np.pi**6/(36*modulus_eq**4) + 4/3*np.pi**2 * sigma_y * dem_density * (dem_velocity**2 - velocity_y**2))

    return velocity_y, F_single_min, F_single_max, E_Fmax


def generate_waveform(wave_type, num_points_wave, amplitude=1.0):
    """生成基础单峰波形"""
    x = np.linspace(0, 1, num_points_wave)

    if wave_type == 'sine':
        return amplitude * np.sin(np.pi * x)
    
    elif wave_type == 'triangle':
        half = round(num_points_wave / 2)
        rise = np.linspace(0, amplitude, half)
        fall = np.linspace(amplitude, 0, half)
        return np.concatenate((rise, fall[1:],np.array([0])),axis=0)
    
    elif wave_type == 'square':
        return np.full(num_points_wave, amplitude)
    
    elif wave_type == 'sawtooth':
        return amplitude * x

    elif wave_type == 'gaussian':
        return amplitude * np.exp(-20 * (x - 0.5) ** 2)
    
    else:
        raise ValueError(f"Unsupported wave type: {wave_type}")

def addition_waveforms(wave_type, wave_duration, delta_t, amplitude=1.0, num_steps=1000):
    """生成波形序列并求叠加"""
    num_waves = int(wave_duration / delta_t) + 1  # 计算冲击时长内的重叠波形数量
    time_step = wave_duration / num_steps
    num_shift_steps = round(delta_t / time_step)  

    shift_steps = max(1, num_shift_steps)
    total_points = (num_waves - 1) * shift_steps + num_steps + 1
    time_values = np.arange(total_points) * time_step

    base_wave = generate_waveform(wave_type, num_steps+1, amplitude)
    total_wave = np.zeros(total_points)

    for i in range(num_waves):
        wave = np.zeros(total_points)
        start = i * shift_steps
        end = min(start + num_steps, total_points)
        length = end - start
        wave[start:end] = base_wave[:length]
        plt.plot(time_values, wave, '-o', label=f'Wave {i+1}')
        total_wave += wave
    plt.plot(time_values, total_wave, '-o', label='Total Wave')
    plt.show()

    return np.max(total_wave)

def compute_gamma_s(pier_shape):
    if Pier_shape == 'round':
        gamma_s = np.pi/4          # 0.65
    elif Pier_shape =='square':
        gamma_s = 1
    else:
        raise ValueError('Shape Of Section Not Found!')
    return gamma_s

def compute_delta_t(DEM_flow_rate, flow_time, radius_max, radius_min, ratio_solid):
    flow_time = 1  # s
    volume_total = DEM_flow_rate * flow_time
    radius_avg = (radius_max + radius_min)/2
    number_of_DEM = int(ratio_solid * volume_total / (4/3 * np.pi * (radius_avg**3 )))
    print('\tnumber_of_DEM =', number_of_DEM)
    delta_t = flow_time / number_of_DEM

    return delta_t



if __name__ == '__main__':
    # 参数定义

    # This study
    DEM_Volumn = 1000      # 碎屑流方量：m^3
    DEM_depth = (3.8 + (8.25-3.8)/(4000-1000) * (DEM_Volumn-1000))      
    DEM_velocity = (12.8 + (10.8-12.8)/(16000-1000) * (DEM_Volumn-1000))      # m/s
    DEM_density = 2550      # kg/m3  花岗岩密度2500kg/m3
    DEM_modulus = 50e9      # Pa   花岗岩弹性模量50-100GPa
    DEM_miu = 0.2          # Poisson's ratio  花岗岩泊松比0.1-0.3
    radius_min = 0.6  # m
    radius_max = 1.2  # m
    ratio_solid = (0.55 + (0.55-0.55)/(16000-1000) * (DEM_Volumn-1000)) # 固相体积分数np.pi/6.0
    impact_angle_deg = 90   # 冲击角度 °
    
    # Pier_shape = 'square'
    Pier_shape = 'round'
    Pier_width = 2.2        # m
    Pier_modulus = 31e9    # Pa 混凝土弹性模量:31GPa
    Pier_miu = 0.2          # 混凝土Poisson's ratio ：0.2
    sigma_y = 30e6          # Pa C30混凝土强度:30 MPa

    # 调整半径
    radius_min, radius_max = adjust_radius(radius_min, radius_max)
    DEM_flow_rate = compute_effective_flow_rate(Pier_width, DEM_depth, radius_max, DEM_velocity)    # m^3/s

    # 计算冲击时间
    impact_duration_elastic = compute_impact_duration(DEM_density, DEM_modulus, DEM_miu, radius_max, DEM_velocity, Pier_modulus, Pier_miu)[0]
    impact_duration_elastoplastic = compute_elastoplastic_impact_duration(DEM_density, DEM_modulus, DEM_miu, radius_max, DEM_velocity, Pier_modulus, Pier_miu, sigma_y)
    print('impact_duration_elastic =', np.round(impact_duration_elastic,9), 's')
    print('impact_duration_elastoplastic =', np.round(impact_duration_elastoplastic,9), 's')

    # 计算等效弹性模量
    modulus_equ = 1 / ((1-Pier_miu**2) / Pier_modulus + (1-DEM_miu**2) / DEM_modulus)

    # Hertz弹性接触理论计算冲击力（接触力）
    force_min_e, force_max_e, force_equ_e, E_Fmax_e = compute_Hertz_contact_forces(radius_min, radius_max, modulus_equ, DEM_density, DEM_velocity)
    print('[Hertz Elastic Theory]: ', '\n\tF_min_e=', np.round(force_min_e,3), 'F_max_e=',np.round(force_max_e,3), 'F_equ_e=', np.round(force_equ_e,3),'E_Fmax_e=',np.round(E_Fmax_e,3),'N')

    wave_types = ['sine', 'triangle', 'square', 'sawtooth', 'gaussian']
    wave_type = wave_types[1]

    delta_te = compute_delta_t(DEM_flow_rate, impact_duration_elastic, radius_max, radius_min, ratio_solid)
    gamma_te = addition_waveforms(wave_type,impact_duration_elastoplastic,delta_te)
    angle_e = np.sin(np.radians(impact_angle_deg))
    gamma_se = compute_gamma_s(Pier_shape)

    total_force_e = gamma_te * gamma_se * angle_e * E_Fmax_e
    print('gamma_te=', gamma_te, 'gamma_se=', gamma_se, 'angle_e=', angle_e, 'E_Fmax_e=', E_Fmax_e,'total_force_e=', total_force_e)

    # Thornton弹性-理想塑性接触理论计算冲击力（接触力）
    print('[Thornton Elasto-Plastic Theory]: ')
    v_y, F_min_p, F_max_p, E_Fmax_p = compute_Thornton_contact_force(radius_min, radius_max, modulus_equ, DEM_density, DEM_velocity, sigma_y)
    print(f'\tF_min_p = {np.round(F_min_p,3)}, F_max_p = {np.round(F_max_p,3)}, force_average_p = {np.round(E_Fmax_p,3)}', 'N')

    # 碰撞过程时间离散性和总冲击力
    delta_tp = compute_delta_t(DEM_flow_rate, impact_duration_elastoplastic, radius_max, radius_min, ratio_solid)
    gamma_tp = addition_waveforms(wave_type,impact_duration_elastoplastic,delta_tp)
    angle_p = np.sin(np.radians(impact_angle_deg))
    gamma_sp = compute_gamma_s(Pier_shape)

    total_force_p = gamma_tp * gamma_sp * angle_p * E_Fmax_p
    print('gamma_tp=', gamma_tp, 'gamma_s=', gamma_sp, 'angle_p=', angle_p, 'E_Fmax_p=', E_Fmax_p,'total_force_p=', total_force_p)

    '''     
    # Wang et al. 2025 参数
    DEM_density = 2550      # kg/m3
    DEM_depth = 0.05        # m
    DEM_modulus = 60e9      # Pa
    DEM_miu = 0.25          # Poisson’s ratio
    DEM_velocity = 1.6      # m/s
    radius_min = 4.0e-3     # m
    radius_max = 4.0e-3     # m
    ratio_solid = 0.45      # 固相体积分数
    impact_angle_deg = 72   # 冲击角度 °

    #Pier_shape = 'square'
    Pier_shape = 'round'
    Pier_width = 0.1
    Pier_modulus = 3.2e9    # Pa
    Pier_miu = 0.35         # Poisson’s ratio
    sigma_y = 30e6          # Pa
    
    
    # Choi et al. 2020参数
    DEM_velocity = 1.8      # m/s
    DEM_depth = 0.031       # m
    DEM_density = 2500      # kg/m3  玻璃密度2500kg/m3
    DEM_modulus = 55e9      # Pa  玻璃弹性模量55GPa
    DEM_miu = 0.25          # Poisson's ratio  玻璃泊松比0.25
    radius_min = 10.0e-3/2   # m
    radius_max = 10.0e-3/2   # m
    ratio_solid = np.pi/6.0 # 固相体积分数np.pi/6.0
    impact_angle_deg = 90   # 冲击角度 °

    Pier_shape = 'square'
    # Pier_shape = 'round'
    Pier_width = 0.2        # m
    Pier_modulus = 3.0e9    # Pa PMMA:3.0GPa
    Pier_miu = 0.3          # Poisson's ratio 
    sigma_y = 50e6          # Pa PMMA:50 - 77 MPa

    # Barbara et al. 2010 参数
    DEM_density = 1530      # kg/m3
    DEM_depth = 0.025       # m
    DEM_modulus = 30e9      # Pa
    DEM_miu = 0.30          # Poisson's ratio
    DEM_velocity = 2.9      # m/s
    radius_min = 0.3e-3/2   # m
    radius_max = 6.0e-3/2   # m
    ratio_solid = 0.4 #np.pi/6.0 # 固相体积分数
    impact_angle_deg = 51   # 冲击角度 °

    #Pier_shape = 'square'
    Pier_shape = 'round'
    Pier_width = 0.025      # m
    Pier_modulus = 4.0e9    # Pa PVC:2.4GPa - 4.1GPa
    Pier_miu = 0.3          # Poisson's ratio PVC:0.38
    sigma_y = 40e6          # Pa PVC:40 - 44 MPa
    

    # Zhong et al. 2022 参数
    DEM_density = 1550      # kg/m3
    DEM_depth = 0.8         # m
    DEM_modulus = 30e9      # Pa
    DEM_miu = 0.30          # Poisson's ratio
    DEM_velocity = 28.23    # m/s
    radius_min = 0.1        # m
    radius_max = 0.2        # m
    ratio_solid = np.pi/6.0 # 固相体积分数
    impact_angle_deg = 60   # 冲击角度 °

    Pier_shape = 'round'
    Pier_width = 1.8        # m
    Pier_modulus = 32.5e9   # Pa 30GPa - 32.5GPa
    Pier_miu = 0.3          # Poisson's ratio
    sigma_y = 40e6          # Pa

    # Wang et al. 2025 参数
    DEM_density = 2550      # kg/m3
    DEM_depth = 0.05        # m
    DEM_modulus = 60e9      # Pa
    DEM_miu = 0.25          # Poisson’s ratio
    DEM_velocity = 1.6      # m/s
    radius_min = 4.0e-3     # m
    radius_max = 4.0e-3     # m
    ratio_solid = 0.45      # 固相体积分数
    impact_angle_deg = 72   # 冲击角度 °

    #Pier_shape = 'square'
    Pier_shape = 'round'
    Pier_width = 0.1
    Pier_modulus = 3.2e9    # Pa
    Pier_miu = 0.35         # Poisson’s ratio
    sigma_y = 30e6          # Pa
    '''