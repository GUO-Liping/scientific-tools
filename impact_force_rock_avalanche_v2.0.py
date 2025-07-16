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
    DEM_depth_effect = DEM_depth + 2 * radius_max
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
        print('\tdem_velocity <= velocity_y !', 'dem_velocity=', np.round(dem_velocity,3), 'velocity_y=',np.round(velocity_y,3))
        F_single_min = (4/3) * modulus_eq**0.4 * (5 * dem_density * np.pi * dem_velocity**2 / 4)**0.6 * radius_min**2
        F_single_max = (4/3) * modulus_eq**0.4 * (5 * dem_density * np.pi * dem_velocity**2 / 4)**0.6 * radius_max**2
        E_Fmax = (4/9) * (radius_max**2 + radius_max*radius_min + radius_min**2) * modulus_eq**0.4 * (5 * dem_density * np.pi * dem_velocity**2 / 4)**0.6
    else:
        print('\tdem_velocity > velocity_y !', 'dem_velocity=', np.round(dem_velocity,3), 'velocity_y=',np.round(velocity_y,3))
        F_single_min = np.sqrt(Fy_r_min**2 +  np.pi * sigma_y * dem_density * (4/3) * np.pi * radius_min**3 * (dem_velocity**2 - velocity_y**2)*radius_min)
        F_single_max = np.sqrt(Fy_r_max**2 +  np.pi * sigma_y * dem_density * (4/3) * np.pi * radius_max**3 * (dem_velocity**2 - velocity_y**2)*radius_max)
        E_Fmax = 1/3 * (radius_max**2 + radius_max*radius_min + radius_min**2) * np.sqrt(sigma_y**6*np.pi**6/(36*modulus_eq**4) + 4/3*np.pi**2 * sigma_y * dem_density * (dem_velocity**2 - velocity_y**2))

    return velocity_y, F_single_min, F_single_max, E_Fmax

def compute_total_impact_force_triangle(DEM_flow_rate, Pier_shape, ratio_solid, radius_min, radius_max, impact_angle_deg, impact_duration, E_Fmax):
    """根据三角形脉冲计算碰撞过程中总冲击力和相关时间离散参数。"""
    flow_time = 1  # s
    volume_total = DEM_flow_rate * flow_time
    radius_avg = (radius_max + radius_min) / 2
    number_of_DEM = math.ceil(ratio_solid * volume_total / (4/3 * np.pi * (radius_avg**3 )))
    t_per_DEM = flow_time / number_of_DEM
    num_pieces = max(math.ceil(impact_duration/t_per_DEM),1)

    array_size = 100

    length = array_size // 2
    if array_size % 2 == 0:
        front = np.linspace(0, 1, length)
        end = np.linspace(1, 0, length)
        pulse_array = np.concatenate([front, end[1:], np.array([0])])
    else:
        front = np.linspace(0, 1, length + 1)
        end = np.linspace(1, 0, length + 1)
        pulse_array = np.concatenate([front, end[1:]])  # 例如 array_size = 101
    pulse_array = pulse_array[:array_size]  # 最终强制长度一致（保险）

    multi_pulse_arrays = []
    length_add = round(length * 2*t_per_DEM/impact_duration)
    # 对基础数组进行平移，生成其余数组
    for i in range(num_pieces):
        shifted_array = np.concatenate([np.zeros(i*length_add), pulse_array, np.zeros((num_pieces - i-1)*length_add)])
        multi_pulse_arrays.append(shifted_array)
    
    time_pulse = np.linspace(0, (len(shifted_array) - 1) * t_per_DEM, len(shifted_array))

    # 将所有数组进行求和
    sum_array = np.sum(multi_pulse_arrays, axis=0)

    # 绘制所有数组和它们的和
    plt.figure(figsize=(10, 6))
    for i, array in enumerate(multi_pulse_arrays):
        plt.plot(time_pulse, array, label=f'Array {i+1}')

    plt.plot(time_pulse,sum_array, label='Sum of Arrays', color='red', linewidth=2)
    plt.legend()
    plt.title('三角脉冲波形之和 vs 时间')
    plt.xlabel('时间(s)')
    plt.ylabel('脉冲强度')
    plt.grid(True)
    plt.show()

    angle_impact = np.sin(np.radians(impact_angle_deg))

    if Pier_shape == 'round':
        gamma_space = np.pi/4
    elif Pier_shape =='square':
        gamma_space = 1
    else:    
        raise ValueError('Shape Of Section Not Found!')
    
    if t_per_DEM >= (impact_duration/2):
        total_force = gamma_space * angle_impact * E_Fmax
    else:
        total_force = gamma_space * angle_impact * E_Fmax * np.max(sum_array)
    
    return num_pieces, t_per_DEM, total_force

def compute_total_impact_force_sine(DEM_flow_rate, Pier_shape, ratio_solid, radius_min, radius_max, impact_angle_deg, impact_duration, E_Fmax):
    """根据正弦脉冲计算碰撞过程中总冲击力和相关时间离散参数。"""
    flow_time = 1  # s
    volume_total = DEM_flow_rate * flow_time
    radius_avg = (radius_max + radius_min) / 2
    number_of_DEM = math.ceil(ratio_solid * volume_total / (4/3 * np.pi * (radius_avg**3 )))
    t_per_DEM = flow_time / number_of_DEM
    angle_impact = np.sin(np.radians(impact_angle_deg))

    num_pieces = max(math.ceil(impact_duration/t_per_DEM),1)
    num_points = 1000

    t_pieces  = np.zeros((num_pieces,num_points)) 
    sin_total = np.zeros((num_pieces,num_points))
    cos_total = np.zeros((num_pieces,num_points))

    for i in range(num_pieces):
        sin_array = np.zeros((1,num_points))
        cos_array = np.zeros((1,num_points))
        t_pieces[i] = np.linspace(i*t_per_DEM, (i+1)*t_per_DEM, num_points)

        for j in range(i+1):
            sin_array = np.sin((np.pi/impact_duration)*(t_pieces[i]-j*t_per_DEM))
            cos_array = np.cos((np.pi/impact_duration)*(t_pieces[i]-j*t_per_DEM))

            sin_array[sin_array < 0] = 0
            cos_array[sin_array < 0] = 0

            sin_total[i] = sin_total[i] + sin_array
            cos_total[i] = cos_total[i] + cos_array
            
        plt.plot(t_pieces[i], sin_total[i],'-*')
        plt.plot(t_pieces[i], cos_total[i],'-s')
        plt.xlabel('Time (s)')
        plt.ylabel('Function value')
        plt.title('Function value over time')

    plt.plot(np.array([0,impact_duration]),np.array([np.max(sin_total), np.max(sin_total)]),'-.*r')
    plt.plot(np.array([impact_duration,impact_duration]),np.array([0, np.max(sin_total)]),'-.*r')
    plt.xlim(0, 1.05*np.max(t_pieces)) # 设置x轴范围从0到冲击时间
    plt.ylim(1.05*min(np.min(sin_total),np.min(cos_total)), 1.05*max(np.max(sin_total),np.max(cos_total)))  # 设置y轴范围从0到最大sine_total值
    plt.show()

    if Pier_shape == 'round':
        gamma_space = np.pi/4
    elif Pier_shape =='square':
        gamma_space = 1
    else:    
        raise ValueError('Shape Of Section Not Found!')

    total_force = gamma_space * angle_impact * np.max(sin_total) * E_Fmax

    return num_pieces, t_per_DEM, total_force

if __name__ == '__main__':
    # 参数定义

    # This study
    DEM_Volumn = 16000      # 碎屑流方量：m^3
    DEM_depth = 2.4 + (20-2.4)/(16000-1000) * (DEM_Volumn-1000)      #  16000m^3方量：20m；8000m^3方量：12m；4000m^3方量：8m；2000m^3方量：4m；1000m^3方量：2.4m
    DEM_velocity = 10.8      # m/s
    DEM_density = 2286      # kg/m3  花岗岩密度2500kg/m3
    DEM_modulus = 25.8e9      # Pa   花岗岩弹性模量50-100GPa
    DEM_miu = 0.2          # Poisson's ratio  花岗岩泊松比0.1-0.3
    radius_min = 0.3  # m
    radius_max = 0.6  # m
    ratio_solid = np.pi/6.0 # 固相体积分数np.pi/6.0
    impact_angle_deg = 90   # 冲击角度 °
    
    # Pier_shape = 'square'
    Pier_shape = 'round'
    Pier_width = 2.2        # m
    Pier_modulus = 30e9    # Pa 混凝土弹性模量:31GPa
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
    force_min, force_max, force_equ, force_average = compute_Hertz_contact_forces(radius_min, radius_max, modulus_equ, DEM_density, DEM_velocity)
    print('[Hertz Elastic Theory]: ', '\n\tF_min=', np.round(force_min,3), 'F_max=',np.round(force_max,3), 'F_equ=', np.round(force_equ,3),'F_average=',np.round(force_average,3),'N')

    # Thornton弹性-理想塑性接触理论计算冲击力（接触力）
    print('[Thornton Elasto-Plastic Theory]: ')
    v_y, F_min, F_max, E_Fmax = compute_Thornton_contact_force(radius_min, radius_max, modulus_equ, DEM_density, DEM_velocity, sigma_y)
    print(f'\tF_min = {np.round(F_min,3)}, F_max = {np.round(F_max,3)}, force_average = {np.round(E_Fmax,3)}', 'N')

    # 碰撞过程时间离散性和总冲击力
    #num_pieces, t_per_DEM, total_force = compute_total_impact_force_triangle( DEM_flow_rate, Pier_shape, ratio_solid, radius_min, radius_max, impact_angle_deg, impact_duration_elastoplastic, E_Fmax)
    num_pieces, t_per_DEM, total_force = compute_total_impact_force_sine(DEM_flow_rate, Pier_shape, ratio_solid, radius_min, radius_max, impact_angle_deg, impact_duration_elastoplastic, E_Fmax)
    print('\tNumber of 3D prticles in a single period =', np.round(num_pieces), '\n\tt_per_DEM =', np.round(t_per_DEM,9), 'total_force =', np.round(total_force,3), 'N')
    

    ''' 
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

    #Pier_shape = 'square'
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