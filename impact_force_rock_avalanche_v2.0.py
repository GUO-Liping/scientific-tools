import numpy as np
from scipy.integrate import quad

def adjust_radius(radius_min, radius_max):
    """如果半径上下限相等，则微调radius_max，防止除零错误。"""
    if radius_max == radius_min:
        radius_max += 1e-6
    return radius_min, radius_max

def compute_impact_duration(DEM_density, DEM_modulus, DEM_miu, DEM_radius, DEM_velocity):
    k1 = (1-miu1**2)/(np.pi*E1)
    k2 = (1-miu2**2)/(np.pi*E2)
    n0 = np.sqrt(16/(9*np.pi**2) * radius1 * radius2 / ((k1+k2)**2*(R1+R2)))
    n1 = (m1 + m2) / (m1*m2)

    alpha1 = (5*velocity_relative**2/(4*n0*n1))**(2/5)
    alpha_test = 2.943 * (5*np.sqrt(2)/4 * np.pi*DEM_density * (1-DEM_miu**2)/DEM_modulus)**(2/5) * DEM_radius / ((2*DEM_velocity)**(1/5)) # s
    return alpha1

def compute_effective_pier_width(pier_shape, pier_width):
    """计算桥墩有效宽度，根据截面形状调整。"""
    if pier_shape == 'round':
        return pier_width / 1.3
    elif pier_shape == 'square':
        return pier_width
    else:
        raise ValueError('Shape Of Section Not Found!')

def compute_elastic_modulus_equivalent(pier_modulus, pier_miu, dem_modulus, dem_miu):
    """计算等效弹性模量。"""
    return 1 / ((1-pier_miu**2) / pier_modulus + (1-dem_miu**2) / dem_modulus)

def compute_Hertz_contact_forces(radius_min, radius_max, modulus_eq, dem_density, dem_velocity):
    """计算弹性Hertz接触理论中的接触力。"""
    radius_equ = (1/3 * (radius_max**2 + radius_max*radius_min + radius_min**2)) ** 0.5
    factor = (4/3) * modulus_eq**(2/5) * (5*np.pi/4)**(3/5) * dem_density**(3/5) * dem_velocity**(6/5)
    force_min = factor * radius_min**2
    force_max = factor * radius_max**2
    force_equ = factor * radius_equ**2

    epr_average = 4/9 * (radius_max**2 + radius_max*radius_min + radius_min**2)
    force_average = epr_average * (5*np.pi/4)**(3/5) * modulus_eq**(2/5) * dem_density**(3/5) * dem_velocity**(6/5)

    return force_min, force_max, force_equ, force_average

def compute_Thornton_contact_force(radius_min, radius_max, modulus_eq, dem_density, dem_velocity, sigma_y):
    """计算弹性/理想塑性-接触力的期望。"""
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

def compute_total_impact_force(area_effect, dem_velocity, ratio_solid, radius_min, radius_max, impact_angle_deg, impact_duration, E_Fmax):
    """计算碰撞过程中总冲击力和相关时间离散参数。"""
    flow_time = 1  # s
    volume_total = area_effect * dem_velocity * flow_time
    number_of_DEM = ratio_solid * volume_total / (4/3 * np.pi * (radius_max**3 + radius_min**3))
    delta_t_DEM = flow_time / number_of_DEM
    k_slope = int(impact_duration / (2 * delta_t_DEM))
    angle_impact = np.sin(np.radians(impact_angle_deg))
    total_force = angle_impact * E_Fmax * (k_slope + 1 - k_slope * (k_slope + 1) / 2 * delta_t_DEM / (0.5 * impact_duration))
    return int(number_of_DEM), delta_t_DEM, total_force


if __name__ == '__main__':
    # 参数定义
    DEM_density = 2550      # kg/m3
    DEM_depth = 0.035       # m
    DEM_modulus = 60e9      # Pa
    DEM_miu = 0.25          # Poisson’s ratio
    DEM_velocity = 1.21     # m/s
    sigma_y = 30e6          # Pa
    radius_min = 4.0e-3     # m
    radius_max = 4.0e-3     # m
    ratio_solid = 0.45      # 固相体积分数
    impact_angle_deg = 72   # 冲击角度 °

    Pier_shape = 'round'
    Pier_width = 0.1
    Pier_modulus = 3.2e9    # Pa
    Pier_miu = 0.35         # Poisson’s ratio

    impact_duration = compute_impact_duration(DEM_density, DEM_modulus, DEM_miu, radius_max, DEM_velocity)
    print('impact_duration=', np.round(impact_duration,6), 's')

    # 调整半径
    radius_min, radius_max = adjust_radius(radius_min, radius_max)

    # 计算桥墩有效宽度
    pier_width_effective = compute_effective_pier_width(Pier_shape, Pier_width)

    # 计算等效弹性模量
    modulus_equ = compute_elastic_modulus_equivalent(Pier_modulus, Pier_miu, DEM_modulus, DEM_miu)

    # Hertz弹性接触理论计算冲击力（接触力）
    force_min, force_max, force_equ, force_average = compute_Hertz_contact_forces(radius_min, radius_max, modulus_equ, DEM_density, DEM_velocity)
    print('[Hertz Elastic Theory]: ', '\n\tF_min=', np.round(force_min,3), 'F_max=',np.round(force_max,3), 'F_equ=', np.round(force_equ,3),'F_average=',np.round(force_average,3),'N')

    # Thornton弹性-理想塑性接触理论计算冲击力（接触力）
    print('[Thornton Elasto-Plastic Theory]: ')
    v_y, F_min, F_max, E_Fmax = compute_Thornton_contact_force(radius_min, radius_max, modulus_equ, DEM_density, DEM_velocity, sigma_y)
    print(f'\tF_min = {np.round(F_min,3)}, F_max = {np.round(F_max,3)}, force_average = {np.round(E_Fmax,3)}', 'N')

    # 碰撞过程时间离散性和总冲击力
    area_effect = pier_width_effective * DEM_depth
    number_of_DEM, delta_t_DEM, total_force = compute_total_impact_force( area_effect, DEM_velocity, ratio_solid, radius_min, radius_max, impact_angle_deg, impact_duration, E_Fmax)
    print('\tNumber of 3D prticles =', np.round(number_of_DEM), '\n\tdelta_t_DEM =', np.round(delta_t_DEM,5), 'total_force =', np.round(total_force,3), 'N')
