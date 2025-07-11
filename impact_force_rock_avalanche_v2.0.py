import numpy as np
from scipy.integrate import quad

def adjust_radius(radius_min, radius_max):
    """如果半径上下限相等，则微调radius_max，防止除零错误。"""
    if radius_max == radius_min:
        radius_max += 1e-6
    return radius_min, radius_max

def compute_effective_pier_width(section_shape, pier_width):
    """计算桥墩有效宽度，根据截面形状调整。"""
    if section_shape == 'round':
        return pier_width / 1.3
    elif section_shape == 'square':
        return pier_width
    else:
        raise ValueError('Shape Of Section Not Found!')

def compute_number_effect(pier_width_effect, dem_depth, radius_min, radius_max):
    """计算填充的圆形数量估算。"""
    area_effect = pier_width_effect * dem_depth
    average_circle_area = np.pi * (radius_max**2 + radius_min**2) / 2
    number_effect = area_effect / average_circle_area
    return int(number_effect)

def compute_elastic_modulus_equivalent(pier_modulus, dem_modulus):
    """计算等效弹性模量。"""
    return 1 / (1 / pier_modulus + 1 / dem_modulus)

def compute_radius_equivalent(radius_min, radius_max):
    """计算等效半径。"""
    return ((1/3 * (radius_max**3 - radius_min**3) / (radius_max - radius_min)) ** 0.5)

def compute_elastic_contact_forces(radius_min, radius_max, radius_equ, modulus_eq, dem_density, dem_velocity):
    """计算弹性接触理论中的接触力。"""
    factor = (4/3) * (5*np.pi/4)**(3/5) * modulus_eq**(2/5) * dem_density**(3/5) * dem_velocity**(6/5)
    force_min = factor * radius_min**2
    force_max = factor * radius_max**2
    force_equ = factor * radius_equ**2
    return force_min, force_max, force_equ

def compute_average_elastic_force(radius_min, radius_max, modulus_eq, dem_density, dem_velocity):
    """计算平均接触力。"""
    epr_average = 1/3 * (radius_max**3 - radius_min**3) / (radius_max - radius_min)
    force_average = epr_average * (4/3) * (5*np.pi/4)**(3/5) * modulus_eq**(2/5) * dem_density**(3/5) * dem_velocity**(6/5)
    return force_average

def compute_elasto_plastic_forces(radius_min, radius_max, modulus_eq, dem_density, dem_velocity, sigma_y):
    """计算弹性-理想塑性接触力及平均力。"""
    #mass_star = dem_density * 4/3 * np.pi * radius_max**3
    #v_y = 3.194*(sigma_y**5 * radius_max**3 / (modulus_equal**4 * mass_star))**0.5
    velocity_yield = 1.56 * (sigma_y**5 / (modulus_eq**4 * dem_density))**0.5
    Force_yield_max = sigma_y**3 * np.pi**3 * radius_max**2 / (6 * modulus_eq**2)
    Force_yield_min = sigma_y**3 * np.pi**3 * radius_min**2 / (6 * modulus_eq**2)

    if dem_velocity < velocity_yield:
        Force_radius_min = (4/3) * modulus_eq**0.4 * radius_min**2 * (5 * dem_density * np.pi * dem_velocity**2 / 4)**0.6
        Force_radius_max = (4/3) * modulus_eq**0.4 * radius_max**2 * (5 * dem_density * np.pi * dem_velocity**2 / 4)**0.6
        Expect_force = (4/9) * modulus_eq**0.4 * (5 * dem_density * np.pi * dem_velocity**2 / 4)**0.6 * \
                 (radius_max**3 - radius_min**3) / (radius_max - radius_min)
    else:
        Force_radius_max = np.sqrt(Force_radius_max**2 + (4/3) * np.pi**2 * sigma_y * dem_density * (dem_velocity**2 - v_y**2) * radius_max**4)
        Force_radius_min = np.sqrt(Force_radius_min**2 + (4/3) * np.pi**2 * sigma_y * dem_density * (dem_velocity**2 - v_y**2) * radius_min**4)
        A = F_y**2
        B = (4/3) * np.pi**2 * sigma_y * dem_density * (dem_velocity**2 - v_y**2)

        def integrand(x):
            return np.sqrt(A + B * x**4)

        Int_value, Int_error = quad(integrand, radius_min, radius_max)
        E_Fmax = Int_value / (radius_max - radius_min)

    return v_y, F_y, F_max, E_Fmax

def compute_collision_force(area_effect, dem_velocity, ratio_solid, radius_min, radius_max, impact_angle_deg, impact_duration, E_Fmax):
    """计算碰撞过程中总冲击力和相关时间离散参数。"""
    volume_total = area_effect * dem_velocity * 1
    number_of_DEM = ratio_solid * volume_total / (4/3 * np.pi * (radius_max**3 + radius_min**3))
    delta_t_DEM = 1 / number_of_DEM
    k = int(impact_duration / (2 * delta_t_DEM))
    angle_impact = np.sin(np.radians(impact_angle_deg))
    total_force = angle_impact * E_Fmax * (k + 1 - k * (k + 1) / 2 * delta_t_DEM / (0.5 * impact_duration))
    return int(number_of_DEM), delta_t_DEM, total_force

def main():
    # 参数定义
    DEM_density = 2550      # kg/m3
    DEM_depth = 0.03        # m
    section_shape = 'round'
    Pier_width = 0.1        # m
    Pier_modulus = 3.2e9    # Pa
    sigma_y = 10e6          # Pa
    radius_min = 4.0e-3     # m
    radius_max = 4.0e-3     # m
    DEM_modulus = 60e9      # Pa
    DEM_velocity = 1.4      # m/s
    ratio_solid = 0.45      # 固相体积分数
    impact_angle_deg = 72   # 冲击角度 °
    impact_duration = 0.002 # s

    # 调整半径
    radius_min, radius_max = adjust_radius(radius_min, radius_max)

    # 计算桥墩有效宽度
    pier_width_effective = compute_effective_pier_width(section_shape, Pier_width)

    # 计算有效填充个数
    number_effect = compute_number_effect(pier_width_effective, DEM_depth, radius_min, radius_max)
    print('number_effect =', number_effect)

    # 计算等效弹性模量和半径
    modulus_equ = compute_elastic_modulus_equivalent(Pier_modulus, DEM_modulus)
    radius_equ = compute_radius_equivalent(radius_min, radius_max)

    # 弹性接触理论计算冲击力（接触力）
    force_min, force_max, force_equ = compute_elastic_contact_forces(
        radius_min, radius_max, radius_equ, modulus_equ, DEM_density, DEM_velocity)
    force_average = compute_average_elastic_force(radius_min, radius_max, modulus_equ, DEM_density, DEM_velocity)
    print('Elastic Theory: average contact force =', np.round(force_average, 3), 'N')

    # 弹性-理想塑性接触理论计算冲击力（接触力）
    v_y, F_y, F_max, E_Fmax = compute_elasto_plastic_forces(
        radius_min, radius_max, modulus_equ, DEM_density, DEM_velocity, sigma_y)
    print(f'v_y = {np.round(v_y,3)}, F_y = {np.round(F_y,3)}')
    print(f'Elasto-Plastic Theory: F_max = {np.round(F_max)}, average contact force = {np.round(E_Fmax)}')

    # 碰撞过程时间离散性和总冲击力
    area_effect = pier_width_effective * DEM_depth
    number_of_DEM, delta_t_DEM, total_force = compute_collision_force(
        area_effect, DEM_velocity, ratio_solid, radius_min, radius_max, impact_angle_deg, impact_duration, E_Fmax)
    print('number_of_DEM =', number_of_DEM, 'delta_t_DEM =', delta_t_DEM, 'total_force =', total_force)

if __name__ == '__main__':
    main()
