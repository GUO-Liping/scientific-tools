import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy import stats
import math


# 默认打印禁止科学计数法
np.set_printoptions(suppress=True)
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用 SimHei 字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


if __name__ == '__main__':
    '''
    # 参数定义: Fujikake K, Li B, Soeun S (2009) https://doi.org/10.1061/(ASCE)ST.1943-541X.0000039
    case_number = 1
    DEM_velocity = 4.85 * np.ones(case_number)  # (11.8 + (9.8-11.8)/(8000-1000) * (DEM_Volumn-1000))     # m/s
    DEM_modulus = 210e9 * np.ones(case_number)      # Pa   花岗岩弹性模量50-100GPa
    DEM_miu = 0.30 * np.ones(case_number)          # Poisson's ratio  花岗岩泊松比0.1-0.3
    DEM_strength = 235e6 * np.ones(case_number)     # 花岗岩强度 Pa
    DEM_radius = 0.09*np.ones(case_number)
    DEM_density = 400/(4/3*np.pi*DEM_radius**3)      # kg/m3  花岗岩密度2500kg/m3
    DEM_mass = DEM_density * 4/3*np.pi* DEM_radius**3

    Pier_velocity = 0 * np.ones(case_number)  # (11.8 + (9.8-11.8)/(8000-1000) * (DEM_Volumn-1000))     # m/s
    Pier_density = 2500 * np.ones(case_number)      # kg/m3  花岗岩密度2500kg/m3
    Pier_modulus = 100e9 * np.ones(case_number)      # Pa   花岗岩弹性模量50-100GPa
    Pier_miu = 0.20 * np.ones(case_number)          # Poisson's ratio  花岗岩泊松比0.1-0.3
    Pier_strength = 30e6 * np.ones(case_number)     # 花岗岩强度 Pa
    Pier_radius = np.inf*np.ones(case_number)
    '''

    # 参数定义: Majeed ZZA, Lam NTK, Lam C, et al (2019) https://doi.org/10.1016/j.ijimpeng.2019.103324
    case_number = 3
    DEM_velocity = np.array([2.2, 3.8, 5.4])  # (11.8 + (9.8-11.8)/(8000-1000) * (DEM_Volumn-1000))     # m/s
    DEM_modulus = 65e9 * np.ones(case_number)      # Pa   花岗岩弹性模量50-100GPa
    DEM_miu = 0.2 * np.ones(case_number)          # Poisson's ratio  花岗岩泊松比0.1-0.3
    DEM_strength = 160e6 * np.ones(case_number)     # 花岗岩强度 Pa
    DEM_radius = 0.05*np.ones(case_number)
    DEM_density = 1.4/(4/3*np.pi*DEM_radius**3)      # kg/m3  花岗岩密度2500kg/m3
    DEM_mass = DEM_density * 4/3*np.pi* DEM_radius**3

    Pier_velocity = 0 * np.ones(case_number)  # (11.8 + (9.8-11.8)/(8000-1000) * (DEM_Volumn-1000))     # m/s
    Pier_density = 2700 * np.ones(case_number)      # kg/m3  混凝土密度2500kg/m3
    Pier_modulus = 3000e9 * np.ones(case_number)      # Pa   混凝土弹性模量50-100GPa
    Pier_miu = 0.20 * np.ones(case_number)          # Poisson's ratio  混凝土泊松比0.1-0.3
    Pier_strength = 30000e6 * np.ones(case_number)     # 混凝土强度 Pa
    Pier_radius = np.inf*np.ones(case_number)

    # 计算等效弹性模量
    modulus_eq = 1 / ((1-Pier_miu**2) / Pier_modulus + (1-DEM_miu**2) / DEM_modulus)
    radius_eq =  1 / (1 / DEM_radius + 1 / Pier_radius)
    velocity_eq = np.abs(DEM_velocity - Pier_velocity)
    sigma_y = np.minimum(DEM_strength, Pier_strength)

    # 计算屈服条件
    velocity_y_Th = (np.pi/(2*modulus_eq))**2 * (8*np.pi*radius_eq**3 / (15*DEM_mass))**(1/2) * sigma_y**(5/2)
    velocity_y_Th0 = 3.194 * (1/modulus_eq)**2 * (radius_eq**3 / DEM_mass)**(1/2) * sigma_y**(5/2)
    velocity_y_Th1 = (np.pi/(2*modulus_eq))**2 * (2/(5*DEM_density))**(1/2) * sigma_y**(5/2)
    velocity_y_Th2 = 1.56 * (sigma_y**5 / (modulus_eq**4 * DEM_density))**(1/2)

    force_y_Th = sigma_y**3 * np.pi**3 * radius_eq**2 / (6 * modulus_eq**2)
    delta_y_Th = sigma_y**2 * np.pi**2 * radius_eq / (4*modulus_eq**2)

    C_JG = 1.295*np.exp(0.736*DEM_miu)
    velocity_y_JG = velocity_y_Th * C_JG**(5/2)# 
    force_y_JG = force_y_Th * C_JG**3
    delta_y_JG = delta_y_Th * C_JG**2

    # Hertz弹性接触理论计算冲击力（接触力）
    force_Hertz  = (4/3) * modulus_eq**(2/5) * radius_eq**(1/5) * DEM_mass**(3/5) * velocity_eq**(6/5) * (15/16)**(3/5)
    force_Hertz0 = (4/3) * modulus_eq**(2/5) * (5*np.pi/4)**(3/5) * DEM_density**(3/5) * velocity_eq**(6/5) * DEM_radius**2

    # Thornton弹性-理想塑性接触理论计算冲击力（接触力）
    force_Thorn  = np.sqrt(force_y_JG**2 + np.pi*sigma_y*DEM_mass * (velocity_eq**2 - velocity_y_JG**2) * radius_eq)
    force_Thorn0 = np.sqrt(force_y_JG**2 + np.pi*sigma_y*(DEM_density*4/3*np.pi*DEM_radius**3) * (velocity_eq**2 - velocity_y_JG**2) * radius_eq)

    # 恢复系数e_rebond
    c1 = velocity_y_JG / velocity_eq
    coeff_r = (6 * 3**(1/2) / 5)**(1/2) * (1-1/6 * c1**2)**(1/2) * (c1/(c1 + 2*(6/5 - 1/5*c1**2)**(1/2)))**(1/4)

    print('DEM_mass      =',   np.array2string(DEM_mass,                     separator=', ', precision=6), 'kg')      
    print('velocity_eq   =',   np.array2string(velocity_eq,                  separator=', ', precision=6), 'm/s')      
    print('velocity_y_JG =',   np.array2string(velocity_y_JG*1000,           separator=', ', precision=6), 'mm/s')      
    print('force_y_JG    =',   np.array2string(force_y_JG,                   separator=', ', precision=6), 'N')      
    print('delta_y_JG    =',   np.array2string(delta_y_JG*1000,              separator=', ', precision=6), 'mm')      
    print('force_y_Th    =',   np.array2string(force_y_Th,                   separator=', ', precision=6), 'N')      
    print('delta_y_Th    =',   np.array2string(delta_y_Th*1000,              separator=', ', precision=6), 'mm') 
    print('force_Hertz   =',   np.array2string(force_Hertz/1000,             separator=', ', precision=6), 'kN') 
    print('force_Hertz0  =',   np.array2string(force_Hertz0/1000,            separator=', ', precision=6), 'kN') 
    print('force_Thorn   =',   np.array2string(force_Thorn/1000,             separator=', ', precision=6), 'kN') 
    print('force_Thorn0  =',   np.array2string(force_Thorn0/1000,            separator=', ', precision=6), 'kN') 
    print('coeff_r  =',        np.array2string(coeff_r,                      separator=', ', precision=3), ' ') 
