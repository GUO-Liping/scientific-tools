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
    DEM_velocity = np.sqrt(2*9.81*np.array([0.0, 0.01, 0.05, 0.10, 0.15, 0.3, 0.6, 1.2, 2.4]) ) #      # m/s
    case_number = len(DEM_velocity)
    DEM_modulus = 210e9 * np.ones(case_number)      # Pa   花岗岩弹性模量50-100GPa
    DEM_miu = 0.3 * np.ones(case_number)          # Poisson's ratio  花岗岩泊松比0.1-0.3
    DEM_strength = 235e6 * np.ones(case_number)     # 花岗岩强度 Pa
    DEM_radius = 0.09*np.ones(case_number)
    DEM_density = 400/(4/3*np.pi*DEM_radius**3)      # kg/m3  花岗岩密度2500kg/m3
    DEM_mass = DEM_density * 4/3*np.pi* DEM_radius**3

    Pier_velocity = 0 * np.ones(case_number)  # (11.8 + (9.8-11.8)/(8000-1000) * (DEM_Volumn-1000))     # m/s
    Pier_density = 2500 * np.ones(case_number)      # kg/m3  花岗岩密度2500kg/m3
    Pier_modulus = 30e9 * np.ones(case_number)      # Pa   花岗岩弹性模量50-100GPa
    Pier_miu = 0.2 * np.ones(case_number)          # Poisson's ratio  花岗岩泊松比0.1-0.3
    Pier_strength = 42e6 * np.ones(case_number)     # 花岗岩强度 Pa
    Pier_radius = np.inf*np.ones(case_number)
    
    
    # 参数定义: Majeed ZZA, Lam NTK, Lam C, et al (2019) https://doi.org/10.1016/j.ijimpeng.2019.103324
    DEM_velocity = np.array([15.0])  # np.array([2.2, 3.1, 3.8, 4.4, 5.4, 6.3]) ,np.array([9.52, 11.72, 14.2, 17.2, 20.8, 23.8, 26.3])
    case_number = len(DEM_velocity)
    DEM_modulus = 65e9 * np.ones(case_number)      # Pa   花岗岩弹性模量50-100GPa
    DEM_miu = 0.3 * np.ones(case_number)          # Poisson's ratio  花岗岩泊松比0.1-0.3
    DEM_strength = 160e6 * np.ones(case_number)     # 花岗岩强度 Pa
    DEM_radius = 0.05*np.ones(case_number)
    DEM_density = 1.4/(4/3*np.pi*DEM_radius**3)      # kg/m3  花岗岩密度2500kg/m3
    DEM_mass = DEM_density * 4/3*np.pi* DEM_radius**3

    Pier_velocity = 0 * np.ones(case_number)  # (11.8 + (9.8-11.8)/(8000-1000) * (DEM_Volumn-1000))     # m/s
    Pier_density = 2700 * np.ones(case_number)      # kg/m3  混凝土密度2500kg/m3
    Pier_modulus = 30e9 * np.ones(case_number)      # Pa   混凝土弹性模量30-50GPa
    Pier_miu = 0.30 * np.ones(case_number)          # Poisson's ratio  混凝土泊松比0.1-0.3
    Pier_strength = 42e6 * np.ones(case_number)     # 混凝土强度 Pa
    Pier_radius = np.inf*np.ones(case_number)
    '''

    # 参数定义: Choi, et al (2020) 
    DEM_velocity = np.array([3.2])  # np.array([2.2, 3.1, 3.8, 4.4, 5.4, 6.3]) ,np.array([9.52, 11.72, 14.2, 17.2, 20.8, 23.8, 26.3])
    case_number = len(DEM_velocity)
    DEM_modulus = 55e9 * np.ones(case_number)      # Pa   花岗岩弹性模量50-100GPa
    DEM_miu = 0.25 * np.ones(case_number)          # Poisson's ratio  花岗岩泊松比0.1-0.3
    DEM_strength = 70e6 * np.ones(case_number)     # 花岗岩强度 Pa
    DEM_radius = 0.005*np.ones(case_number)
    DEM_density = 2500      # kg/m3  花岗岩密度2500kg/m3
    DEM_mass = DEM_density * 4/3*np.pi* DEM_radius**3

    Pier_velocity = 0 * np.ones(case_number)  # (11.8 + (9.8-11.8)/(8000-1000) * (DEM_Volumn-1000))     # m/s
    Pier_density = 2500 * np.ones(case_number)      # kg/m3  混凝土密度2500kg/m3
    Pier_modulus = 3e9 * np.ones(case_number)      # Pa   混凝土弹性模量30-50GPa
    Pier_miu = 0.30 * np.ones(case_number)          # Poisson's ratio  混凝土泊松比0.1-0.3
    Pier_strength = 50e6 * np.ones(case_number)     # 混凝土强度 Pa
    Pier_radius = np.inf*np.ones(case_number)
    
    # 计算等效弹性模量
    modulus_eq = 1 / ((1-Pier_miu**2) / Pier_modulus + (1-DEM_miu**2) / DEM_modulus)
    radius_eq =  1 / (1 / DEM_radius + 1 / Pier_radius)
    velocity_eq = np.abs(DEM_velocity - Pier_velocity)

    C_JG = 1.6* np.ones(case_number)  # 1.295*np.exp(0.736*DEM_miu)
    sigma_y_Th = np.minimum(DEM_strength, Pier_strength)
    sigma_y_JG = sigma_y_Th * C_JG

    # 计算屈服条件
    velocity_y_Th = (np.pi/(2*modulus_eq))**2 * (8*np.pi*radius_eq**3 / (15*DEM_mass))**(1/2) * sigma_y_Th**(5/2)
    velocity_y_Th0 = 3.194 * (1/modulus_eq)**2 * (radius_eq**3 / DEM_mass)**(1/2) * sigma_y_Th**(5/2)
    velocity_y_Th1 = (np.pi/(2*modulus_eq))**2 * (2/(5*DEM_density))**(1/2) * sigma_y_Th**(5/2)
    velocity_y_Th2 = 1.56 * (sigma_y_Th**5 / (modulus_eq**4 * DEM_density))**(1/2)
    print('velocity_y_Th =', velocity_y_Th )
    print('velocity_y_Th0=', velocity_y_Th0)
    print('velocity_y_Th1=', velocity_y_Th1)
    print('velocity_y_Th2=', velocity_y_Th2)
    force_y_Th = sigma_y_Th**3 * np.pi**3 * radius_eq**2 / (6 * modulus_eq**2)
    delta_y_Th = sigma_y_Th**2 * np.pi**2 * radius_eq / (4*modulus_eq**2)

    velocity_y_JG = velocity_y_Th * C_JG**(5/2)# 
    force_y_JG = force_y_Th * C_JG**3
    delta_y_JG = delta_y_Th * C_JG**2

    # Hertz弹性接触理论计算冲击力（接触力）
    force_Hertz  = (4/3) * modulus_eq**(2/5) * radius_eq**(1/5) * DEM_mass**(3/5) * velocity_eq**(6/5) * (15/16)**(3/5)
    force_Hertz0 = (4/3) * modulus_eq**(2/5) * (5*np.pi/4)**(3/5) * DEM_density**(3/5) * velocity_eq**(6/5) * DEM_radius**2

    # Thornton弹性-理想塑性接触理论计算冲击力（接触力）
    force_Th_raw  = np.sqrt(force_y_Th**2 + np.pi*sigma_y_Th*DEM_mass * (velocity_eq**2 - velocity_y_Th**2) * radius_eq)
    force_Th0 = np.sqrt(force_y_Th**2 + np.pi*sigma_y_Th*(DEM_density*4/3*np.pi*DEM_radius**3) * (velocity_eq**2 - velocity_y_Th**2) * radius_eq)

    # JG弹性-理想塑性接触理论计算冲击力（接触力）
    force_JG_raw  = np.sqrt(force_y_JG**2 + np.pi*sigma_y_JG*DEM_mass * (velocity_eq**2 - velocity_y_JG**2) * radius_eq)
    force_JG0 = np.sqrt(force_y_JG**2 + np.pi*sigma_y_JG*(DEM_density*4/3*np.pi*DEM_radius**3) * (velocity_eq**2 - velocity_y_JG**2) * radius_eq)

    # 弹、塑性分段函数计算冲击力
    force_Th = np.where(velocity_eq <= velocity_y_Th, force_Hertz, force_Th_raw)
    force_JG = np.where(velocity_eq <= velocity_y_JG, force_Hertz, force_JG_raw)

    # 恢复系数e_rebond
    c1 = velocity_y_Th / velocity_eq
    coeff_Th = (6 * 3**(1/2) / 5)**(1/2) * (1-1/6 * c1**2)**(1/2) * (c1/(c1 + 2*(6/5 - 1/5*c1**2)**(1/2)))**(1/4)

    # A Finite Element Study of Elasto-Plastic Hemispherical Contact Against a Rigid Flat
    V1_star_JG = velocity_eq/velocity_y_JG
    epsilon_y_JG = sigma_y_JG / modulus_eq

    coeff_Th2 = 1.185 * (velocity_y_Th / velocity_eq)**(1/4)
    coeff_JG = 1-0.0361*(epsilon_y_JG**(-0.114)) * np.log(V1_star_JG) * (V1_star_JG-1)**(9.5*epsilon_y_JG)

    print('DEM_mass      =',   np.array2string(DEM_mass,                     separator=', ', precision=3), 'kg')      
    print('DEM_velocity  =',   np.array2string(DEM_velocity,                 separator=', ', precision=3), 'm/s')      
    print('C_JG          =',   np.array2string(C_JG,                         separator=', ', precision=3), ' ')      
    print('velocity_y_Th =',   np.array2string(velocity_y_Th*1000,           separator=', ', precision=4), 'mm/s')      
    print('velocity_y_JG =',   np.array2string(velocity_y_JG*1000,           separator=', ', precision=4), 'mm/s')      
    print('delta_y_Th    =',   np.array2string(delta_y_Th*1000,              separator=', ', precision=4), 'mm') 
    print('delta_y_JG    =',   np.array2string(delta_y_JG*1000,              separator=', ', precision=4), 'mm')      
    print('force_y_Th    =',   np.array2string(force_y_Th,                   separator=', ', precision=3), 'N')      
    print('force_y_JG    =',   np.array2string(force_y_JG,                   separator=', ', precision=3), 'N')      
    print('force_Hz      =',   np.array2string(force_Hertz/1000,             separator=', ', precision=2), 'kN') 
    print('force_Th      =',   np.array2string(force_Th/1000,                separator=', ', precision=2), 'kN') 
    print('force_JG      =',   np.array2string(force_JG/1000,                separator=', ', precision=2), 'kN') 
    print('coeff_Th      =',   np.array2string(coeff_Th,                     separator=', ', precision=3), ' ') 
    print('coeff_JG      =',   np.array2string(coeff_JG,                     separator=', ', precision=3), ' ') 
