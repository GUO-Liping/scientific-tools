import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy import stats
import math


# 默认打印禁止科学计数法
np.set_printoptions(suppress=True)
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用 SimHei 字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def adjust_radius(radius_min, radius_max):
    """如果半径上下限相等，则微调radius_max，防止除零错误。"""
    # 转为numpy数组，方便处理
    radius_min = np.array(radius_min, dtype=float)
    radius_max = np.array(radius_max, dtype=float)

    # 找出相等的位置
    mask_equal = (radius_min == radius_max)

    # 微调
    radius_min[mask_equal] -= 1e-5
    radius_max[mask_equal] += 1e-5

    return radius_min, radius_max


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

    coeff_re = np.minimum(coeff_re, 1.0)

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
    return t_elastic, t_elastoplastic

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


def compute_Thornton_contact_force(radius_min, radius_max, modulus_eq, dem_density, dem_velocity, sigma_y, distribute):
    """计算Thornton弹性/理想塑性-接触力的期望。"""

    F_single_min = np.zeros_like(dem_velocity)
    F_single_max = np.zeros_like(dem_velocity)
    E_Fmax_elastic = np.zeros_like(dem_velocity)
    E_Fmax_plastic = np.zeros_like(dem_velocity)
    E_Fmax = np.zeros_like(dem_velocity)

    # 计算弹性碰撞结果
    const_F_elastic = (4/3) * modulus_eq**0.4 * (5 * dem_density * np.pi * dem_velocity**2 / 4)**0.6
    F_single_min_elastic = const_F_elastic * radius_min**2
    F_single_max_elastic = const_F_elastic * radius_max**2
    #E_Fmax_elastic = (4/9) * (radius_max**2 + radius_max*radius_min + radius_min**2) * modulus_eq**0.4 * (5 * dem_density * np.pi * dem_velocity**2 / 4)**0.6

    # 计算塑性碰撞结果
    velocity_y = np.sqrt(np.pi**4/40) * (sigma_y**5 / (dem_density * modulus_eq**4))**0.5  # np.sqrt(np.pi**4/40)=1.560521475613219791

    const_F_plastic = np.sqrt((sigma_y**3*np.pi**3/(6*modulus_eq**2))**2 + np.pi**2*sigma_y*4/3*dem_density*(dem_velocity**2-velocity_y**2))
    F_single_min_plastic = const_F_plastic * radius_min**2
    F_single_max_plastic = const_F_plastic * radius_max**2
    #E_Fmax_elastic = (4/9) * (radius_max**2 + radius_max*radius_min + radius_min**2) * modulus_eq**0.4 * (5 * dem_density * np.pi * dem_velocity**2 / 4)**0.6

    def integrate_r(r, const_F):
            return const_F * r**2

    if distribute == 'uniform':

        def integrand_unif_distribute(r_min, r_max, const_F, field_x,field_n):
            """计算均匀分布下的接触力"""
            # 1. 截断均匀分布 (uniform 在 [r_min, r_max] 上天然归一化)
            data_expanded = np.repeat(field_x, field_n)
            dist_unif = stats.uniform(loc=r_min, scale=r_max - r_min)
            print(f"截断均匀分布 Uniform:\t loc={r_min:.4f}, scale={r_max - r_min:.4f}")
            # 执行数值积分，计算均匀分布下的期望接触力
            unif_result, unif_error = quad(lambda r: integrate_r(r,const_F) * dist_unif.pdf(r), r_min, r_max)

            x_plot = np.linspace(r_min, r_max, 100)
            plt.plot(x_plot, dist_unif.pdf(x_plot))
            return unif_result

        # 计算所有粒径范围的期望接触力
        for i in range(len(dem_velocity)):
            # 展开数据用于拟合
            field_x = np.linspace(radius_min[i], radius_max[i], 8)
            field_n = np.array([19, 65, 60, 38, 20, 16, 8, 7])
            plt.plot(field_x, field_n/((field_x[1] - field_x[0]) * field_n.sum()), 'o')

            E_Fmax_elastic[i] = integrand_unif_distribute(radius_min[i], radius_max[i], const_F_elastic[i], field_x,field_n)
            E_Fmax_plastic[i] = integrand_unif_distribute(radius_min[i], radius_max[i], const_F_plastic[i], field_x,field_n)


    elif distribute == 'normal':

        def integrand_norm_distribute(r_min, r_max, const_F, field_x,field_n):
            """计算正态分布下的接触力"""
            # 2. 截断正态分布
            data_expanded = np.repeat(field_x, field_n)

            mu = (r_max + r_min) / 2
            sigma = np.std(data_expanded, ddof=1)  # 无偏估计
            a, b = (r_min - mu) / sigma, (r_max - mu) / sigma
            dist_norm = stats.truncnorm(a, b, loc=mu, scale=sigma)
            diff_cdf = (dist_norm.cdf(r_max) - dist_norm.cdf(r_min))
            print(f"截断正态分布 TruncNorm:\t mu={mu:.4f}, sigma={sigma:.4f}, a={a:.4f}, b={b:.4f}")
            # 执行数值积分，计算均匀分布下的期望接触力
            norm_result, norm_error = quad(lambda r: integrate_r(r,const_F) * dist_norm.pdf(r)/diff_cdf, r_min, r_max)

            x_plot = np.linspace(r_min, r_max, 100)
            plt.plot(x_plot, dist_unif.pdf(x_plot)/diff_cdf)
            return norm_result
  
        # 计算所有粒径范围的期望接触力
        for i in range(len(dem_velocity)):
            # 展开数据用于拟合
            field_x = np.linspace(radius_min[i], radius_max[i], 8)
            field_n = np.array([19, 65, 60, 38, 20, 16, 8, 7])
            plt.plot(field_x, field_n/((field_x[1] - field_x[0]) * field_n.sum()), 'o')

            E_Fmax_elastic[i] = integrand_norm_distribute(radius_min[i], radius_max[i], const_F_elastic[i], field_x,field_n)
            E_Fmax_plastic[i] = integrand_norm_distribute(radius_min[i], radius_max[i], const_F_plastic[i], field_x,field_n)

    elif distribute == 'weibull_r':

        def integrand_weibull_r_distribute(r_min, r_max, const_F, field_x,field_n):
            """计算weibull分布下的接触力"""
            # 3. 截断 Weibull 右偏
            data_expanded = np.repeat(field_x, field_n)

            c_wei_r, loc_wei_r, scale_wei_r = stats.weibull_min.fit(data_expanded)
            a_r, b_r = (r_min - loc_wei_r) / scale_wei_r, (r_max - loc_wei_r) / scale_wei_r
            weibull_r_dist = stats.truncweibull_min(c_wei_r, a_r, b_r, loc=loc_wei_r, scale=scale_wei_r)
            diff_cdf = (weibull_r_dist.cdf(r_max) - weibull_r_dist.cdf(r_min))
            print(f"截断Weibull右偏 TruncWeibullMin_r:\t c={c_wei_r:.4f}, loc={loc_wei_r:.4f}, scale={scale_wei_r:.4f}, a={a_r:.4f}, b={b_r:.4f}")
            # 执行数值积分，计算均匀分布下的期望接触力
            weibull_r_result, weibull_r_error = quad(lambda r: integrate_r(r,const_F) * weibull_r_dist.pdf(r)/diff_cdf, r_min, r_max)

            x_plot = np.linspace(r_min, r_max, 100)
            plt.plot(x_plot, weibull_r_dist.pdf(x_plot)/diff_cdf)

            return weibull_r_result

        # 计算所有粒径范围的期望接触力
        for i in range(len(dem_velocity)):
            # 展开数据用于拟合
            field_x = np.linspace(radius_min[i], radius_max[i], 8)
            field_n = np.array([19, 65, 60, 38, 20, 16, 8, 7])
            plt.plot(field_x, field_n/((field_x[1] - field_x[0]) * field_n.sum()), 'o')

            E_Fmax_elastic[i] = integrand_weibull_r_distribute(radius_min[i], radius_max[i], const_F_elastic[i], field_x,field_n)
            E_Fmax_plastic[i] = integrand_weibull_r_distribute(radius_min[i], radius_max[i], const_F_plastic[i], field_x,field_n)

    elif distribute == 'weibull_l':

        def integrand_weibull_l_distribute(r_min, r_max, const_F, field_x,field_n):
            """计算weibull分布下的接触力"""
            # 4. 截断 Weibull 左偏（用反向数据拟合）
            data_expanded = np.repeat(field_x, field_n)

            c_wei_l, loc_wei_l, scale_wei_l = stats.weibull_min.fit(np.repeat(field_x, field_n[::-1]))
            a_l, b_l = (r_min - loc_wei_l) / scale_wei_l, (r_max - loc_wei_l) / scale_wei_l
            weibull_l_dist = stats.truncweibull_min(c_wei_l, a_l, b_l, loc=loc_wei_l, scale=scale_wei_l)
            diff_cdf = (weibull_l_dist.cdf(r_max) - weibull_l_dist.cdf(r_min))
            print(f"截断Weibull左偏 TruncWeibullMin_l:\t c={c_wei_l:.4f}, loc={loc_wei_l:.4f}, scale={scale_wei_l:.4f}, a={a_l:.4f}, b={b_l:.4f}")
            # 执行数值积分，计算均匀分布下的期望接触力
            weibull_l_result, weibull_l_error = quad(lambda r: integrate_r(r,const_F) * weibull_l_dist.pdf(r)/diff_cdf, r_min, r_max)
            
            x_plot = np.linspace(r_min, r_max, 100)
            plt.plot(x_plot, weibull_l_dist.pdf(x_plot)/diff_cdf)

            return weibull_l_result

        # 计算所有粒径范围的期望接触力
        for i in range(len(dem_velocity)):
            # 展开数据用于拟合
            field_x = np.linspace(radius_min[i], radius_max[i], 8)
            field_n = np.array([19, 65, 60, 38, 20, 16, 8, 7])
            plt.plot(field_x, field_n/((field_x[1] - field_x[0]) * field_n.sum()), 'o')

            E_Fmax_elastic[i] = integrand_weibull_l_distribute(radius_min[i], radius_max[i], const_F_elastic[i], field_x,field_n)
            E_Fmax_plastic[i] = integrand_weibull_l_distribute(radius_min[i], radius_max[i], const_F_plastic[i], field_x,field_n)

    elif distribute == 'exponential':

        def integrand_expo_distribute(r_min, r_max, const_F, field_x,field_n):
            """计算指数分布下的接触力"""
            # 5. 截断指数分布
            data_expanded = np.repeat(field_x, field_n)

            b_fit, loc_fit, scale_fit = stats.truncexpon.fit(data_expanded)
            dist_expo = stats.truncexpon(b_fit, loc=loc_fit, scale=scale_fit)
            diff_cdf = (dist_expo.cdf(r_max) - dist_expo.cdf(r_min))
            # 执行数值积分，计算均匀分布下的期望接触力
            expo_result, expo_error = quad(lambda r: integrate_r(r,const_F) * dist_expo.pdf(r)/diff_cdf, r_min, r_max)
            
            x_plot = np.linspace(r_min, r_max, 100)
            plt.plot(x_plot, dist_expo.pdf(x_plot)/diff_cdf)

            return expo_result

        # 计算所有粒径范围的期望接触力
        for i in range(len(dem_velocity)):
            # 展开数据用于拟合
            field_x = np.linspace(radius_min[i], radius_max[i], 8)
            field_n = np.array([19, 65, 60, 38, 20, 16, 8, 7])
            plt.plot(field_x, field_n/((field_x[1] - field_x[0]) * field_n.sum()), 'o')

            E_Fmax_elastic[i] = integrand_expo_distribute(radius_min[i], radius_max[i], const_F_elastic[i], field_x,field_n)
            E_Fmax_plastic[i] = integrand_expo_distribute(radius_min[i], radius_max[i], const_F_plastic[i], field_x,field_n)
         
    # 用mask选择结果，分别赋值
    mask_elastic = dem_velocity <= velocity_y      # 低速冲击掩码，True表示速度小于等于临界速度
    mask_plastic = ~mask_elastic                       # 高速冲击掩码，True表示速度大于临界速度
    
    F_single_min = np.zeros_like(dem_velocity)
    F_single_min[mask_elastic] = F_single_min_elastic[mask_elastic]
    F_single_min[mask_plastic] = F_single_min_plastic[mask_plastic]
    
    F_single_max = np.zeros_like(dem_velocity)
    F_single_max[mask_elastic] = F_single_max_elastic[mask_elastic]
    F_single_max[mask_plastic] = F_single_max_plastic[mask_plastic]
    
    E_Fmax[mask_elastic] = E_Fmax_elastic[mask_elastic]
    E_Fmax[mask_plastic] = E_Fmax_plastic[mask_plastic]

    return velocity_y, F_single_min, F_single_max, E_Fmax

def compute_total_impact_force_triangle(DEM_flow_rate, Pier_shape, ratio_solid, radius_min, radius_max, impact_angle_deg, impact_duration, E_Fmax):
    """根据三角形脉冲计算碰撞过程中总冲击力和相关时间离散参数。"""
    flow_time = 1  # s
    volume_total = DEM_flow_rate * flow_time
    #radius_avg = np.sqrt(1/3 * (radius_max**2 + radius_max*radius_min + radius_min**2))
    radius_avg = (radius_max + radius_min)/2
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
    try:
        # 确保输入参数为numpy数组
        DEM_flow_rate = np.array(DEM_flow_rate, dtype=float)
        impact_duration = np.array(impact_duration, dtype=float)
        E_Fmax = np.array(E_Fmax, dtype=float)
        
        flow_time = np.ones_like(E_Fmax)  # s
        volume_total = DEM_flow_rate * flow_time
        radius_avg = (radius_max + radius_min) / 2
        number_of_DEM = np.ceil(ratio_solid * volume_total / (4/3 * np.pi * (radius_avg**3)))
        t_per_DEM = flow_time / number_of_DEM
        angle_impact = np.sin(np.radians(impact_angle_deg))

        # 计算 num_pieces 并确保其为整数
        num_pieces = np.maximum(np.ceil(impact_duration / t_per_DEM), 1).astype(int)
        num_points = 1000

        coeff_sin_total = np.zeros_like(E_Fmax)

        # 向量化计算
        for k in range(len(E_Fmax)):
            t_pieces = np.linspace(0, impact_duration[k], num_pieces[k] * num_points).reshape(num_pieces[k], num_points)
            t_per_DEM_k = t_per_DEM[k]
            sin_total = np.zeros((num_pieces[k], num_points))

            for i in range(num_pieces[k]):
                t_piece = t_pieces[i]
                sin_array = np.sin((np.pi / impact_duration[k]) * (t_piece - t_per_DEM_k * np.arange(i + 1).reshape(-1, 1)))
                sin_array[sin_array < 0] = 0
                sin_total[i] = np.sum(sin_array, axis=0)
                #print('i=', i, 'k=', k)

            coeff_sin_total[k] = np.max(sin_total)

            # 移除绘图代码，以减少不必要的计算和提高效率
            # plt.plot(t_pieces[i], sin_total[i], '-*')
            # plt.xlabel('Time (s)')
            # plt.ylabel('Function value')
            # plt.title('Function value over time')
            # plt.show()

        # 根据桥墩形状设置 gamma_space
        if Pier_shape == 'round':
            gamma_space = np.pi / 4
        elif Pier_shape == 'square':
            gamma_space = 1
        else:    
            raise ValueError('Shape Of Section Not Found!')

        # 计算总冲击力
        total_force = gamma_space * angle_impact * coeff_sin_total * E_Fmax
        return num_pieces, t_per_DEM, total_force

    except Exception as e:
        print(f"Error in compute_total_impact_force_sine: {e}")
        return None, None, None


def compute_total_impact_force_sine_old(DEM_flow_rate, Pier_shape, ratio_solid, radius_min, radius_max, impact_angle_deg, impact_duration, E_Fmax):
    """根据正弦脉冲计算碰撞过程中总冲击力和相关时间离散参数。"""
    flow_time = np.ones_like(E_Fmax)  # s
    volume_total = DEM_flow_rate * flow_time
    radius_avg = (radius_max + radius_min)/2
    number_of_DEM = np.ceil(ratio_solid * volume_total / (4/3 * np.pi * (radius_avg**3 )))
    t_per_DEM = flow_time / number_of_DEM
    angle_impact = np.sin(np.radians(impact_angle_deg))

    num_pieces = np.maximum(np.ceil(impact_duration/t_per_DEM),1).astype(int)
    num_points = 1000

    coeff_sin_total = np.zeros_like(E_Fmax)

    for k in range(len(E_Fmax)):
        t_pieces  = np.zeros((num_pieces[k],num_points)) 
        sin_total = np.zeros((num_pieces[k],num_points))

        for i in range(num_pieces[k]):
            sin_array = np.zeros(num_points)
            t_pieces[i] = np.linspace(i*t_per_DEM[k], (i+1)*t_per_DEM[k], num_points)
    
            for j in range(i+1):
                sin_array = np.sin((np.pi/impact_duration[k])*(t_pieces[i]-j*t_per_DEM[k]))
    
                sin_array[sin_array < 0] = 0
    
                sin_total[i] = sin_total[i] + sin_array
                #print(f"i={i}, j={j},k={k}")
                
            plt.plot(t_pieces[i], sin_total[i],'-*')
            plt.xlabel('Time (s)')
            plt.ylabel('Function value')
            plt.title('Function value over time')
        #plt.show()
        coeff_sin_total[k] = np.max(sin_total)
        #plt.plot(np.array([0,impact_duration]),np.array([np.max(sin_total), np.max(sin_total)]),'-.*r')
        #plt.plot(np.array([impact_duration,impact_duration]),np.array([0, np.max(sin_total)]),'-.*r')
        #plt.show()

    if Pier_shape == 'round':
        gamma_space = np.pi/4
    elif Pier_shape =='square':
        gamma_space = 1
    else:    
        raise ValueError('Shape Of Section Not Found!')

    total_force = gamma_space * angle_impact * coeff_sin_total * E_Fmax

    return num_pieces, t_per_DEM, total_force

if __name__ == '__main__':
    # 参数定义

    # This study
    case_number = 9
    DEM_Volumn = np.linspace(2000, 2000, case_number)      # 碎屑流方量：m^3
    DEM_depth = (3.2 + (14.0-3.2)/(8000-1000) * (DEM_Volumn-1000))
    print('DEM_Volumn:', DEM_Volumn)      
    #  Prticle size: 0.3-0.6: 16000m^3方量：20m；8000m^3方量：13.5-14.5m/12.7m/s；4000m^3方量：6.4-8.3m/12m/s；2000m^3方量：3.9-4.9m/11m/s；1000m^3方量：2.9-3.45m/10.8m/s
    #  Prticle size: 0.6-1.2: 16000m^3方量：20m；8000m^3方量：12m；4000m^3方量：8m；2000m^3方量：4m；1000m^3方量：2.4m
    #  Prticle size: 0.3-1.2: 16000m^3方量：20m；8000m^3方量：12m；4000m^3方量：8m；2000m^3方量：4m；1000m^3方量：2.4m
    DEM_velocity = 11.5 * np.ones(case_number)      # m/s
    DEM_density = 2550 * np.ones(case_number)      # kg/m3  花岗岩密度2500kg/m3
    DEM_modulus = 50e9 * np.ones(case_number)      # Pa   花岗岩弹性模量50-100GPa
    DEM_miu = 0.2 * np.ones(case_number)          # Poisson's ratio  花岗岩泊松比0.1-0.3
    DEM_strength = 30e6 * np.ones(case_number)     # 花岗岩强度 Pa
    c_radius = np.array([0.45,0.75,1.05])
    r_radius = np.array([0.01,0.05,0.15])
    radius_min = np.repeat(c_radius, 3) - np.tile(r_radius, 3)  # m
    radius_max = np.repeat(c_radius, 3) + np.tile(r_radius, 3)  # m

    ratio_solid = np.pi/6.0 * np.ones(case_number) # 固相体积分数np.pi/6.0
    impact_angle_deg = 90 * np.ones(case_number)   # 冲击角度 °
    
    # Pier_shape = 'square'
    Pier_shape = 'round'
    Pier_width = 2.2 * np.ones(case_number)        # m
    Pier_modulus = 30e9 * np.ones(case_number)    # Pa 混凝土弹性模量:31GPa
    Pier_miu = 0.2 * np.ones(case_number)          # 混凝土Poisson's ratio ：0.2
    Pier_strength = 30e6 * np.ones(case_number)          # Pa C30混凝土强度:30 MPa
    
    # 调整半径
    radius_min, radius_max = adjust_radius(radius_min, radius_max)
    print('radius_min, radius_max=', radius_min, radius_max)
    DEM_flow_rate = compute_effective_flow_rate(Pier_width, DEM_depth, radius_max, DEM_velocity)    # m^3/s
    sigma_y = np.minimum(DEM_strength, Pier_strength)

    # 计算冲击时间
    impact_duration_elastic, impact_duration_elastoplastic = compute_elastoplastic_impact_duration(DEM_density, DEM_modulus, DEM_miu, radius_max, DEM_velocity, Pier_modulus, Pier_miu, sigma_y)
    #print('impact_duration_elastic =', np.round(impact_duration_elastic,9), 's')
    #print('impact_duration_elastoplastic =', np.round(impact_duration_elastoplastic,9), 's')

    # 计算等效弹性模量
    modulus_equ = 1 / ((1-Pier_miu**2) / Pier_modulus + (1-DEM_miu**2) / DEM_modulus)

    # Hertz弹性接触理论计算冲击力（接触力）
    force_min, force_max, force_equ, force_average = compute_Hertz_contact_forces(radius_min, radius_max, modulus_equ, DEM_density, DEM_velocity)
    #print('[Hertz Elastic Theory]: ', '\n\tF_min=', np.round(force_min/1000,3), '\n\tF_max=',np.round(force_max/1000,3),'\n\tF_average=',np.round(force_average/1000,3),'kN')

    # Thornton弹性-理想塑性接触理论计算冲击力（接触力）
    #print('[Thornton Elasto-Plastic Theory]: ')
    # 'exponential','uniform','normal','weibull_l','weibull_r'
    v_y, F_min, F_max, E_Fmax = compute_Thornton_contact_force(radius_min, radius_max, modulus_equ, DEM_density, DEM_velocity, sigma_y, 'weibull_r')
    #print(f'\tF_min = {np.round(F_min/1000,3)}, \n\tF_max = {np.round(F_max/1000,3)}, \n\tforce_average = {np.round(E_Fmax/1000,3)}', 'kN')

    # 碰撞过程时间离散性和总冲击力
    #num_pieces, t_per_DEM, total_force = compute_total_impact_force_triangle( DEM_flow_rate, Pier_shape, ratio_solid, radius_min, radius_max, impact_angle_deg, impact_duration_elastoplastic, E_Fmax)
    num_pieces, t_per_DEM, total_force = compute_total_impact_force_sine(DEM_flow_rate, Pier_shape, ratio_solid, radius_min, radius_max, impact_angle_deg, impact_duration_elastoplastic, E_Fmax)
    
    print(
    f"\tNumber of 3D particles in a single period = {np.round(num_pieces)}"
    f"\n\tt_per_DEM = {np.round(t_per_DEM, 5)}"
    f"\n\tDEM_velocity = {DEM_velocity}"
    f"\n\ttotal_force = {np.round(0.001*total_force, 3)} kN"
)
    plt.show()


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
