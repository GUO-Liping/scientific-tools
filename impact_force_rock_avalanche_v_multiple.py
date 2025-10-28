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


def compute_elastoplastic_t_contact(DEM_density, DEM_modulus, DEM_miu, DEM_radius, DEM_velocity, Pier_modulus, Pier_miu, sigma_y):
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

def compute_effective_volume_flux(pier_width, DEM_depth, radius_max, DEM_velocity):
    """计算桥墩有效宽度，根据截面形状调整。"""
    pier_width_effect = pier_width + 2 * radius_max
    DEM_depth_effect = DEM_depth
    effective_volume_flux = pier_width_effect * DEM_depth_effect * DEM_velocity
    
    return effective_volume_flux

def compute_DEM_depth(x_query):
    """
    查询给定 x_query 对应的平滑拟合值。
    曲线特点：先上升后平缓，右端趋于18。
    内部自动完成拟合和绘图。
    """
    # 原始数据
    x_data = np.array([1000, 2000, 4000, 8000, 16000])
    y_data = np.array([3.2, 6.4, 12.8, 16.0, 18.0])

    # 指数饱和模型： y = a - b * exp(-c * x)
    def exp_saturate(x, a, b, c):
        return a - b * np.exp(-c * x)

    # 初始猜测参数
    p0 = [max(y_data), max(y_data)-min(y_data), 1e-4]

    # 曲线拟合
    popt, _ = curve_fit(exp_saturate, x_data, y_data, p0=p0, maxfev=10000)

    # 绘制拟合曲线
    '''
    x_fit = np.linspace(min(x_data), max(x_data), 200)
    y_fit = exp_saturate(x_fit, *popt)  # * 的作用是将列表解包成多个独立参数

    plt.figure(figsize=(6,4))
    plt.scatter(x_data, y_data, color='red', label='data points')
    plt.plot(x_fit, y_fit, 'b-', label='fitted smooth curve')
    plt.axvline(x_query, color='gray', linestyle='--', label=f'x={x_query}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Smooth saturating fit (approaching 18)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    '''
    depth = exp_saturate(x_query, *popt)
    if depth <= 0:
        raise ValueError('The flow depth is negative, please check the input parameters!')
    # 返回查询值
    return depth


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
            #print(f"截断均匀分布 Uniform:\t loc={r_min:.4f}, scale={r_max - r_min:.4f}")
            # 执行数值积分，计算均匀分布下的期望接触力
            unif_result, unif_error = quad(lambda r: integrate_r(r,const_F) * dist_unif.pdf(r), r_min, r_max)

            x_plot = np.linspace(r_min, r_max, 100)
            plt.plot(x_plot, dist_unif.pdf(x_plot), '-', linewidth=3.0)
            return unif_result

        # 计算所有粒径范围的期望接触力
        for i in range(len(dem_velocity)):
            # 展开数据用于拟合
            field_x = np.linspace(radius_min[i], radius_max[i], 8)
            field_n = np.array([19, 65, 60, 38, 20, 16, 8, 7])
            plt.bar(field_x, field_n/((field_x[1] - field_x[0]) * field_n.sum()), 0.5*(field_x[1]-field_x[0]), color='C0')

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
            #print(f"截断正态分布 TruncNorm:\t mu={mu:.4f}, sigma={sigma:.4f}, a={a:.4f}, b={b:.4f}")
            # 执行数值积分，计算均匀分布下的期望接触力
            norm_result, norm_error = quad(lambda r: integrate_r(r,const_F) * dist_norm.pdf(r)/diff_cdf, r_min, r_max)

            x_plot = np.linspace(r_min, r_max, 100)
            plt.plot(x_plot, dist_norm.pdf(x_plot)/diff_cdf, '-', linewidth=3.0)
            return norm_result
  
        # 计算所有粒径范围的期望接触力
        for i in range(len(dem_velocity)):
            # 展开数据用于拟合
            field_x = np.linspace(radius_min[i], radius_max[i], 8)
            field_n = np.array([19, 65, 60, 38, 20, 16, 8, 7])
            plt.bar(field_x, field_n/((field_x[1] - field_x[0]) * field_n.sum()), 0.5*(field_x[1]-field_x[0]), color='C0')

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
            #print(f"截断Weibull右偏 TruncWeibullMin_r:\t c={c_wei_r:.4f}, loc={loc_wei_r:.4f}, scale={scale_wei_r:.4f}, a={a_r:.4f}, b={b_r:.4f}")
            # 执行数值积分，计算均匀分布下的期望接触力
            weibull_r_result, weibull_r_error = quad(lambda r: integrate_r(r,const_F) * weibull_r_dist.pdf(r)/diff_cdf, r_min, r_max)

            x_plot = np.linspace(r_min, r_max, 100)
            plt.plot(x_plot, weibull_r_dist.pdf(x_plot)/diff_cdf, '-', linewidth=3.0)

            return weibull_r_result

        # 计算所有粒径范围的期望接触力
        for i in range(len(dem_velocity)):
            # 展开数据用于拟合
            field_x = np.linspace(radius_min[i], radius_max[i], 8)
            field_n = np.array([19, 65, 60, 38, 20, 16, 8, 7])
            plt.bar(field_x, field_n/((field_x[1] - field_x[0]) * field_n.sum()), 0.5*(field_x[1]-field_x[0]), color='C0')

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
            #print(f"截断Weibull左偏 TruncWeibullMin_l:\t c={c_wei_l:.4f}, loc={loc_wei_l:.4f}, scale={scale_wei_l:.4f}, a={a_l:.4f}, b={b_l:.4f}")
            # 执行数值积分，计算均匀分布下的期望接触力
            weibull_l_result, weibull_l_error = quad(lambda r: integrate_r(r,const_F) * weibull_l_dist.pdf(r)/diff_cdf, r_min, r_max)
            
            x_plot = np.linspace(r_min, r_max, 100)
            plt.plot(x_plot, weibull_l_dist.pdf(x_plot)/diff_cdf, '-', linewidth=3.0)

            return weibull_l_result

        # 计算所有粒径范围的期望接触力
        for i in range(len(dem_velocity)):
            # 展开数据用于拟合
            field_x = np.linspace(radius_min[i], radius_max[i], 8)
            field_n = np.array([19, 65, 60, 38, 20, 16, 8, 7])
            plt.bar(field_x, field_n/((field_x[1] - field_x[0]) * field_n.sum()), 0.5*(field_x[1]-field_x[0]), color='C0')

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
            #print(f"截断指数分布:\t b_fit={b_fit:.4f}, loc_fit={loc_fit:.4f}, scale_fit={scale_fit:.4f})
            # 执行数值积分，计算均匀分布下的期望接触力
            expo_result, expo_error = quad(lambda r: integrate_r(r,const_F) * dist_expo.pdf(r)/diff_cdf, r_min, r_max)
            
            x_plot = np.linspace(r_min, r_max, 100)
            plt.plot(x_plot, dist_expo.pdf(x_plot)/diff_cdf, '-', linewidth=3.0)

            return expo_result

        # 计算所有粒径范围的期望接触力
        for i in range(len(dem_velocity)):
            # 展开数据用于拟合
            field_x = np.linspace(radius_min[i], radius_max[i], 8)
            field_n = np.array([19, 65, 60, 38, 20, 16, 8, 7])
            plt.bar(field_x, field_n/((field_x[1] - field_x[0]) * field_n.sum()), 0.5*(field_x[1]-field_x[0]), color='C0')

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

    plt.show()

    return velocity_y, F_single_min, F_single_max, E_Fmax


def compute_gamma_space(Pier_shape):
        # 根据桥墩形状设置 gamma_space
    if Pier_shape == 'round':
        gamma_space = np.pi / 4
    elif Pier_shape == 'square':
        gamma_space = 1
    else:    
        raise ValueError('Shape Of Section Not Found!')
    return gamma_space

def generate_waveform(wave_type, num_points=2400, amplitude=1.0):
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
    
    elif wave_type == 'sawtooth':
        return amplitude * x

    elif wave_type == 'trapezoidal':
        num_rise = round(num_points / 4)
        rise = np.linspace(0, amplitude, num_rise)
        const = np.full(num_points - 2*num_rise, amplitude)
        fall = np.linspace(amplitude, 0, num_rise)
        return np.concatenate((rise, const, fall),axis=0)

    
    elif wave_type in ('exponential', 'shock'):
        b = -np.log(0.02)/1.0
        return amplitude * np.exp(-b*x)

    elif wave_type == 'gaussian':
        return amplitude * np.exp(-20 * (x - 0.5) ** 2)
    
    else:
        raise ValueError(f"Unsupported wave type: {wave_type}")

def compute_gamma_time(wave_type,t_contact,delta_t_DEMs,amplitude=1, num_points=5000):
    """生成波形序列并求叠加"""
    num_waves = np.maximum(np.ceil(t_contact / delta_t_DEMs), 1).astype(int)
    time_step = t_contact / num_points
    max_total_waves = np.zeros_like(t_contact)

    for i in range(len(t_contact)):
        if num_waves[i] == 1:
            max_total_waves[i] = 1
            continue
        
        shift_points = np.maximum(np.round(delta_t_DEMs[i] / time_step[i]), 1).astype(int)
        total_points = (num_waves[i] - 1) * shift_points + num_points
        time_values = np.arange(total_points) * time_step[i]

        base_wave = generate_waveform(wave_type, num_points, amplitude)
        total_wave = np.zeros(total_points)
        waveforms = []
        # 另一种方法为卷积实现，但当shift_points很大时，计算效率不如循环，可能造成资源浪费
        for j in range(num_waves[i]):
            wave = np.zeros(total_points)
            start = j * shift_points
            end = min(start + num_points, total_points)
            length = end - start
            wave[start:end] = base_wave[:length]
            waveforms.append(wave)
            total_wave += wave

        max_total_waves[i] = np.max(total_wave)
        
    # 绘制前几条单个波形
    for k in range(len(waveforms)):
        plt.plot(time_values, waveforms[k], alpha=0.6, label=f'Wave {k+1}')
    # 绘制叠加波形
    plt.plot(time_values, total_wave, '-', linewidth=3.0, label='Sum')
    plt.legend()
    plt.show()
        
    return max_total_waves


if __name__ == '__main__':
    # 参数定义

    # This study
    case_number = 1
    DEM_Volumn = 0.1 * np.ones(case_number)# np.linspace(1000, 16000, case_number)      # 碎屑流方量：m^3

    #  Prticle size: 0.3-0.6: 16000m^3方量：20m；8000m^3方量：13.5-14.5m/12.7m/s；4000m^3方量：6.4-8.3m/12m/s；2000m^3方量：3.9-4.9m/11m/s；1000m^3方量：2.9-3.45m/10.8m/s
    #  Prticle size: 0.6-1.2: 16000m^3方量：20m；8000m^3方量：12m；4000m^3方量：8m；2000m^3方量：4m；1000m^3方量：2.4m
    #  Prticle size: 0.3-1.2: 16000m^3方量：20m；8000m^3方量：12m；4000m^3方量：8m；2000m^3方量：4m；1000m^3方量：2.4m
    DEM_velocity = 2.6 * np.ones(case_number)  # (11.8 + (9.8-11.8)/(8000-1000) * (DEM_Volumn-1000))     # m/s
    DEM_density = 2500 * np.ones(case_number)      # kg/m3  花岗岩密度2500kg/m3
    DEM_modulus = 55e9 * np.ones(case_number)      # Pa   花岗岩弹性模量50-100GPa
    DEM_miu = 0.25 * np.ones(case_number)          # Poisson's ratio  花岗岩泊松比0.1-0.3
    DEM_strength = 30e6 * np.ones(case_number)     # 花岗岩强度 Pa

    # c_radius = np.array([0.45,0.75,1.05])
    # r_radius = np.array([0.01,0.05,0.15])
    # radius_min = np.repeat(c_radius, 3) - np.tile(r_radius, 3)  # m
    # radius_max = np.repeat(c_radius, 3) + np.tile(r_radius, 3)  # m
    radius_min = 0.010/2*np.ones(case_number)
    radius_max = 0.010/2*np.ones(case_number)

    ratio_solid = np.pi/6.0 * np.ones(case_number) # 固相体积分数0.61-0.68
    impact_angle_deg = 90 * np.ones(case_number)   # 冲击角度 °
    wave_type = 'triangle'     # 脉冲型式：'sine'，'triangle'，'square'，'sawtooth'，'gaussian', 'exponential'/'shock','trapezoidal'
    dist_type = 'uniform'  # 'uniform','normal','exponential','weibull_l','weibull_r'


    # Pier_shape = 'square'
    Pier_shape = 'square'
    Pier_width = 0.2 * np.ones(case_number)        # m
    Pier_modulus = 3e9 * np.ones(case_number)    # Pa 混凝土弹性模量:31GPa
    Pier_miu = 0.2 * np.ones(case_number)          # 混凝土Poisson's ratio ：0.2
    Pier_strength = 30e6 * np.ones(case_number)          # Pa C30混凝土强度:30 MPa
    
    # 调整半径
    radius_min, radius_max = adjust_radius(radius_min, radius_max)
    DEM_depth = 0.037 * np.ones(case_number)  # compute_DEM_depth(DEM_Volumn), This function is for Yaoheba Rock Avalanche only.
    DEM_volume_flux = compute_effective_volume_flux(Pier_width, DEM_depth, radius_max, DEM_velocity)    # m^3/s

    sigma_y = np.minimum(DEM_strength, Pier_strength)

    # 计算冲击时间
    t_contact_elastic, t_contact_elastoplastic = compute_elastoplastic_t_contact(DEM_density, DEM_modulus, DEM_miu, radius_max, DEM_velocity, Pier_modulus, Pier_miu, sigma_y)
    #print('t_contact_elastic =', np.round(t_contact_elastic,9), 's')
    #print('t_contact_elastoplastic =', np.round(t_contact_elastoplastic,9), 's')

    # 计算等效弹性模量
    modulus_equ = 1 / ((1-Pier_miu**2) / Pier_modulus + (1-DEM_miu**2) / DEM_modulus)

    # Hertz弹性接触理论计算冲击力（接触力）
    force_min, force_max, force_equ, force_average = compute_Hertz_contact_forces(radius_min, radius_max, modulus_equ, DEM_density, DEM_velocity)
    #print('[Hertz Elastic Theory]: ', '\n\tF_min=', np.round(force_min/1000,3), '\n\tF_max=',np.round(force_max/1000,3),'\n\tF_average=',np.round(force_average/1000,3),'kN')

    # Thornton弹性-理想塑性接触理论计算冲击力（接触力）
    #print('[Thornton Elasto-Plastic Theory]: ')
    v_y, F_min, F_max, E_Fmax = compute_Thornton_contact_force(radius_min, radius_max, modulus_equ, DEM_density, DEM_velocity, sigma_y, dist_type)
    #print(f'\tF_min = {np.round(F_min/1000,3)}, \n\tF_max = {np.round(F_max/1000,3)}, \n\tforce_average = {np.round(E_Fmax/1000,3)}', 'kN')

    flow_time = np.ones_like(E_Fmax)  # s
    flow_volume_total = DEM_volume_flux * flow_time
    radius_avg = np.sqrt((radius_max**2 + radius_max*radius_min + radius_min**2)/3)
    DEM_impact_rate = np.round(ratio_solid * flow_volume_total / (4/3 * np.pi * (radius_avg**3 )))
    delta_t_DEMs = flow_time / DEM_impact_rate
    num_waves = np.maximum(np.ceil(t_contact_elastoplastic / delta_t_DEMs), 1).astype(int)

    # 碰撞过程时间离散性和总冲击力

    gamma_space = compute_gamma_space(Pier_shape)
    gamma_time  = compute_gamma_time(wave_type,t_contact_elastoplastic,delta_t_DEMs)
    angle_impact = np.sin(np.radians(impact_angle_deg))

    total_force = gamma_time * gamma_space * angle_impact * E_Fmax

    print('DEM_impact_rate=',  np.array2string(DEM_impact_rate,              separator=', ', precision=1), 's^{-1}')
    print('DEM_Volumn    =',   np.array2string(DEM_Volumn,                   separator=', ', precision=1), 'm^3')      
    print('DEM_depth     =',   np.array2string(DEM_depth,                    separator=', ', precision=1), 'm')      
    print('ratio_solid   =',   np.array2string(ratio_solid,                  separator=', ', precision=2), ' ')      
    print('DEM_volume_flux =', np.array2string(DEM_volume_flux,              separator=', ', precision=1), 'm^3/s')      
    print('num_waves     =',   np.array2string(num_waves,                    separator=', ', precision=1), ' ')      
    print('delta_t_DEMs  =',   np.array2string(delta_t_DEMs*1000,            separator=', ', precision=3), 'ms')      
    print('t_contact     =',   np.array2string(t_contact_elastoplastic*1000, separator=', ', precision=3), 'ms')      
    print('DEM_velocity  =',   np.array2string(DEM_velocity,                 separator=', ', precision=5), 'm/s')      
    print('radius_min    =',   np.array2string(radius_min*1000,              separator=', ', precision=1), 'mm')      
    print('radius_max    =',   np.array2string(radius_max*1000,              separator=', ', precision=1), 'mm')      
    print('F_min         =',   np.array2string(F_min/1.000,                  separator=', ', precision=2), 'N')      
    print('F_max         =',   np.array2string(F_max/1.000,                  separator=', ', precision=2), 'N')      
    print('E_Fmax        =',   np.array2string(E_Fmax/1.000,                 separator=', ', precision=2), 'N')      
    print('total_force   =',   np.array2string(total_force/1.000,            separator=', ', precision=2), 'N')

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
