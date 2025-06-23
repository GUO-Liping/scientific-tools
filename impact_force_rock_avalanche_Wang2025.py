# python code
# 该程序用于计算任意一个矩形框内填充的圆形数量
# 圆的直径可服从常见的随机分布函数
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


def generate_diameter(distribution_type, d_min, d_max, rng, **params):
    """根据分布类型生成符合范围的随机直径"""
    while True:
        if distribution_type == 'uniform':
            diameter = rng.uniform(d_min, d_max)
        elif distribution_type == 'normal':
            diameter = rng.normal(params.get('mean', 0), params.get('stddev', 1))
        elif distribution_type == 'exponential':
            diameter = rng.exponential(params.get('scale', 1))
        elif distribution_type == 'poisson':
            diameter = rng.poisson(params.get('lam', 3))
        elif distribution_type == 'gamma':
            diameter = rng.gamma(params.get('shape', 2), params.get('scale', 1))
        elif distribution_type == 'beta':
            diameter = d_min + (d_max - d_min) * rng.beta(params.get('a', 2), params.get('b', 2))
        else:
            raise ValueError(f"Unsupported distribution type: {distribution_type}")
        
        # 检查直径是否在范围内
        if d_min <= diameter <= d_max:
            return diameter


def is_valid_position(x, y, radius, circles):
    """检查新圆是否与已放置的圆相切或分离（不重叠）"""
    if len(circles) == 0:
        return True  # 初始无圆，直接有效
    
    # 已有圆的圆心和半径
    existing_centers = circles[:, :2]  # (N, 2)
    existing_radii = circles[:, 2]    # (N,)
    
    # 计算新圆与已有圆的距离
    distances = np.sqrt((existing_centers[:, 0] - x)**2 + (existing_centers[:, 1] - y)**2)
    min_distances = distances - (existing_radii + radius)  # 检查是否有重叠
    
    return np.all(min_distances >= 0)  # 只有所有圆都分离时返回 True


def fill_circles(width, height, d_min, d_max, distribution_type, max_attempts=1000, **params):
    """填充圆，确保圆不重叠"""
    circles = []  # 存储圆的信息：x, y, radius
    rng = np.random.default_rng()  # NumPy 的随机数生成器
    
    for _ in range(max_attempts):
        # 根据分布生成直径
        diameter = generate_diameter(distribution_type, d_min, d_max, rng, **params)
        if diameter > width or diameter > height:
            continue
        radius = diameter / 2

        # 随机生成圆心位置
        x = rng.uniform(radius, width - radius)
        y = rng.uniform(radius, height - radius)
        
        # 检查是否有效
        if is_valid_position(x, y, radius, np.array(circles)):
            circles.append((x, y, radius))
    
    global N  # 填充的圆颗粒个数
    N =  len(circles)

    return np.array(circles)

def plot_circles(circles, width, height):
    """绘制所有圆"""
    fig, ax = plt.subplots()
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal', adjustable='box')
    
    for x, y, radius in circles:
        circle = plt.Circle((x, y), radius, edgecolor='black', facecolor='gray', alpha=0.5)
        ax.add_patch(circle)
    
    plt.show()


def print_circle_info(circles):
    """打印圆的信息"""
    diameters = circles[:, 2] * 2  # 计算每个圆的直径

    print(f"填充的圆的数量: {N}")
    # print(f"圆的直径: {diameters}")
    print(f"最小直径: {diameters.min():.2f}, 最大直径: {diameters.max():.2f}, 平均直径: {diameters.mean():.2f}")

    radius_s = diameters/2  # 颗粒半径，国际单位：m
    force_s = 4/3 * (5*np.pi/4)**(3/5) * modulus_equal**(2/5) * DEM_dencity**(3/5) * DEM_velocity**(6/5) * radius_s**2

    print(f"冲击力分别为: {np.round(force_s/1000, 3)} kN, \n冲击力之和为: {np.sum(force_s/1000)} kN")


if __name__ == '__main__':
    # 输入参数——全局变量

    DEM_dencity = 2550  # 颗粒密度，国际单位：kg/m3
    DEM_depth = 0.03  # 颗粒流厚度，0.03~0.05m

    # 桥墩参数
    Pier_width = 0.1
    Pier_modulus = 3.2e9
    sigma_y =30e6  # 圆柱的屈服强度

    # 碎屑颗粒流参数
    radius_min = 4.0e-3
    radius_max = 4.0e-3
    DEM_modulus = 60e9  # 弹性模量，国际单位：Pa
    DEM_velocity = 1.4  # 颗粒速度，1.4~2.9 国际单位：m/s
    #DEM_Volumn = 18  # 碎屑流方量：m^3
    ##DEM_Area = 943.39  # 碎屑流流动区域面积：m^2
    #channel_alpha = np.radians(171)  # 滑槽角度——模型化为三角形，国际单位：degree
    #channel_lenght = 50  # 滑槽长度——模型化为三角形，61.5国际单位：m
    #DEM_depth = np.sqrt(DEM_Volumn / (channel_lenght*np.tan(channel_alpha/2)))  # 16000m^3方量：20m；8000m^3方量：12m；4000m^3方量：8m；2000m^3方量：4m；1000m^3方量：2.4m；
    
    ratio_solid = 0.45
    angle_impact = np.sin(np.radians(72))
    impact_duration = 0.002  # 单个颗粒与桥墩碰撞过程的冲击时间，一般为3至4ms

    # 等效参数
    modulus_equal = 1/(1/Pier_modulus + 1/DEM_modulus)  # 弹性模量，国际单位：Pa
    #radius_up = radius_max
    #radius_low = radius_min + 0.0*(radius_max-radius_min)
    if radius_max == radius_min:
        radius_max = radius_max + 1e-6
    else:
        pass
    radius_e_equ = (1/3 * (radius_max**3-radius_min**3)/(radius_max-radius_min))**(1/2)
    # N = int(np.maximum((Pier_width+2*radius_eq) * (DEM_depth+2*radius_eq) / (2*radius_eq)**2, 1))
    #print('N=', N)
    # ----------------------------------------------------------------------------------------------------------------------------#
    # 填充算法
    # 均匀分布
    #print("\n均匀分布:", '体积=', DEM_Volumn, '粒径范围=', radius_min, '~', radius_max, '冲击厚度=', round(DEM_depth,2))
    #circles_uniform = fill_circles(Pier_width, DEM_depth, 2*radius_min, 2*radius_max, distribution_type='uniform')
    #radius_up = np.max(circles_uniform[:, 2])
    #radius_low = np.min(circles_uniform[:, 2])
    #prob_ST = 1/N + (1-np.exp(-0.005*N))
    #print('radius_e_equ=', radius_e_equ)

    area_effect = Pier_width*DEM_depth
    number_effect =  area_effect/(np.pi*(radius_max**2+radius_min**2)/2)

    print('number_effect=', int(number_effect))
    

    # ----------------------------------------------------------------------------------------------------------------------------#
    # 弹性接触理论
    # 单个颗粒对桥墩的冲击力
    # 单位系统：N
    force_single_e_min = 4/3 * (5*np.pi/4)**(3/5) * (modulus_equal)**(2/5) * radius_min**2 * DEM_dencity**(3/5) * DEM_velocity**(6/5)
    force_single_e_max = 4/3 * (5*np.pi/4)**(3/5) * (modulus_equal)**(2/5) * radius_max**2 * DEM_dencity**(3/5) * DEM_velocity**(6/5)
    force_single_e_equ = 4/3 * (5*np.pi/4)**(3/5) * (modulus_equal)**(2/5) * radius_e_equ**2 * DEM_dencity**(3/5) * DEM_velocity**(6/5)
    force_single_ref1 = 4/3 * (5*np.pi/4)**(3/5) * (20e9)**(2/5) * 0.45**2 * 2400**(3/5) * 5.0**(6/5)
    
    # 碎屑颗粒冲击力
    epr_average_e = 1/3*(radius_max**3-radius_min**3)/(radius_max-radius_min)
    force_average_e = epr_average_e * 4/3 * (5*np.pi/4)**(3/5) * modulus_equal**(2/5) * DEM_dencity**(3/5) * DEM_velocity**(6/5)
    #force_impact_elastic = ratio_Area * sin_theta * force_average_e
    print('Elastic Theory: average contact force=', np.round(force_average_e,3), 'N')
    #print('force_single_e_equ=', round(force_single_e_equ/1000,2), 'kN')
    
    #print('force_impact_elastic=', round(force_impact_elastic/1000,2), 'kN')
    
    # 碎屑颗粒冲击力
    # ----------------------------------------------------------------------------------------------------------------------------#
    # ----------------------------------------------------------------------------------------------------------------------------#
    # 弹性-理想塑性接触理论(Thornton, 1997)
    mass_star = DEM_dencity * 4/3 * np.pi * radius_max**3
    delta_max = (15*mass_star*DEM_velocity**2 / (16*modulus_equal*radius_max**0.5))**0.4
    v_y1 = 3.194*(sigma_y**5 * radius_max**3 / (modulus_equal**4 * mass_star))**0.5
    v_y = 1.56*(sigma_y**5 /(modulus_equal**4 * DEM_dencity))**0.5
    F_y = sigma_y**3 * np.pi**3 * radius_max**2 / (6*modulus_equal**2)
    E_Fy = (radius_max**3-radius_min**3)/(radius_max-radius_min) * (sigma_y**3 * np.pi**3) / (18*modulus_equal**2)
    if DEM_velocity < v_y:
        print('DEM_velocity=',DEM_velocity, ' < v_y=', v_y)

        F_max1 = 4/3*modulus_equal**0.4*radius_max**0.2 * (15*mass_star*DEM_velocity**2/16)**0.6
        F_max = 4/3*modulus_equal**0.4*radius_max**2 * (5*DEM_dencity*np.pi*DEM_velocity**2/4)**0.6
        E_Fmax = 4/9*modulus_equal**0.4 * (5*DEM_dencity*np.pi*DEM_velocity**2/4)**0.6 * (radius_max**3-radius_min**3)/(radius_max-radius_min)

    else:
        print('DEM_velocity=',DEM_velocity, ' >= v_y=', np.round(v_y,3))
        F_max = np.sqrt(F_y**2 + 4/3*np.pi**2*sigma_y*DEM_dencity*(DEM_velocity**2 - v_y**2) * radius_max**4)

        # 定义被积函数
        A = F_y**2
        B = 4/3*np.pi**2*sigma_y*DEM_dencity * (DEM_velocity**2 - v_y**2)
        # 定义被积函数
        def integrand(x, A, B):
            return np.sqrt(A + B * x**4)

        # 使用 quad 进行数值积分
        Int_value, Int_error = quad(integrand, radius_min, radius_max, args=(A, B))
        E_Fmax = 1/(radius_max-radius_min) * Int_value
    print('v_y=', np.round(v_y, 3), 'F_y=', np.round(F_y,3))
    print('Elasto-Plastic Theory: F_max=',np.round(F_max), 'average contact force=', np.round(E_Fmax))
    #print_circle_info(circles_uniform)
    #plot_circles(circles_uniform, Pier_width, DEM_depth)
    
    # ----------------------------------------------------------------------------------------------------------------------------#
    # 碰撞过程中的时间离散性
    # 单位时间1s内穿过有效横截面区域的颗粒数量
    volume_total = area_effect * DEM_velocity * 1
    #number_of_DEM = ratio_solid * volume_total / (4/3*np.pi*(radius_max**3+radius_min**3)/2)
    number_of_DEM = ratio_solid * volume_total / (4/3*np.pi*(radius_max**3+radius_min**3))
    delta_t_DEM = 1/number_of_DEM
    k = int(impact_duration/(2*delta_t_DEM))
    total_force = angle_impact * E_Fmax * (k+1 - k*(k+1)/2 * delta_t_DEM/(0.5*impact_duration))
    print('number_of_DEM=', int(number_of_DEM), 'delta_t_DEM=', delta_t_DEM, 'total_force=',total_force)

    # ----------------------------------------------------------------------------------------------------------------------------#
    
    '''
    # 均匀分布
    print("\n均匀分布:")
    circles_uniform = fill_circles(width, height, d_min, d_max, distribution_type='uniform')
    print_circle_info(circles_uniform)
    
    # 正态分布
    print("\n正态分布:")
    circles_normal = fill_circles(width, height, d_min, d_max, distribution_type='normal', mean=2, stddev=0.5)
    print_circle_info(circles_normal)
    plot_circles(circles_normal, width, height)
    
    # 指数分布
    print("\n指数分布:")
    circles_exponential = fill_circles(width, height, d_min, d_max, distribution_type='exponential', scale=1)
    print_circle_info(circles_exponential)
    plot_circles(circles_exponential, width, height)
    
    # 泊松分布
    print("\n泊松分布:")
    circles_poisson = fill_circles(width, height, d_min, d_max, distribution_type='poisson', lam=2)
    print_circle_info(circles_poisson)
    plot_circles(circles_poisson, width, height)
    
    # Gamma分布
    print("\nGamma分布:")
    circles_gamma = fill_circles(width, height, d_min, d_max, distribution_type='gamma', shape=2, scale=1)
    print_circle_info(circles_gamma)
    plot_circles(circles_gamma, width, height)
    
    # Beta分布
    print("\nBeta分布:")
    circles_beta = fill_circles(width, height, d_min, d_max, distribution_type='beta', a=2, b=2)
    print_circle_info(circles_beta)
    plot_circles(circles_beta, width, height)
    '''