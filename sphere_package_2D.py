# python code
# 该程序用于计算任意一个矩形框内填充的圆形数量
# 圆的直径可服从常见的随机分布函数
import numpy as np
import matplotlib.pyplot as plt


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
    global N  # 填充的圆颗粒个数
    N =  len(circles)
    print(f"填充的圆的数量: {N}")
    # print(f"圆的直径: {diameters}")
    print(f"最小直径: {diameters.min():.2f}, 最大直径: {diameters.max():.2f}, 平均直径: {diameters.mean():.2f}")

    radius_s = diameters/2  # 颗粒半径，国际单位：m
    force_s = 4/3 * (5*np.pi/4)**(3/5) * modulus_equal**(2/5) * rho_s**(3/5) * velocity_s**(6/5) * radius_s**2

    print(f"冲击力分别为: {np.round(force_s/1000, 2)} kN, \n冲击力之和为: {np.sum(force_s/1000)} kN")


if __name__ == '__main__':
    # 输入参数——全局变量
    width = 2.2
    height = 4
    d_min = 0.3
    d_max = 2.4
    
    modulus_equal = 20e9  # 弹性模量，国际单位：Pa
    rho_s = 2500  # 颗粒密度，国际单位：kg/m3
    velocity_s = 10  # 颗粒速度，国际单位：m/s
    
    radius_max = d_max/2
    radius_min = d_min/2
    radius_upper = radius_max
    radius_lower = radius_min + 0.92*(radius_max-radius_min)

    # 均匀分布
    print("均匀分布:")
    circles_uniform = fill_circles(width, height, d_min, d_max, distribution_type='uniform')
    print_circle_info(circles_uniform)
    
    estimate_epr = 1/3*(radius_upper**3-radius_lower**3)/(radius_max-radius_min)
    force_estimate = N * estimate_epr * 4/3 * (5*np.pi/4)**(3/5) * modulus_equal**(2/5) * rho_s**(3/5) * velocity_s**(6/5)
    print('estimate_epr=', estimate_epr, 'N=', N, 'force_estimate=', force_estimate/1000)
    
    plot_circles(circles_uniform, width, height)

    '''
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
    