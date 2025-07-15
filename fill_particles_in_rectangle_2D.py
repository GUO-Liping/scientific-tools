import numpy as np
import matplotlib.pyplot as plt

# 设置矩形区域大小
width = 2.2  # 矩形宽度
height = 3.0  # 矩形高度

# 颗粒直径范围
min_diameter = 0.6
max_diameter = 2.4

# 填充参数
max_particles = 1e5       # 最大尝试填入的颗粒数
max_attempts = 1e5       # 最大尝试次数（防止死循环）
particle_list = []

def generate_random_particle():
    d = np.random.uniform(min_diameter, max_diameter)
    r = d / 2
    x = np.random.uniform(r, width - r)
    y = np.random.uniform(r, height - r)
    return x, y, r

def is_overlap(x, y, r, particles):
    for px, py, pr in particles:
        dist = np.sqrt((x - px) ** 2 + (y - py) ** 2)
        if dist <= r + pr:
            return True
    return False

# 粒子填充主循环
attempts = 0
while len(particle_list) < max_particles and attempts < max_attempts:
    x, y, r = generate_random_particle()
    if not is_overlap(x, y, r, particle_list):
        particle_list.append((x, y, r))
    attempts += 1
    print('attempt = ', attempts)

# 输出结果
print(f"成功填充的颗粒数量: {len(particle_list)}")

# 可视化
# 设置颜色映射
cmap = plt.cm.Blues  # 使用 viridis 颜色映射，可以更换为其他颜色映射，如 plt.cm.plasma
min_radius = min_diameter / 2  # 最小半径
max_radius = max_diameter / 2  # 最大半径


fig, ax = plt.subplots()
ax.set_aspect('equal')

for x, y, r in particle_list:
    norm_radius = (r - min_radius) / (max_radius - min_radius)  # 归一化半径到 [0, 1]
    color_map = cmap(norm_radius)  # 获取颜色
    circle = plt.Circle((x, y), r, fill=True, facecolor=color_map, edgecolor='k', linewidth=1.0)
    ax.add_patch(circle)

plt.xlim(0, width)
plt.ylim(0, height)
plt.title(f"Filled particles: {len(particle_list)}")
plt.xlabel("Width")
plt.ylabel("Height")
plt.grid(True)

plt.show()
