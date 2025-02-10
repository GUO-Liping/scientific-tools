import numpy as np
import pyvista as pv

# 定义参数
pi = np.pi
cos = np.cos
sin = np.sin
R = 150
r = 3

# 计算 beta 和 gamma 角度范围
beta_max = 16 * (2 * np.pi)
beta = np.linspace(0.0 * pi, beta_max, 500)
gamma = np.linspace(0.0 * pi, 2.0 * pi, 500)

# 计算网环捻角
phi = np.arctan(beta_max * r / (R * 2 * np.pi))
print('网环捻角为', phi, 'rad', '(', phi * 180 / np.pi, '°)')

# 计算中心环形路径
x0 = R * cos(gamma)
y0 = R * sin(gamma)
z0 = np.zeros_like(x0)

# 计算多个偏移路径
xi_values = [i * 2 * np.pi / 6 for i in range(1, 7)]
paths = []
for xi in xi_values:
    x = (R + r * sin(xi + beta)) * cos(gamma)
    y = (R + r * sin(xi + beta)) * sin(gamma)
    z = r * cos(xi + beta)
    paths.append((x, y, z))

# 创建 PyVista Plotter
plotter = pv.Plotter()

# 绘制中心管状曲线
tube_radius = 0.5 * r
spline0 = pv.Spline(np.column_stack((x0, y0, z0)), 500)
tube0 = spline0.tube(radius=tube_radius)
plotter.add_mesh(tube0, color="blue")

# 绘制其他管状曲线
colors = ["red", "green", "yellow", "cyan", "magenta", "orange"]
colors = ["white", "white", "white", "white", "white", "white"]
for (x, y, z), color in zip(paths, colors):
    spline = pv.Spline(np.column_stack((x, y, z)), 500)
    tube = spline.tube(radius=tube_radius)
    plotter.add_mesh(tube, color=color)

# 显示绘图
plotter.show()
