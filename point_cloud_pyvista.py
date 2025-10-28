import numpy as np
import pyvista as pv

# 1. 读取 xyz 文件
data = np.loadtxt("example.xyz")  # 格式：x y z
x, y, z = data[:, 0], data[:, 1], data[:, 2]

# 2. 构建点云
points = np.column_stack((x, y, z))

# 3. 转换为 PyVista 点云对象
cloud = pv.PolyData(points)

# 4. 通过插值生成曲面 (使用 delaunay2D)
surface = cloud.delaunay_2d()

# 5. 可视化（可选）
plotter = pv.Plotter()
plotter.add_mesh(surface, cmap="terrain", show_edges=True)
plotter.show()

# 6. 导出为 STL 文件
surface.save("example_surface.stl")

print("✅ STL 曲面已成功导出：example_surface.stl")
