import os
import numpy as np
import matplotlib.pyplot as plt
from openpiv import tools, pyprocess, validation, filters, scaling

# --- 检查路径 --- 
print("Current dir:", os.getcwd())

# 1. 读取两帧图像
frame_a = tools.imread('frames_192.168.8.47_top/frame_0130.jpg')
frame_b = tools.imread('frames_192.168.8.47_top/frame_0150.jpg')

# 2. 计算PIV速度场
winsize = 32
searchsize = 32
overlap = 16
dt = 0.02
threshold = 1.0
pixels_per_meter = 2523.191094619666
u_phy = 200 # pixels/s
v_phy = 0 # pixels/s

u, v, sig2noise = pyprocess.extended_search_area_piv(
    frame_a.astype(np.int32),
    frame_b.astype(np.int32),
    window_size=winsize,
    overlap=overlap,
    dt=dt,
    search_area_size=searchsize,
    sig2noise_method='peak2peak'
)
mask = sig2noise < threshold
x, y = pyprocess.get_coordinates(frame_a.shape, winsize, overlap)
u, v = filters.replace_outliers(u, v, mask, method='localmean', max_iter=3, kernel_size=2)

mask_u = abs(u) > u_phy; u[mask_u] = 0
mask_v = v > v_phy; v[mask_v] = 0

# 3. 可视化 - 四图合一（2x2）
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
plt.tight_layout(pad=3.0)

# 图1：Frame A
axes[0, 0].imshow(frame_a, cmap='gray')
axes[0, 0].set_title('Frame A')

# 图2：Frame B
axes[0, 1].imshow(frame_b, cmap='gray')
axes[0, 1].set_title('Frame B')

# 图3：Velocity Field (quiver)

axes[1, 0].imshow(frame_b, cmap='gray', origin='lower')
axes[1, 0].quiver(x, y, u, v, color='r', scale=None)
axes[1, 0].invert_yaxis()
axes[1, 0].set_title('Velocity Field (OpenPIV)')

# 图4：Magnitude Map（速度大小）
x, y, u, v = scaling.uniform(x, y, u, v, scaling_factor=pixels_per_meter)
velocity_magnitude = np.sqrt(u**2 + v**2)
im = axes[1, 1].imshow(velocity_magnitude, cmap='jet', origin='lower', extent=[x.min(), x.max(), y.min(), y.max()])
axes[1, 1].set_title('Velocity Magnitude')
axes[1, 1].invert_yaxis()
fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

plt.show()
