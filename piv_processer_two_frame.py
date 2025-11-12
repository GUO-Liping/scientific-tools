# -*- coding: utf-8 -*-
from openpiv import tools, pyprocess, filters, scaling
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 检查路径 ---
print("Current dir:", os.getcwd())

# --- 读取两帧 ---
frame_a = tools.imread('frame_0130.jpg')
frame_b = tools.imread('frame_0140.jpg')

print("frame_a shape:", frame_a.shape)
plt.imshow(frame_a, cmap='gray')
plt.title('Check Frame A')
plt.show()

# --- PIV 核心计算 ---
u, v, sig2noise = pyprocess.extended_search_area_piv(
    frame_a, frame_b,
    window_size=32, overlap=16, dt=0.01,
    search_area_size=32, sig2noise_method='peak2peak'
)

x, y = pyprocess.get_coordinates(frame_a.shape, 32, 16)

# --- 检查数值 ---
print("u range:", np.nanmin(u), np.nanmax(u))
print("v range:", np.nanmin(v), np.nanmax(v))
print("mean SNR:", np.nanmean(sig2noise))

# --- 过滤 ---
flags = sig2noise < 0.3
u, v = filters.replace_outliers(u, v, flags, method='localmean', max_iter=3, kernel_size=2)

# --- 单位转换 ---
x, y, u, v = scaling.uniform(x, y, u, v, scaling_factor=1)  #0.000396341

# --- 矢量图 ---
plt.figure(figsize=(8, 6))
plt.imshow(frame_a, cmap='gray', origin='lower',
           extent=[x.min(), x.max(), y.min(), y.max()])
plt.quiver(x, y, u, v, color='r', scale=None)
plt.title('Velocity Field (OpenPIV)')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.tight_layout()
plt.show()

# --- 速度场 ---
speed = np.sqrt(u**2 + v**2)
plt.figure(figsize=(8, 6))
plt.imshow(speed, cmap='jet', origin='lower',
           extent=[x.min(), x.max(), y.min(), y.max()])
plt.colorbar(label='Speed [m/s]')
plt.title('Velocity Magnitude Field')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.tight_layout()
plt.show()
