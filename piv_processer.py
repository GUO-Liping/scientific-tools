# -*- coding: utf-8 -*-
# @Author: Liping Guo
# @Time: 2025/11/11
# @Function: piv analysis of frames extracted from a video 


import os
import numpy as np
import matplotlib.pyplot as plt
from openpiv import tools, pyprocess, validation, filters, scaling

# -------------------- 用户配置 --------------------
#frame_folder = "frames_192.168.8.22_front"  # 帧图片路径
frame_folder = "frames_192.168.8.47_top"  # 帧图片路径
output_folder = "piv_results + frame_folder"                          # 结果保存路径
os.makedirs(output_folder, exist_ok=True)

frame_prefix = "frame_"   # 图片前缀
frame_ext = ".jpg"        # 图片后缀
frame_step = 10           # 图片间隔帧
video_fps = 1000          # 原始视频帧率
dt = frame_step/video_fps # 帧间时间间隔（秒），根据实际帧率设定

piv_interval = 10          # PIV 处理每隔几个画面图片
window_size = 32          # PIV 计算图像灰度位移的探测窗口(像素)大小,2 的幂次。
overlap = 16              # PIV 相邻窗口之间的重叠像素数,决定探测窗口中心的密度，一般取窗口值的1/2
threshold_snr = 1.3   # 信噪比阈值

# -------------------- 获取帧序列 --------------------
piv_frames = sorted([f for f in os.listdir(frame_folder) if f.startswith(frame_prefix) and f.endswith(frame_ext)])

# -------------------- 循环计算 PIV --------------------
for i in range(0, len(piv_frames)-piv_interval, piv_interval):
    img1_path = os.path.join(frame_folder, piv_frames[i])
    img2_path = os.path.join(frame_folder, piv_frames[i + piv_interval])

    frame_a = tools.imread(img1_path)
    frame_b = tools.imread(img2_path)

    # -------------------- PIV 处理 --------------------
    u, v, sig2noise = pyprocess.extended_search_area_piv(
        frame_a.astype(np.int32),
        frame_b.astype(np.int32),
        window_size=window_size,
        overlap=overlap,
        dt=dt,
        search_area_size=window_size,
        sig2noise_method='peak2peak'
    )

    x, y = pyprocess.get_coordinates(frame_a.shape, window_size, overlap)


    # 生成 flags：True 表示异常向量
    flags = sig2noise < threshold_snr

    # 使用 replace_outliers 替换异常向量
    u, v = filters.replace_outliers(u, v, flags, method='localmean', max_iter=3, kernel_size=2)

    # -------------------- 保存 PIV 数据 --------------------
    result_file = os.path.join(output_folder, f"piv_{i:04d}.npz")
    np.savez(result_file, x=x, y=y, u=u, v=v, sig2noise=sig2noise, flags=flags)

    # -------------------- 可视化 --------------------
    plt.figure(figsize=(10, 8))
    plt.imshow(frame_a, cmap='rainbow')
    plt.quiver(x, y, u, -v, color='r', scale=50)
    plt.title(f"PIV Result Frame {i} -> {i+piv_interval}")
    plt.savefig(os.path.join(output_folder, f"piv_{i:04d}.png"))
    plt.close()

print("✅ PIV 分析完成，结果保存在:", output_folder)
