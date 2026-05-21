import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from openpiv import tools, pyprocess, validation, filters, scaling

# ====================== 配置部分 ======================
print("Current dir:", os.getcwd())      # 打印当前工作目录，便于检查路径

video_path = 'DJI_20260427191850_0320_D.mp4'                    
frame_1st = 4335                         # 第一帧的帧号（从0开始计数）
frame_2nd = 4336                         # 第二帧的帧号（高速视频通常取连续或间隔很小的两帧）
frame_rate = 120.0

# PIV 参数设置
winsize = 32                            # 相关窗口大小（像素），必须是2的倍数
searchsize = 48                         # 搜索窗口大小，建议大于winsize，提高相关性成功率
overlap = 16                            # 窗口重叠量，overlap越大，矢量点越密集
dt = (frame_2nd-frame_1st)/frame_rate   # 两帧之间的时间间隔（单位：秒）

sig2noise_threshold = 1.5              # 信噪比阈值，低于此值的向量会被视为无效
pixels_per_meter = 881/0.6              # 像素到距离单位的转换系数
print(f"dt = {dt:.4f} s | pixels_per_meter = {pixels_per_meter:.1f} px/m")

# ====================== 1. 读取高速视频的两帧画面 ======================
cap = cv2.VideoCapture(video_path)      # 打开视频文件
if not cap.isOpened():                  # 检查视频是否成功打开
    print("错误：无法打开视频文件！")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"视频分辨率: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}  总帧数: {total_frames}")
# 读取第一帧
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_1st)   # 设置读取位置到指定帧
ret, frame_a_color = cap.read()               # 读取彩色图像

# 读取第二帧
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_2nd)
ret, frame_b_color = cap.read()

cap.release()                           # 释放视频资源，节省内存

# 检查是否成功读取到图像
if frame_a_color is None or frame_b_color is None:
    print("错误：读取帧失败！请检查帧号是否超出视频总长度。")
    exit()

# 保存两帧的彩色原图
cv2.imwrite(f'frame_{frame_1st}.png', frame_a_color)
cv2.imwrite(f'frame_{frame_2nd}.png', frame_b_color)

# 转为灰度
frame_a = cv2.cvtColor(frame_a_color, cv2.COLOR_BGR2GRAY)
frame_b = cv2.cvtColor(frame_b_color, cv2.COLOR_BGR2GRAY)

# ====================== 对比度增强（CLAHE） ======================
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
frame_a = clahe.apply(frame_a)
frame_b = clahe.apply(frame_b)
print("应用CLAHE增强对比度")

# ====================== 2. 执行 PIV 计算 ======================
u, v, sig2noise = pyprocess.extended_search_area_piv(
    frame_a.astype(np.int32),           # 输入第一帧（转为int32）
    frame_b.astype(np.int32),           # 输入第二帧（转为int32）
    window_size=winsize,                # 窗口大小
    overlap=overlap,                    # 重叠量
    dt=dt,                              # 时间间隔
    search_area_size=searchsize,        # 搜索区域大小
    sig2noise_method='peak2peak'        # 信噪比计算方法
)

# 获取每个矢量对应的空间坐标
x, y = pyprocess.get_coordinates(
    image_size=frame_a.shape,           # 图像尺寸
    search_area_size=searchsize,
    overlap=overlap
)

# ====================== 3. 后处理（非常重要） ======================
# 3.1 使用信噪比过滤低质量向量
invalid_mask = validation.sig2noise_val(sig2noise, threshold=sig2noise_threshold)
u[invalid_mask] = np.nan                # 将低质量点设为NaN
v[invalid_mask] = np.nan

# 3.2 使用局部均值替换异常值（插值）
u, v = filters.replace_outliers(
    u, v, 
    flags=invalid_mask,                 # 必须传入flags参数
    method='localmean',                 # 使用局部均值插值
    max_iter=5,                         # 最大迭代次数
    kernel_size=3                       # 核大小
)

# 3.3 全局速度范围过滤（防止出现物理上不可能的极端速度）
global_mask = validation.global_val(u, v, 
                                    u_thresholds=(-800, 800),   # 水平速度合理范围
                                    v_thresholds=(-400, 400))   # 垂直速度合理范围

u[global_mask] = np.nan
v[global_mask] = np.nan

# 再次进行异常值替换，使矢量场更平滑
u, v = filters.replace_outliers(
    u, v, 
    flags=global_mask,
    method='localmean', 
    max_iter=3, 
    kernel_size=3
)

# ====================== 4. 可视化结果（2x2布局） ======================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))   # 创建2行2列的子图
plt.tight_layout(pad=4.0)                          # 调整子图间距

# 图1：原始第一帧
axes[0, 0].imshow(frame_a, cmap='gray')
axes[0, 0].set_title('Frame A')

# 图2：原始第二帧
axes[0, 1].imshow(frame_b, cmap='gray')
axes[0, 1].set_title('Frame B')

# 图3：速度矢量场叠加在第二帧上
axes[1, 0].imshow(frame_b, cmap='gray', origin='lower')
axes[1, 0].quiver(x, y, u, v, color='red', scale=1000, width=0.003)
axes[1, 0].set_title('Velocity Field')
axes[1, 0].invert_yaxis()                       # 翻转Y轴以匹配图像坐标

# 图4：速度大小云图（单位：m/s）
x_phys, y_phys, u_phys, v_phys = scaling.uniform(
    x, y, u, v, scaling_factor=pixels_per_meter
)

velocity_magnitude = np.sqrt(u_phys**2 + v_phys**2)   # 计算速度模长

im = axes[1, 1].imshow(velocity_magnitude, cmap='jet', origin='lower',
                       extent=[x_phys.min(), x_phys.max(), y_phys.min(), y_phys.max()])
axes[1, 1].set_title('Velocity Magnitude (m/s)')
axes[1, 1].invert_yaxis()
fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)  # 添加色条

plt.show()

print("PIV 处理完成！")