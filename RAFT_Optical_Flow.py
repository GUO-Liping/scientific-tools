import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large, Raft_Small_Weights, raft_small

# ====================== 配置 ======================
video_path = 'DJI_20260427191850_0320_D.mp4' # '126-tower60csphrer2cmdeg37.5m4500rho476vo5.28T0.0share7940g-2104-2980-1st.avi' 
frame_1st = 4336
frame_2nd = 4337
frame_rate = 120.0

print(torch.cuda.is_available())

pixels_per_meter = 881 / 0.6
dt = (frame_2nd - frame_1st) / frame_rate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
print(f"dt = {dt:.5f} s | pixels_per_meter = {pixels_per_meter:.2f} px/m")

# ====================== 1. 读取两帧 ======================
cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_1st)
ret, frame_a_color = cap.read()
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_2nd)
ret, frame_b_color = cap.read()
cap.release()

if frame_a_color is None or frame_b_color is None:
    print("读取帧失败！")
    exit()

print(f"读取成功，尺寸: {frame_a_color.shape}")

# ====================== 2. 亮度与对比度提升 ======================
alpha = 1.0    # 对比度
beta = 0       # 亮度

frame_a_enhanced = cv2.convertScaleAbs(frame_a_color, alpha=alpha, beta=beta)
frame_b_enhanced = cv2.convertScaleAbs(frame_b_color, alpha=alpha, beta=beta)

print(f"图像亮度提升完成 (alpha={alpha}, beta={beta})")

# ====================== 3. 显示提升前后的对比图 ======================
fig_compare, axes = plt.subplots(2, 2, figsize=(12, 10))
plt.tight_layout(pad=3.0)

axes[0, 0].imshow(cv2.cvtColor(frame_a_color, cv2.COLOR_BGR2RGB), origin='upper')
axes[0, 0].set_title(f'Frame {frame_1st} - Original')
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(frame_a_enhanced, cv2.COLOR_BGR2RGB), origin='upper')
axes[0, 1].set_title(f'Frame {frame_1st} - Enhanced')
axes[0, 1].axis('off')

axes[1, 0].imshow(cv2.cvtColor(frame_b_color, cv2.COLOR_BGR2RGB), origin='upper')
axes[1, 0].set_title(f'Frame {frame_2nd} - Original')
axes[1, 0].axis('off')

axes[1, 1].imshow(cv2.cvtColor(frame_b_enhanced, cv2.COLOR_BGR2RGB), origin='upper')
axes[1, 1].set_title(f'Frame {frame_2nd} - Enhanced')
axes[1, 1].axis('off')

plt.suptitle('Brightness & Contrast Enhancement Comparison', fontsize=16, fontweight='bold')
plt.show()

# 转为 RGB 并归一化到 [0,1]（RAFT 输入要求）
frame_a = cv2.cvtColor(frame_a_color, cv2.COLOR_BGR2RGB)
frame_b = cv2.cvtColor(frame_b_color, cv2.COLOR_BGR2RGB)

frame_a_tensor = torch.from_numpy(frame_a).permute(2, 0, 1).float() / 255.0
frame_b_tensor = torch.from_numpy(frame_b).permute(2, 0, 1).float() / 255.0

# 添加 batch 维度
frame_a_tensor = frame_a_tensor.unsqueeze(0).to(device)
frame_b_tensor = frame_b_tensor.unsqueeze(0).to(device)

print("图像加载完成")

# ====================== 2. 加载 RAFT 模型 ======================
print("加载 RAFT Small 模型...")
weights = Raft_Small_Weights.DEFAULT
model = raft_small(weights=weights).to(device)
model.eval()


with torch.no_grad():
    # RAFT 返回多个迭代结果，取最后一个
    flow_list = model(frame_a_tensor, frame_b_tensor)
    flow = flow_list[-1]          # shape: (1, 2, H, W)

print("RAFT 光流计算完成")

# ====================== 3. 提取速度场并转为物理单位 ======================
u = flow[0, 0].cpu().numpy()      # 水平方向 (pixels)
v = flow[0, 1].cpu().numpy()      # 垂直方向 (pixels)

magnitude = np.sqrt(u**2 + v**2)

# 转为物理单位 m/s
u_phys = u / dt / pixels_per_meter
v_phys = v / dt / pixels_per_meter
magnitude_phys = magnitude / dt / pixels_per_meter

print(f"最大速度: {magnitude_phys.max():.3f} m/s")
print(f"平均速度: {magnitude_phys.mean():.3f} m/s")

# ====================== 4. 专业可视化 ======================
fig = plt.figure(figsize=(16, 6))
plt.tight_layout(pad=4.0)

ax1 = plt.subplot(2, 3, 1)
ax1.imshow(cv2.cvtColor(frame_a_color, cv2.COLOR_BGR2RGB))
ax1.set_title(f'Frame {frame_1st}')
ax1.axis('off')

ax2 = plt.subplot(2, 3, 2)
ax2.imshow(cv2.cvtColor(frame_b_color, cv2.COLOR_BGR2RGB))
ax2.set_title(f'Frame {frame_2nd}')
ax2.axis('off')

ax3 = plt.subplot(2, 3, 3)
ax3.imshow(magnitude_phys, cmap='RdBu_r', origin='lower')
ax3.set_title('Velocity Magnitude (m/s)')
plt.colorbar(ax3.images[0], ax=ax3, fraction=0.046, pad=0.04)
ax3.axis('off')

# 彩色矢量场
ax4 = plt.subplot(2, 3, 4)
step = 12
h, w = magnitude.shape
y_idx, x_idx = np.mgrid[0:h:step, 0:w:step]

ax4.imshow(cv2.cvtColor(frame_b_color, cv2.COLOR_BGR2RGB), origin='upper')
quiv = ax4.quiver(x_idx, y_idx, 
                  u[y_idx, x_idx], v[y_idx, x_idx],
                  magnitude[y_idx, x_idx], cmap='RdBu_r',
                  scale=800, width=0.003)
ax4.set_title('Velocity Vector Field')
ax4.invert_yaxis()
plt.colorbar(quiv, ax=ax4, fraction=0.046, pad=0.04)

ax5 = plt.subplot(2, 3, 5)
im5 = ax5.imshow(u_phys, cmap='RdBu_r', origin='lower', vmin=-2, vmax=2)
ax5.set_title('Horizontal Velocity u (m/s)')
plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
ax5.axis('off')

ax6 = plt.subplot(2, 3, 6)
im6 = ax6.imshow(v_phys, cmap='RdBu_r', origin='lower', vmin=-2, vmax=2)
ax6.set_title('Vertical Velocity v (m/s)')
plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
ax6.axis('off')

plt.suptitle('RAFT Optical Flow - Wet-Avalanche Flow Analysis', fontsize=16, fontweight='bold')
plt.show()

# ====================== 保存结果 ======================
plt.savefig(f'RAFT_Analysis_{frame_1st}_{frame_2nd}.png', dpi=300, bbox_inches='tight')

np.savez_compressed(f'RAFT_flow_data_{frame_1st}_{frame_2nd}.npz',
                    u_phys=u_phys, v_phys=v_phys, magnitude_phys=magnitude_phys,
                    frame_b=frame_b)

print("\nRAFT 分析完成！结果已保存。")