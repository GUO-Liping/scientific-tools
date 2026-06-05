import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
import torchvision.transforms.functional as F

# 设置中文字体（防止 matplotlib 乱码）
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def preprocess_image(img_path):
    """读取并预处理图像，使其符合 RAFT 模型的输入要求"""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 转换为 PyTorch Tensor，并归一化到 [-1, 1] 范围（RAFT 模型的输入标准）
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
    img_tensor = (img_tensor / 255.0) * 2.0 - 1.0

    # RAFT 要求图像高宽必须是 8 的倍数，进行向下裁剪或填充
    h, w = img_tensor.shape[1:]
    h_new = h - (h % 8)
    w_new = w - (w % 8)
    img_tensor = img_tensor[:, :h_new, :w_new]

    return img_tensor.unsqueeze(0), img[:, :h_new, :w_new]


def flow_to_hsv(flow):
    """将二维光流场 (u, v) 映射到科学通用的 HSV 颜色空间"""
    u, v = flow[0], flow[1]
    magnitude, angle = cv2.cartToPolar(u, v)

    # 初始化 HSV 图像
    hsv = np.zeros((u.shape[0], u.shape[1], 3), dtype=np.uint8)

    # 色调 (H) 代表运动方向：角度 [0, 360] 映射到 [0, 180]
    hsv[..., 0] = angle * 180 / np.pi / 2
    # 饱和度 (S) 保持最大
    hsv[..., 1] = 255
    # 亮度 (V) 代表运动速度：速度幅值归一化映射到 [0, 255]
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # 转回 RGB 格式供 matplotlib 绘制
    rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb_flow, magnitude


def plot_scientific_results(img1, rgb_flow, magnitude, flow, step=16):
    """科学规范地绘制光流分析结果"""
    u, v = flow[0], flow[1]
    h, w = u.shape

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 原始第一帧画面
    axes[0, 0].imshow(img1)
    axes[0, 0].set_title("原始参考帧 (Frame 1)", fontsize=14)
    axes[0, 0].axis("off")

    # 2. 密集光流 HSV 色学可视化（常用于表现运动趋势与边界）
    axes[0, 1].imshow(rgb_flow)
    axes[0, 1].set_title("RAFT 密集光流场 (HSV 编码)", fontsize=14)
    axes[0, 1].axis("off")

    # 3. 速度幅值热力图（定量分析速度大小分布）
    im = axes[1, 0].imshow(magnitude, cmap="jet")
    axes[1, 0].set_title("运动速率绝对值分布 (Magnitude)", fontsize=14)
    axes[1, 0].axis("off")
    fig.colorbar(
        im, ax=axes[1, 0], orientation="horizontal", pad=0.05, label="像素位移 / 帧"
    )

    # 4. 稀疏矢量箭头图（Quiver Plot，流体力学/动作力学经典表达）
    # 创建稀疏采样网格，避免箭头过密无法分辨
    x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))
    x_sparse = x[::step, ::step]
    y_sparse = y[::step, ::step]
    u_sparse = u[::step, ::step]
    v_sparse = v[::step, ::step]

    # 在第一帧背景上叠加矢量箭头
    axes[1, 1].imshow(img1, alpha=0.7)
    # 倒置 Y 轴使 matplotlib 的坐标系与图像坐标系对齐
    axes[1, 1].quiver(
        x_sparse,
        y_sparse,
        u_sparse,
        -v_sparse,
        color="lime",
        angles="xy",
        scale_units="xy",
        scale=0.5,
        width=0.0025,
    )
    axes[1, 1].set_title(f"位移矢量箭头图 (采样步长: {step}px)", fontsize=14)
    axes[1, 1].set_xlim(0, w)
    axes[1, 1].set_ylim(h, 0)  # 反转y轴匹配图像
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig("raft_optical_flow_analysis.png", dpi=300)
    plt.show()


def main(img1_path, img2_path):
    # 1. 检查运行设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"正在使用设备: {device}")

    # 2. 加载 PyTorch 官方内置的 RAFT Large 模型
    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights, progress=True).to(device)
    model.eval()

    # 3. 数据加载与预处理
    print("正在读取并转换图像...")
    img1_tensor, img1_rgb = preprocess_image(img1_path)
    img2_tensor, _ = preprocess_image(img2_path)

    img1_tensor = img1_tensor.to(device)
    img2_tensor = torch.to(device) if hasattr(torch, 'to') else img2_tensor.to(device)

    # 4. 模型推理计算光流
    print("正在利用 RAFT 计算密集光流...")
    with torch.no_grad():
        # RAFT 推理会返回迭代过程中的所有流，我们取最后一次迭代（最精准）的结果
        list_of_flows = model(img1_tensor, img2_tensor)
        predicted_flow = list_of_flows[-1].squeeze(0).cpu().numpy()  # 形状: (2, H, W)

    # 5. 分析结果转换为科学表现形式
    print("计算完成，正在生成科学可视化图表...")
    rgb_flow, magnitude = flow_to_hsv(predicted_flow)

    # 6. 绘图展示
    plot_scientific_results(img1_rgb, rgb_flow, magnitude, predicted_flow, step=16)


if __name__ == "__main__":
    # 使用时请将下面路径替换为你本地的连续帧图像路径
    image_frame1 = "frame_002459_crop.jpg"
    image_frame2 = "frame_002460_crop.jpg"

    main(image_frame1, image_frame2)