'''
# 该程序用于读取高速视频
# 提取关键帧画面
# 支持帧画面图像裁剪
# 支持帧画面图像增强
# 进行RAFT光流分析
'''

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large, Raft_Small_Weights, raft_small


# ====================== 配置区域 ======================
USE_VIDEO = False                    # True: 从视频读取帧    False: 直接读取图片
VIDEO_BACKEND = "decord"            # "decord"在 USE_VIDEO=True 时生效

# ================== 文件路径 ==================
VIDEO_PATH = '105-tower60m4500vol8.3T-0.1share8395g-2254-2880-3rd.avi'

FRAME1_PATH = 'frame_002461_crop.jpg'
FRAME2_PATH = 'frame_002462_crop.jpg'
FRAME_1ST = 2461
FRAME_2ND = 2462

crop_frame = False     # 画面裁剪
scale_frame = False    # 画面缩放
pad_frame = True      # 画面补充/RAFT要求
enhance_frame = False # 图像增强

# ================== 帧参数 ==================
FRAME_RATE = 120.0
PIXELS_PER_METER = 893/0.6  # 1461 / 0.6
DT = (FRAME_2ND - FRAME_1ST) / FRAME_RATE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RAFT_model = 'large'  # 'large' or  'small'

# ================== cv2帧图像增强处理参数 ==================
ENHANCE_PARAMS = {
    'clip_limit': 2.0,          # CLAHE 图像增强算法中对比度限制（越大增强越强，建议 2.0~6.0）
    'tile': 16,                 # CLAHE 图像增强算法中的分块大小（tileGridSize），越小越局部增强，通常8或16
    'gamma': 1.0,               # Gamma 校正系数（>1 变亮，<1 变暗）
    'sharpen': True,            # 是否开启锐化（提升颗粒纹理）
    'denoise_strength': 2       # 去噪强度（fastNlMeansDenoisingColored 的 h 参数），范围10~25，越大去噪越强，但可能丢失细节
}

POSTPROCESS_THRESHOLD = 0.1  # 用于过滤小于速度峰值该倍数的速度场，数值范围0-1.0
RAW_SCALE, RAW_WIDTH, RAW_SPACE = 600, 0.005, 30  # 用于绘制速度场箭头的参数，scale越大，箭头越短， width越大，箭头越粗, space越大，箭头间距越大
X_COOR, Y_COOR, W_WIDTH, H_HEIGHT = 660, 140, 3120, 1680  # 用于矩形裁剪的参数
ENHANCE_FIG_SIZE = (8, 10)
RAFT_FIG_SIZE = (12, 8)

# ====================== 视频读取函数 ======================
def read_frame_decord(video_path, frame_idx):
    """Decord 读取指定帧（推荐，速度快）"""
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0))
        frame = vr[frame_idx].asnumpy()                    # RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)     # 转为 BGR
        return frame
    except ImportError:
        print("未安装 decord，请执行: pip install decord")
        return None
    except Exception as e:
        print(f"Decord 读取错误: {e}")
        return None

def load_two_frames():
    """加载两帧：支持图片模式和视频模式"""
    if not USE_VIDEO:
        # 图片模式
        frame_a = cv2.imread(FRAME1_PATH)
        frame_b = cv2.imread(FRAME2_PATH)
        print(" 从图片文件读取两帧")
    else:
        # 视频模式
        print(f" 从视频读取帧 | 后端: {VIDEO_BACKEND} | 帧号: {FRAME_1ST}, {FRAME_2ND}")
        if VIDEO_BACKEND == "decord":
            frame_a = read_frame_decord(VIDEO_PATH, FRAME_1ST)
            frame_b = read_frame_decord(VIDEO_PATH, FRAME_2ND)
        else:
            print("不支持的 VIDEO_BACKEND")
            return None, None

    if frame_a is None or frame_b is None:
        print("帧读取失败，请检查路径或安装对应库")
        return None, None

    return frame_a, frame_b

def crop_image(img, x=0, y=0, w=None, h=None):
    """
        img: 输入图像 (numpy array)
        x, y: 左上角坐标 (默认0,0)
        w, h: 裁剪宽度和高度 (默认裁到图像右下角)
    """
    if img is None:
        raise ValueError("输入图像为空")
    
    height, width = img.shape[:2]
    
    # 设置默认值（裁到图像边界）
    w = w if w is not None else width - x
    h = h if h is not None else height - y
    
    # 防止越界
    x = max(0, int(x))
    y = max(0, int(y))
    w = min(int(w), width - x)
    h = min(int(h), height - y)
    
    # 执行裁剪
    cropped = img[y:y+h, x:x+w]
    
    print(f"裁剪完成: 位置({x},{y}) 尺寸({w}×{h})")
    return cropped

# ====================== 图像处理函数 ======================
def pad_to_multiple_of_8(img):
    if img is None:
        raise ValueError("输入图像为空")
    h, w = img.shape[:2]
    new_h = ((h + 7) // 8) * 8
    new_w = ((w + 7) // 8) * 8
    pad_bottom = new_h - h
    pad_right = new_w - w
    return cv2.copyMakeBorder(img, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)


def enhance_for_optical_flow(img, **kwargs):
    params = {**ENHANCE_PARAMS, **kwargs}
    denoised = cv2.fastNlMeansDenoisingColored(
        img, None, h=params['denoise_strength'], hColor=params['denoise_strength'],
        templateWindowSize=7, searchWindowSize=21
    )
    
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=params['clip_limit'], tileGridSize=(params['tile'], params['tile']))
    l_clahe = clahe.apply(l)
    
    l_gamma = cv2.LUT(l_clahe, np.array([((i / 255.0) ** (1 / params['gamma'])) * 255 
                                       for i in range(256)]).astype("uint8"))
    
    enhanced = cv2.cvtColor(cv2.merge((l_gamma, a, b)), cv2.COLOR_LAB2BGR)
    
    if params['sharpen']:
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=5)
    return enhanced


def postprocess_flow(u, v, magnitude, threshold):
    mask = magnitude > threshold * magnitude.max()
    u_filtered = u.copy()
    v_filtered = v.copy()
    u_filtered[~mask] = 0
    v_filtered[~mask] = 0
    u_filtered = cv2.medianBlur(u_filtered.astype(np.float32), 5)
    v_filtered = cv2.medianBlur(v_filtered.astype(np.float32), 5)
    return u_filtered, v_filtered


# ====================== 可视化函数 ======================
def visualize_enhancement_comparison(frame_a_orig, frame_a_enh, frame_b_orig, frame_b_enh):
    fig, axes = plt.subplots(2, 2, figsize=ENHANCE_FIG_SIZE)

    axes[0, 0].imshow(cv2.cvtColor(frame_a_orig, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f'Frame {FRAME_1ST} - Original')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(cv2.cvtColor(frame_a_enh, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f'Frame {FRAME_1ST} - Enhanced')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(cv2.cvtColor(frame_b_orig, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f'Frame {FRAME_2ND} - Original')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(cv2.cvtColor(frame_b_enh, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f'Frame {FRAME_2ND} - Enhanced')
    axes[1, 1].axis('off')

    plt.suptitle('Brightness & Contrast Enhancement Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout(pad=2.0)
    plt.show()


def visualize_results(frame_a_color, frame_b_color, u_phys, v_phys, magnitude_phys, u, v, magnitude):
    fig = plt.figure(figsize=RAFT_FIG_SIZE)

    max_colorbar = magnitude_phys.max()

    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(cv2.cvtColor(frame_a_color, cv2.COLOR_BGR2RGB))
    ax1.set_title(f'Frame {FRAME_1ST}')
    ax1.axis('off')

    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(cv2.cvtColor(frame_b_color, cv2.COLOR_BGR2RGB))
    ax2.set_title(f'Frame {FRAME_2ND}')
    ax2.axis('off')

    ax3 = plt.subplot(2, 3, 3)
    im3 = ax3.imshow(magnitude_phys, cmap='RdBu_r', origin='lower', vmin=0.00, vmax=max_colorbar)
    ax3.set_title('Velocity Magnitude (m/s)')
    ax3.invert_yaxis()
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    ax3.axis('off')

    ax4 = plt.subplot(2, 3, 4)
    step = RAW_SPACE
    h, w = magnitude_phys.shape
    y_idx, x_idx = np.mgrid[0:h:step, 0:w:step]
    ax4.imshow(cv2.cvtColor(frame_b_color, cv2.COLOR_BGR2RGB), origin='upper')
    quiv = ax4.quiver(x_idx, y_idx, u[y_idx, x_idx], v[y_idx, x_idx],
                      magnitude_phys[y_idx, x_idx], cmap='RdBu_r', scale=RAW_SCALE, width=RAW_WIDTH)
    quiv.set_clim(0.00, max_colorbar)
    ax4.set_title('Velocity Vector Field')
    plt.colorbar(quiv, ax=ax4, fraction=0.046, pad=0.04)

    ax5 = plt.subplot(2, 3, 5)
    im5 = ax5.imshow(u_phys, cmap='RdBu_r', origin='lower', vmin=-max_colorbar, vmax=max_colorbar)
    ax5.set_title('Horizontal Velocity u (m/s)')
    ax5.invert_yaxis()
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    ax5.axis('off')

    ax6 = plt.subplot(2, 3, 6)
    im6 = ax6.imshow(v_phys, cmap='RdBu_r', origin='lower', vmin=-max_colorbar, vmax=max_colorbar)
    ax6.set_title('Vertical Velocity v (m/s)')
    ax6.invert_yaxis()
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
    ax6.axis('off')

    plt.suptitle('RAFT Optical Flow - Wet-Avalanche Flow Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout(pad=2.0)
    plt.show()


# ====================== 主函数 ======================
def main():
    print(f"使用设备: {DEVICE}")
    print(f"dt = {DT:.5f} s | pixels_per_meter = {PIXELS_PER_METER:.2f} px/m")

    # 1. 统一读取两帧
    frame_a_origin, frame_b_origin = load_two_frames()
    if frame_a_origin is None or frame_b_origin is None:
        print(f"帧画面读取失败")

    # 保存读取到的原始帧
    #cv2.imwrite(f"frame_{FRAME_1ST}_origin.jpg", frame_a_origin)
    #cv2.imwrite(f"frame_{FRAME_2ND}_origin.jpg", frame_b_origin)
    print(f"读取成功，原始尺寸: {frame_a_origin.shape}")

    # 大图像裁剪
    if crop_frame == True:
        frame_a_color = crop_image(frame_a_origin, x=X_COOR, y=Y_COOR, w=W_WIDTH, h=H_HEIGHT)
        frame_b_color = crop_image(frame_b_origin, x=X_COOR, y=Y_COOR, w=W_WIDTH, h=H_HEIGHT)
    if crop_frame == False:
        frame_a_color = frame_a_origin
        frame_b_color = frame_b_origin

    # 图像尺寸缩放
    if scale_frame == True:
        scale = 0.5
        target_width = int(frame_a_color.shape[1] * scale)
        target_height = int(frame_a_color.shape[0] * scale)
        
        frame_a_color = cv2.resize(frame_a_color, (target_width, target_height), interpolation=cv2.INTER_AREA)
        frame_b_color = cv2.resize(frame_b_color, (target_width, target_height), interpolation=cv2.INTER_AREA)
    
        print(f"缩放后输入 RAFT 的尺寸: {frame_a_color.shape} ")

    # 2. 填充到8的倍数
    if pad_frame == True:
        frame_a_color = pad_to_multiple_of_8(frame_a_color)
        frame_b_color = pad_to_multiple_of_8(frame_b_color)
        print(f"填充后尺寸: {frame_a_color.shape}")

    # 3. 图像增强
    if enhance_frame == True:
        frame_a_color = enhance_for_optical_flow(frame_a_color)
        frame_b_color = enhance_for_optical_flow(frame_b_color)

        print("图像增强完成")

        # 显示增强对比
        visualize_enhancement_comparison(frame_a_origin, frame_a_color,
                                       frame_b_origin, frame_b_color)


    # 4. RAFT 输入准备
    frame_a = cv2.cvtColor(frame_a_color, cv2.COLOR_BGR2RGB)
    frame_b = cv2.cvtColor(frame_b_color, cv2.COLOR_BGR2RGB)

    frame_a_tensor = torch.from_numpy(frame_a).permute(2, 0, 1).float() / 255.0
    frame_b_tensor = torch.from_numpy(frame_b).permute(2, 0, 1).float() / 255.0

    frame_a_tensor = frame_a_tensor.unsqueeze(0).to(DEVICE)
    frame_b_tensor = frame_b_tensor.unsqueeze(0).to(DEVICE)

    # 5. RAFT 光流计算
    if RAFT_model == 'large' or RAFT_model == 'Large' or RAFT_model == 'LARGE':
        print("加载 RAFT Large 模型...")
        weights = Raft_Large_Weights.DEFAULT
        model = raft_large(weights=weights).to(DEVICE)

    if RAFT_model == 'small' or RAFT_model == 'Small' or RAFT_model == 'SMALL' :
        print("加载 RAFT Small 模型...")   # 改成 Small
        weights = Raft_Small_Weights.DEFAULT
        model = raft_small(weights=weights).to(DEVICE)

    model.eval()

    with torch.no_grad():
        flow_list = model(frame_a_tensor, frame_b_tensor, num_flow_updates=20)
        flow = flow_list[-1]

    print("RAFT 光流计算完成")

    # 6. 提取 & 后处理
    u = flow[0, 0].cpu().numpy()
    v = -flow[0, 1].cpu().numpy()
    magnitude = np.sqrt(u**2 + v**2)

    u, v = postprocess_flow(u, v, magnitude, threshold=POSTPROCESS_THRESHOLD)

    # 7. 物理单位转换
    u_phys = u / DT / PIXELS_PER_METER
    v_phys = v / DT / PIXELS_PER_METER
    magnitude_phys = magnitude / DT / PIXELS_PER_METER
    max_magnitude = magnitude_phys.max()
    min_magnitude = magnitude_phys.min()
    mean_magnitude = magnitude_phys.mean()

    print(f"最大速度: {max_magnitude:.3f} m/s")
    print(f"最小速度: {min_magnitude:.3f} m/s")
    print(f"平均速度: {mean_magnitude:.3f} m/s")

    # 8. 可视化
    visualize_results(frame_a_color, frame_b_color, u_phys, v_phys, magnitude_phys, u, v, magnitude)

    # 9. 保存
    '''
    plt.savefig(f'RAFT_Analysis_{FRAME_1ST}_{FRAME_2ND}.png', dpi=300, bbox_inches='tight')
    np.savez_compressed(f'RAFT_flow_data_{FRAME_1ST}_{FRAME_2ND}.npz',
                        u_phys=u_phys, v_phys=v_phys, 
                        magnitude_phys=magnitude_phys, frame_b=frame_b)
    
    '''
    print("\n RAFT 分析完成！")



if __name__ == "__main__":
    main()