import cv2
import numpy as np
import matplotlib.pyplot as plt

# ========================= 参数设置 =========================
video_path = "103-tower60m1500vol8.3T-0.1share8395g-2129-2755-1st.avi"
cross_size = 20                   # 十字光标大小（像素）
positions = {}                    # 存储标记点: {帧号: (原始x, 原始y)}

# 缩放与平移控制（全局状态）
zoom_factor = 1.0                  # 当前缩放倍率
pan_x = 0                          # 水平平移偏移量（基于原图尺度）
pan_y = 0                          # 垂直平移偏移量（基于原图尺度）
is_panning = False                 # 右键抓手是否激活
last_mouse_x = 0
last_mouse_y = 0
keep_zoom = False                  # 切换帧时是否保持当前缩放状态

# 共享给鼠标回调的动态上下文
runtime_context = {
    'orig_frame': None,
    'current_idx': 0,
    'zoom_info': None
}

# ========================= 辅助函数 =========================
def draw_cross(img, x, y, size=cross_size, color=(0, 0, 255), thickness=2):
    """在显示图像上绘制十字光标"""
    cv2.line(img, (x - size, y), (x + size, y), color, thickness)
    cv2.line(img, (x, y - size), (x, y + size), color, thickness)


def get_zoomed_frame(frame, zoom, pan_x=0, pan_y=0):
    """根据缩放倍率和平移量裁剪并缩放图像，保持视口物理分辨率不变"""
    h, w = frame.shape[:2]
    
    # 1. 计算裁剪窗口的大小
    crop_w = max(10, int(w / zoom))
    crop_h = max(10, int(h / zoom))
    
    # 2. 计算理想的裁剪中心（默认中心 + 平移量）
    cx = w // 2 + pan_x
    cy = h // 2 + pan_y
    
    # 3. 计算裁剪边界
    x1 = cx - crop_w // 2
    y1 = cy - crop_h // 2
    
    # 4. 边界越界限制与修正
    if x1 < 0: x1 = 0
    if y1 < 0: y1 = 0
    if x1 + crop_w > w: x1 = w - crop_w
    if y1 + crop_h > h: y1 = h - crop_h
    
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    cropped = frame[y1:y2, x1:x2]
    
    # 5. 将裁剪区域缩放回原图大小以铺满窗口（免去处理画布黑边的麻烦）
    resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    zoom_info = {
        'crop_x1': x1, 'crop_y1': y1,
        'crop_w': crop_w, 'crop_h': crop_h,
        'disp_w': w, 'disp_h': h
    }
    return resized, zoom_info


def refresh_display():
    """集中刷新画面并绘制数据"""
    orig = runtime_context['orig_frame']
    if orig is None: return
    
    current_idx = runtime_context['current_idx']
    
    # 生成变换后的图
    display_frame, zoom_info = get_zoomed_frame(orig, zoom_factor, pan_x, pan_y)
    runtime_context['zoom_info'] = zoom_info  # 实时更新给回调使用
    
    # 绘制所有已标记点（把原图坐标正确投射到当前显示视口）
    for f_idx, (ox, oy) in positions.items():
        # 判断点是否在当前裁剪视口内
        if (zoom_info['crop_x1'] <= ox <= zoom_info['crop_x1'] + zoom_info['crop_w'] and
            zoom_info['crop_y1'] <= oy <= zoom_info['crop_y1'] + zoom_info['crop_h']):
            
            # 计算比例并映射回显示视口的(w, h)
            rx = (ox - zoom_info['crop_x1']) / zoom_info['crop_w']
            ry = (oy - zoom_info['crop_y1']) / zoom_info['crop_h']
            dx = int(rx * zoom_info['disp_w'])
            dy = int(ry * zoom_info['disp_h'])
            
            color = (0, 255, 0) if f_idx == current_idx else (0, 200, 100)
            cv2.circle(display_frame, (dx, dy), 6, color, -1)
            cv2.putText(display_frame, str(f_idx), (dx + 10, dy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # 如果是当前帧的点，额外加个十字光标强调
            if f_idx == current_idx:
                draw_cross(display_frame, dx, dy, size=15, color=(0, 0, 255), thickness=2)

    cv2.imshow("Frame", display_frame)


# ========================= 鼠标回调函数 =========================
def mouse_event(event, x, y, flags, param):  # 鼠标在 "Frame" 窗口中的xy坐标
    global zoom_factor, pan_x, pan_y, is_panning, last_mouse_x, last_mouse_y

    zoom_info = runtime_context['zoom_info']
    orig = runtime_context['orig_frame']
    current_idx = runtime_context['current_idx']
    
    if orig is None or zoom_info is None: 
        return

    # 获取窗口内实际物理图像渲染区域宽度与高度
    window_rect = cv2.getWindowImageRect("Frame")  # 图像区域左上角在屏幕上的水平、垂直坐标，当前图像区域的实际渲染宽度、高度(x, y, width, height)
    win_w, win_h = (window_rect[2], window_rect[3]) if window_rect[2] > 0 else (zoom_info['disp_w'], zoom_info['disp_h'])

    # 将鼠标在窗口中的坐标 (x,y) 转换成 display_frame 上的像素坐标
    cx = int(x * zoom_info['disp_w'] / win_w)
    cy = int(y * zoom_info['disp_h'] / win_h)

    # ==================== 左键点击：精确保存真实原始坐标 ====================
    if event == cv2.EVENT_LBUTTONDOWN:
        # 基于当前视口的裁剪起始点和缩放比例，精确逆向推导原图坐标
        ox = int(zoom_info['crop_x1'] + (cx / zoom_info['disp_w']) * zoom_info['crop_w'])
        oy = int(zoom_info['crop_y1'] + (cy / zoom_info['disp_h']) * zoom_info['crop_h'])
        
        # 边界安全限制
        ox = max(0, min(orig.shape[1] - 1, ox))
        oy = max(0, min(orig.shape[0] - 1, oy))

        positions[current_idx] = (ox, oy)
        print(f"✓ 第 {current_idx} 帧 标记成功! 原始像素坐标: ({ox}, {oy})")
        refresh_display()

    # ==================== 右键拖拽：平移画面 ====================
    elif event == cv2.EVENT_RBUTTONDOWN:
        is_panning = True
        last_mouse_x, last_mouse_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if is_panning:
            # 计算在窗口上的拖拽位移
            dx = x - last_mouse_x
            dy = y - last_mouse_y
            
            # 转化为原图尺度下的位移（除以 zoom_factor）
            pan_x -= int(dx * (zoom_info['crop_w'] / win_w))
            pan_y -= int(dy * (zoom_info['crop_h'] / win_h))
            
            last_mouse_x, last_mouse_y = x, y
            refresh_display()

    elif event == cv2.EVENT_RBUTTONUP:
        is_panning = False

    # ==================== 滚轮滚动：缩放画面 ====================
    elif event == cv2.EVENT_MOUSEWHEEL:
        # 记录缩放前的鼠标指向的【原图绝对坐标】
        mouse_ox = zoom_info['crop_x1'] + (cx / zoom_info['disp_w']) * zoom_info['crop_w']
        mouse_oy = zoom_info['crop_y1'] + (cy / zoom_info['disp_h']) * zoom_info['crop_h']

        delta = flags >> 16
        if delta > 0:
            zoom_factor = min(zoom_factor * 1.25, 40.0)
        else:
            zoom_factor = max(zoom_factor / 1.25, 0.10) # 不允许缩小到比原图还小

        # 更新缩放后，调整 pan_x/pan_y 使得鼠标指针所指的实际物理位置在缩放前后保持不动（以鼠标为中心缩放）
        new_crop_w = int(zoom_info['disp_w'] / zoom_factor)
        new_crop_h = int(zoom_info['disp_h'] / zoom_factor)
        
        pan_x = int(mouse_ox - (cx / zoom_info['disp_w']) * new_crop_w - zoom_info['disp_w'] // 2)
        pan_y = int(mouse_oy - (cy / zoom_info['disp_h']) * new_crop_h - zoom_info['disp_h'] // 2)
        
        refresh_display()


# ========================= 主程序 =========================
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"无法打开视频文件：{video_path}，请检查路径！")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"视频分辨率: {video_w}×{video_h} | 总帧数: {total_frames}")

# 初始化标准交互窗口
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow("Frame", 1080, int(1080 * video_h / video_w))

# 【核心改进】：在循环外只绑定一次回调，传入可变字典
cv2.setMouseCallback("Frame", mouse_event)

frame_idx = 0

try:
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"无法读取第 {frame_idx} 帧，尝试停留在上一帧。")
            frame_idx = max(0, frame_idx - 1)
            if frame_idx == 0: break
            continue

        if not keep_zoom:
            zoom_factor = 1.0
            pan_x = pan_y = 0

        # 更新共享上下文
        runtime_context['orig_frame'] = frame.copy()
        runtime_context['current_idx'] = frame_idx

        # 刷新渲染
        refresh_display()

        print(f"当前第 {frame_idx} / {total_frames-1} 帧 | 缩放: {zoom_factor:.2f}x | "
              f"【左键】标记 | 【右键拖拽】平移 | 【滚轮】缩放 | "
              f"【Enter】下一帧 | 【Backspace】重置并退回 | 【ESC】退出")

        key = cv2.waitKey(0) & 0xFF

        if key == 27 or cv2.getWindowProperty("Frame", cv2.WND_PROP_VISIBLE) < 1:
            break

        if key == 13:      # Enter：前进
            keep_zoom = True  # 向前看时保持当前的缩放跟视口
            frame_idx = min(frame_idx + 1, total_frames - 1)
            
        elif key == 8:     # Backspace：后退并还原视口
            keep_zoom = False
            frame_idx = max(0, frame_idx - 1)

except Exception as e:
    print("程序运行时出现异常:", e)
    import traceback
    traceback.print_exc()

finally:
    cap.release()
    cv2.destroyAllWindows()
    for _ in range(5): cv2.waitKey(1)

# ========================= 结果可视化与保存 =========================
if positions:
    sorted_frames = sorted(positions.items())
    frame_list = [f for f, p in sorted_frames]
    pos_array = np.array([p for f, p in sorted_frames])

    # 解决 matplotlib 中文乱码问题
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(10, 6))
    x, y = pos_array[:, 0], pos_array[:, 1]
    plt.scatter(x, y, color='red', s=60, label='标记位置', zorder=3)
    plt.plot(x, y, color='blue', linestyle='--', linewidth=1.5, label='运动轨迹', zorder=2)
    
    # 标注帧号
    for f, xi, yi in zip(frame_list, x, y):
        plt.text(xi + 5, yi - 5, f"F-{f}", fontsize=8, color='darkgreen')

    plt.gca().invert_yaxis()  # 图像坐标系 Y 轴向下
    plt.xlabel("X 像素 (Width)")
    plt.ylabel("Y 像素 (Height)")
    plt.title("轨迹重建 - 手动标记数据")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 保存数据
    data = np.column_stack((frame_list, x, y))
    np.savetxt("bullet_positions.csv", data, delimiter=",", 
               header="frame,x,y", comments='', fmt='%d')
    print(f"🎉 成功！已保存 {len(positions)} 个精准标记点至 bullet_positions.csv")
else:
    print("提示：未标记任何有效数据点。")