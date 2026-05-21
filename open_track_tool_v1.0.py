import cv2
import numpy as np
import matplotlib.pyplot as plt

# ========================= 参数设置 =========================
video_path = "105-tower60m4500vol8.3T-0.1share8395g-1778-2379-2nd.avi"
cross_size = 35

positions = {}                     # {帧号: (x, y)}
show_all = False                   # 控制是否显示所有标记点

# ========================= 辅助函数 =========================
def draw_cross(img, x, y, size=cross_size, color=(0, 0, 255), thickness=3):
    """绘制十字光标"""
    cv2.line(img, (x - size, y), (x + size, y), color, thickness)
    cv2.line(img, (x, y - size), (x, y + size), color, thickness)


def draw_all_points(img, current_frame_idx):
    """绘制所有已标记点"""
    for f_idx, (x, y) in positions.items():
        color = (0, 255, 0) if f_idx == current_frame_idx else (0, 200, 100)
        cv2.circle(img, (x, y), 8, color, 2)
        cv2.putText(img, str(f_idx), (x + 15, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


# ========================= 鼠标回调 =========================
def mouse_event(event, x, y, flags, param):
    orig = param['orig']
    current_idx = param['frame_idx']

    if event == cv2.EVENT_LBUTTONDBLCLK:
        h, w = orig.shape[:2]
        ox = int(x * w / param['display_shape'][0])
        oy = int(y * h / param['display_shape'][1])
        
        positions[current_idx] = (ox, oy)
        print(f"✓ 第 {current_idx} 帧 双击标记: ({ox}, {oy})")

        img_copy = orig.copy()
        if show_all:                              # 尊重 show_all 设置
            draw_all_points(img_copy, current_idx)
        draw_cross(img_copy, x, y)
        cv2.imshow("Frame", img_copy)


# ========================= 主程序 =========================
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("无法打开视频！")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"视频分辨率: {video_w}×{video_h}  总帧数: {total_frames}")

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
cv2.resizeWindow("Frame", 1600, int(1600 * video_h / video_w))

frame_idx = 0

try:
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            frame_idx = max(0, frame_idx - 1)
            continue

        param = {
            'orig': frame.copy(),
            'display_shape': (frame.shape[1], frame.shape[0]),
            'frame_idx': frame_idx
        }

        display_frame = frame.copy()
        
        # ★★★ 关键修复：根据 show_all 决定是否显示所有点 ★★★
        if show_all:
            draw_all_points(display_frame, frame_idx)

        cv2.imshow("Frame", display_frame)
        cv2.setMouseCallback("Frame", mouse_event, param)

        print(f"\n第 {frame_idx}/{total_frames-1} 帧 | "
              f"左键双击= 标记 | S = 显示/隐藏所有点 | Z = 撤销当前标记 | "
              f"Enter = 下一帧 | Backspace = 上一帧 | ESC = 退出")

        key = cv2.waitKey(0) & 0xFF

        if key == 27 or cv2.getWindowProperty("Frame", cv2.WND_PROP_VISIBLE) < 1:
            print("程序退出")
            break
        elif key == 13:                                 # Enter → 下一帧
            frame_idx = min(frame_idx + 1, total_frames - 1)
        elif key == 8 or key == 127:                    # Backspace → 上一帧
            frame_idx = max(0, frame_idx - 1)
        elif key == ord('s') or key == ord('S'):        # S 键切换
            show_all = not show_all
            print(f"→ 显示所有标记点: {'开启' if show_all else '关闭'}")
        elif key == ord('z') or key == ord('Z'):        # Z 撤销
            if frame_idx in positions:
                del positions[frame_idx]
                print(f"已撤销第 {frame_idx} 帧的标记")

except Exception as e:
    print("程序异常:", e)

finally:
    cap.release()
    cv2.destroyAllWindows()
    for _ in range(5):
        cv2.waitKey(1)

# ========================= 最终结果 =========================
if positions:
    sorted_frames = sorted(positions.items())
    frame_list = [f for f, p in sorted_frames]
    pos_array = np.array([p for f, p in sorted_frames])

    plt.figure(figsize=(12, 8))
    x, y = pos_array[:, 0], pos_array[:, 1]
    plt.scatter(x, y, color='red', s=80, label='标记位置')
    plt.plot(x, y, color='blue', linestyle='--', linewidth=2.5, label='轨迹')
    
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.xlabel("X 像素")
    plt.ylabel("Y 像素")
    plt.title("轨迹 - 手动标记")
    plt.legend()
    plt.grid(True, alpha=0.6)
    plt.tight_layout()
    plt.show()

    data = np.column_stack((frame_list, x, y))
    np.savetxt("bullet_positions.csv", data, delimiter=",", 
               header="frame,x,y", comments='', fmt='%d')
    print(f"已保存 {len(positions)} 个标记点")
else:
    print("未标记任何点")