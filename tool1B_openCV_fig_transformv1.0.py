import cv2
import numpy as np
from pathlib import Path

# ==================== 配置区域 ====================
# 请在这里指定你要裁剪的单张图片路径
INPUT_IMAGE_PATH = "your_image.jpg" 
# =================================================

def order_points(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect = np.zeros((4, 2), dtype=np.float32)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def draw_points(img, pts):
    canvas = img.copy()
    for i, (x, y) in enumerate(pts):
        cv2.circle(canvas, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.putText(canvas, str(i + 1), (int(x) + 10, int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    if len(pts) == 4:
        poly = np.array(pts, dtype=np.int32)
        cv2.polylines(canvas, [poly], True, (255, 0, 0), 1)
    return canvas

def perspective_crop(img, pts):
    pts = order_points(pts)
    (tl, tr, br, bl) = pts
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warped

def mouse_callback(event, x, y, flags, param):
    pts = param['points']
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, (px, py) in enumerate(pts):
            if np.hypot(x - px, y - py) < 15:
                param['dragging_idx'] = i
                return
        if len(pts) < 4:
            pts.append([x, y])
    elif event == cv2.EVENT_MOUSEMOVE:
        if param['dragging_idx'] >= 0:
            pts[param['dragging_idx']] = [x, y]
    elif event == cv2.EVENT_LBUTTONUP:
        param['dragging_idx'] = -1

def main():
    img_path = Path(INPUT_IMAGE_PATH)
    if not img_path.exists():
        print(f"错误: 找不到文件 {INPUT_IMAGE_PATH}，请检查路径。")
        return

    print(f"正在处理单张图片: {img_path.name}")
    # 支持中文路径的读取方式
    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("图片读取失败")
        return

    # 运行时状态控制字典
    state = {'points': [], 'dragging_idx': -1}
    win_name = "Select 4 Points & Preview"

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, mouse_callback, state)

    while True:
        display = draw_points(img, state['points'])
        
        # 满足 4 个点时实时拼合右侧预览图
        if len(state['points']) == 4:
            preview = perspective_crop(img, state['points'])
            h1, w1 = display.shape[:2]
            h2, w2 = preview.shape[:2]
            scale = h1 / h2
            preview_resized = cv2.resize(preview, (int(w2 * scale), h1))
            combined = np.hstack([display, preview_resized])
        else:
            combined = display

        # 绘制操作提示文字
        cv2.putText(combined,
                    "Left click: add | Drag: move | Enter: confirm & save | Esc: exit",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow(win_name, combined)
        key = cv2.waitKey(20)
        
        if key == 13:  # Enter 键确认
            if len(state['points']) == 4:
                break
            else:
                print("请先点击画面标出 4 个控制点再按回车！")
        elif key == 27:  # Esc 键退出
            cv2.destroyAllWindows()
            print("操作已被取消。")
            return

    # 执行最终的透视裁剪并保存
    warped = perspective_crop(img, state['points'])
    output_name = img_path.stem + "_crop.png"
    
    # 支持中文路径的保存方式
    cv2.imencode(".png", warped)[1].tofile(output_name)
    print(f"\n裁剪成功！已保存原始画面至: {output_name}")
    
    # 展示最终结果
    cv2.destroyWindow(win_name)
    cv2.imshow("Final Result", warped)
    print("按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()