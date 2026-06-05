'''
# 该程序的功能为批量自由变换裁剪图像，适用于高速视频帧画面的批量变换及裁剪
# 通过打开图片后点击裁剪图片需要的4个坐标
# 画面的输入格式为jpg，输出格式为png，可以自定义修改。
'''

import cv2
import numpy as np
from pathlib import Path

# 图片列表
image_files = [f for f in Path(".").iterdir() if f.suffix.lower() in [".jpg", ".jpeg"]]
image_files.sort()
if not image_files:
    print("当前目录没有图片")
    exit()

def order_points(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect = np.zeros((4,2), dtype=np.float32)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def draw_points(img, pts):
    canvas = img.copy()
    for i, (x, y) in enumerate(pts):
        cv2.circle(canvas, (int(x), int(y)), 5, (0,0,255), -1)
        cv2.putText(canvas, str(i+1), (int(x)+10, int(y)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255),2)
    if len(pts)==4:
        poly = np.array(pts, dtype=np.int32)
        cv2.polylines(canvas, [poly], True, (255,0,0), 1)
    return canvas

def perspective_crop(img, pts):
    pts = order_points(pts)
    (tl, tr, br, bl) = pts
    widthA = np.linalg.norm(br-bl)
    widthB = np.linalg.norm(tr-tl)
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr-br)
    heightB = np.linalg.norm(tl-bl)
    maxHeight = int(max(heightA, heightB))
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts,dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warped

first_points = None

for idx, img_path in enumerate(image_files):
    print(f"\n处理: {img_path.name}")
    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("读取失败")
        continue

    # 使用字典封装，避免 nonlocal
    state = {'points': first_points.copy() if first_points is not None else [], 'dragging_idx': -1}

    win_name = "Select 4 Points"

    def mouse_callback(event, x, y, flags, param):
        pts = param['points']
        idx_drag = param['dragging_idx']
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, (px, py) in enumerate(pts):
                if np.hypot(x-px, y-py)<15:
                    param['dragging_idx'] = i
                    return
            if len(pts)<4:
                pts.append([x,y])
        elif event == cv2.EVENT_MOUSEMOVE:
            if param['dragging_idx']>=0:
                pts[param['dragging_idx']] = [x,y]
        elif event == cv2.EVENT_LBUTTONUP:
            param['dragging_idx'] = -1

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, mouse_callback, state)

    while True:
        display = draw_points(img, state['points'])
        if len(state['points'])==4:
            preview = perspective_crop(img, state['points'])
            h1, w1 = display.shape[:2]
            h2, w2 = preview.shape[:2]
            scale = h1 / h2
            preview_resized = cv2.resize(preview, (int(w2*scale), h1))
            combined = np.hstack([display, preview_resized])
        else:
            combined = display

        cv2.putText(combined,
                    "Left click: add | Drag: move | Enter: confirm | Esc: exit",
                    (20,40), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2,cv2.LINE_AA)
        cv2.imshow(win_name, combined)
        key = cv2.waitKey(20)
        if key == 13:  # Enter
            if len(state['points'])==4:
                break
        elif key == 27:
            cv2.destroyAllWindows()
            exit()

    if first_points is None:
        first_points = state['points'].copy()

    warped = perspective_crop(img, state['points'])
    output_name = img_path.stem + "_crop.png"
    cv2.imencode(".png", warped)[1].tofile(output_name)
    print(f"保存: {output_name}")
    cv2.imshow("Result", warped)
    cv2.waitKey(500)

cv2.destroyAllWindows()
print("\n全部处理完成")