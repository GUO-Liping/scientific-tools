'''
# 该程序的功能为批量自由变换裁剪图像，适用于高速视频帧画面的批量变换及裁剪
# 要先使用labelme库生成json文件，或者获取裁剪图片需要的4个坐标
# labelme 使用方法，直接在终端输入labelme
# 画面的输入格式为jpg，输出格式为png，可以自定义修改。
'''

import cv2
import numpy as np
import json
from pathlib import Path

# 读取 Labelme JSON
def load_labelme_points(json_path):

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for shape in data["shapes"]:
        if shape.get("shape_type") == "polygon":

            pts = shape["points"]

            if len(pts) == 4:
                return pts

    return None

# 对 Labelme点集进行排序,[左上, 右上, 右下, 左下]
def order_points(pts):
    pts = np.array(pts, dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype="float32")

# 透视裁剪函数
def perspective_crop(img, pts):
    
    # 计算裁剪后图像的宽度和高度
    widthA = np.linalg.norm(pts[2] - pts[3])
    widthB = np.linalg.norm(pts[1] - pts[0])
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(pts[1] - pts[2])
    heightB = np.linalg.norm(pts[0] - pts[3])
    maxHeight = int(max(heightA, heightB))

    # 目标矩形坐标
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(pts, dst)

    # 应用透视变换
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    return warped

if __name__ == "__main__":

    # 支持的图片格式
    image_files = []
    for ext in ("*.png", "*.jpeg"):
        image_files.extend(Path(".").glob(ext))
    image_files = sorted(image_files)

    output_dir = Path("Img_cropped")
    output_dir.mkdir(exist_ok=True)

    label_pts = load_labelme_points("video111_frame461.json")
    print(f'label_pts= {np.round(label_pts, 2)}')
    ordered_pts = order_points(label_pts)

    i = 0
    for img_path in image_files:
        i = i + 1
        img = cv2.imread(str(img_path))
        print(f'[Image {i}] height={img.shape[0]}, width={img.shape[1]}, channel={img.shape[2]}')
        cropped_img = perspective_crop(img, ordered_pts)    
        out_name = output_dir / (img_path.stem + "_crop.jpg")    
        cv2.imwrite(str(out_name), cropped_img)  
        print(f'[Crop  {i}] height={cropped_img.shape[0]}, width={cropped_img.shape[1]}, channel={cropped_img.shape[2]}')  

    print("\n批量透视变换裁剪完成")