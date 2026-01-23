# -*- coding: utf-8 -*-
"""
snow_particles_cellpose_full.py
Cellpose(v3+) 实例分割雪颗粒 + 粒径分布统计（最简可行路线）

依赖：
pip install cellpose opencv-python scikit-image pandas matplotlib

输出：
- particle_sizes.csv
- masks_preview.png
- overlay_preview.png
- psd_hist.png
- psd_cdf.png
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cellpose import models
from skimage.measure import regionprops_table


# =========================
# 0) 用户需要修改的参数
# =========================
IMG_PATH = "snow_particles.jpg"          # <<< 改成你的图片路径
OUT_DIR  = "cellpose_out"      # 输出目录

# 如果你知道像素到毫米的比例，就填一个数（mm/px）
# 例如：PIXEL_TO_MM = 0.02  # 0.02 mm/px
# 不知道就设 None，脚本会输出像素单位粒径分布
PIXEL_TO_MM = None

# 分割相关参数
MODEL_TYPE = "cyto"            # 'cyto' 对颗粒/团块通常更稳
USE_GPU = False                # 没配好CUDA就 False
DIAMETER = None                # 可填一个近似颗粒直径(像素)提升稳定性，例如 25 或 40
MIN_AREA_PX = 30               # 去噪：过滤小连通域（按你的图调整）

# 预处理参数
DO_CLAHE = False               # 光照不均时建议 True
GAUSSIAN_BLUR_K = 3            # 0/1 表示不模糊；3/5表示轻微模糊


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def preprocess_gray(gray: np.ndarray) -> np.ndarray:
    """最简单、稳健的预处理：轻度模糊 + 可选CLAHE"""
    g = gray.copy()

    if DO_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g = clahe.apply(g)

    if GAUSSIAN_BLUR_K and GAUSSIAN_BLUR_K >= 3 and GAUSSIAN_BLUR_K % 2 == 1:
        g = cv2.GaussianBlur(g, (GAUSSIAN_BLUR_K, GAUSSIAN_BLUR_K), 0)

    return g


def masks_to_color(masks: np.ndarray) -> np.ndarray:
    """把实例标签图转伪彩色预览图"""
    if masks.max() == 0:
        return np.zeros((*masks.shape, 3), dtype=np.uint8)
    m = (masks.astype(np.float32) / masks.max() * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(m, cv2.COLORMAP_TURBO)
    return color


def overlay_masks_on_image(gray: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """把mask边界叠加在灰度图上（红色边界）"""
    # 灰度转BGR
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 找边界：mask梯度
    kernel = np.ones((3, 3), np.uint8)
    m_bin = (masks > 0).astype(np.uint8) * 255
    edge = cv2.morphologyEx(m_bin, cv2.MORPH_GRADIENT, kernel)
    edge = (edge > 0)

    overlay = base.copy()
    overlay[edge] = (0, 0, 255)  # BGR: red
    return overlay


def main():
    ensure_dir(OUT_DIR)

    # =========================
    # 1) 读图 -> 灰度
    # =========================
    img_bgr = cv2.imread(IMG_PATH)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {IMG_PATH}")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_p = preprocess_gray(gray)

    # =========================
    # 2) Cellpose v3+ 实例分割
    # =========================
    model = models.CellposeModel(model_type=MODEL_TYPE, gpu=USE_GPU)

    # 注意：CellposeModel.eval 返回 (masks, flows, styles)
    masks, flows, styles = model.eval(
        gray_p,
        diameter=DIAMETER,
        channels=[0, 0]   # 灰度图
    )

    print(f"[INFO] Segmentation done. N_instances = {int(masks.max())}")

    # =========================
    # 3) 区域属性 -> 粒径（等效直径）
    # =========================
    props = regionprops_table(masks, properties=("area", "equivalent_diameter"))
    df = pd.DataFrame(props)

    # 过滤噪声
    df = df[df["area"] >= MIN_AREA_PX].copy()
    print(f"[INFO] After area filter (>= {MIN_AREA_PX}px): N = {len(df)}")

    # 粒径列
    df["d_eq_px"] = df["equivalent_diameter"]

    if PIXEL_TO_MM is not None:
        df["d_eq_mm"] = df["d_eq_px"] * float(PIXEL_TO_MM)
        size_col = "d_eq_mm"
        unit = "mm"
    else:
        size_col = "d_eq_px"
        unit = "px"

    # 分位数
    x = df[size_col].values
    if len(x) == 0:
        raise RuntimeError("No particles detected after filtering. "
                           "Try reducing MIN_AREA_PX or adjusting DIAMETER / preprocessing.")

    d10, d50, d90 = np.percentile(x, [10, 50, 90])
    print(f"[RESULT] {size_col}: d10={d10:.3f}{unit}, d50={d50:.3f}{unit}, d90={d90:.3f}{unit}")

    # =========================
    # 4) 保存表格
    # =========================
    csv_path = os.path.join(OUT_DIR, "particle_sizes.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved: {csv_path}")

    # =========================
    # 5) 可视化输出：mask预览 + 叠加边界
    # =========================
    masks_preview = masks_to_color(masks)
    masks_path = os.path.join(OUT_DIR, "masks_preview.png")
    cv2.imwrite(masks_path, masks_preview)
    print(f"[INFO] Saved: {masks_path}")

    overlay = overlay_masks_on_image(gray, masks)
    overlay_path = os.path.join(OUT_DIR, "overlay_preview.png")
    cv2.imwrite(overlay_path, overlay)
    print(f"[INFO] Saved: {overlay_path}")

    # =========================
    # 6) 粒径分布：直方图 + CDF
    # =========================
    # 直方图
    plt.figure()
    plt.hist(x, bins=30)
    plt.xlabel(f"Equivalent diameter ({unit})")
    plt.ylabel("Count")
    plt.title("Particle size distribution (hist)")
    hist_path = os.path.join(OUT_DIR, "psd_hist.png")
    plt.savefig(hist_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {hist_path}")

    # CDF
    plt.figure()
    xs = np.sort(x)
    cdf = np.arange(1, len(xs) + 1) / len(xs)
    plt.plot(xs, cdf)
    plt.xlabel(f"Equivalent diameter ({unit})")
    plt.ylabel("CDF")
    plt.title("Particle size distribution (CDF)")
    cdf_path = os.path.join(OUT_DIR, "psd_cdf.png")
    plt.savefig(cdf_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {cdf_path}")

    print("[DONE] All outputs written to:", OUT_DIR)


if __name__ == "__main__":
    main()
