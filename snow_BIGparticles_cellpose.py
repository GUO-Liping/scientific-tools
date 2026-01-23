# -*- coding: utf-8 -*-
"""
wet_snow_cellpose_superlarge_only_408.py
Cellpose 4.0.8：只识别超大颗粒（等效直径 >= D_EQ_MIN_PX）+ 导出/可视化/统计

依赖：
pip install cellpose opencv-python scikit-image pandas matplotlib

输出（OUT_DIR）：
- debug_corr.png / debug_grad.png / debug_combo_big.png
- masks_super_preview.png
- overlay_super.png
- super_particles.csv
- psd_hist_super.png / psd_cdf_super.png
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cellpose import models
from skimage.measure import regionprops_table


# =========================
# 0) 用户参数
# =========================
IMG_PATH = "snow_particles.jpg"
OUT_DIR  = "snow_cellpose_superlarge_out"

USE_GPU = True
MODEL_TYPE = "cyto3"   # 4.0.8 推荐先用 cyto3，不行再试 cyto

# 只保留“超大颗粒”的像素等效直径阈值（px）
D_EQ_MIN_PX = 180

# 如果你要换成更保守/更严格，可修改：
# D_EQ_MIN_PX = 200

# 预处理（偏大颗粒：更依赖强度/低频对比）
BG_KERNEL   = 81       # 大块常被背景校正吃掉 → 建议偏小：61/81/101
CLAHE_CLIP  = 2.0
BLUR_K      = 3        # 奇数：3/5

# 只做大尺度 diameter sweep（可按实际图像再扩展）
DIAM_SWEEP_BIG = [150, 180, 200, 230, 260, 300]

# 阈值（漏检多就把 cellprob 调更负；假块太多就把 cellprob 往 0 调）
CELLPROB_TH = -2.0
FLOW_TH     = 0.5

# （可选）面积兜底：避免一些极小噪点进入 props（一般用不上）
MIN_AREA_PX_FLOOR = 0   # 可设 5000 等，但通常仅 D_EQ_MIN_PX 就够了


# =========================
# 工具函数
# =========================
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def preprocess(gray: np.ndarray):
    """背景校正 + CLAHE + blur + Sobel 梯度（用于可选融合）"""
    g = gray.copy()

    k = int(BG_KERNEL)
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.morphologyEx(g, cv2.MORPH_OPEN, kernel)
    corr = cv2.subtract(g, bg)
    corr = cv2.normalize(corr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(8, 8))
    corr = clahe.apply(corr)

    k2 = int(BLUR_K)
    if k2 >= 3:
        if k2 % 2 == 0:
            k2 += 1
        corr = cv2.GaussianBlur(corr, (k2, k2), 0)

    gx = cv2.Sobel(corr, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(corr, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return corr, grad


def build_combo_big(corr, grad):
    """
    大块：尽量保留强度信息，少量边缘即可
    你也可以试 0.95/0.05 或 0.9/0.1
    """
    return cv2.addWeighted(corr, 0.95, grad, 0.05, 0)


def eval_cellpose(model, img, diameter):
    """兼容不同小版本：不支持阈值参数则退化"""
    try:
        return model.eval(
            img,
            diameter=diameter,
            channels=[0, 0],
            cellprob_threshold=CELLPROB_TH,
            flow_threshold=FLOW_TH,
        )
    except TypeError:
        return model.eval(img, diameter=diameter, channels=[0, 0])


def masks_preview(masks: np.ndarray):
    if masks.max() == 0:
        return np.zeros((*masks.shape, 3), dtype=np.uint8)
    m = (masks.astype(np.float32) / (masks.max() + 1e-9) * 255).astype(np.uint8)
    return cv2.applyColorMap(m, cv2.COLORMAP_TURBO)


def overlay_edges(gray: np.ndarray, masks: np.ndarray):
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    m = (masks > 0).astype(np.uint8) * 255
    edge = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    out = base.copy()
    out[edge] = (0, 0, 255)
    return out


def relabel_consecutive(masks: np.ndarray):
    out = np.zeros_like(masks, dtype=np.int32)
    nid = 1
    for lab in range(1, int(masks.max()) + 1):
        mm = (masks == lab)
        if mm.any():
            out[mm] = nid
            nid += 1
    return out


def filter_superlarge_by_deq(masks: np.ndarray, d_eq_min_px: float, min_area_floor: int = 0):
    """
    只保留等效直径 >= d_eq_min_px 的实例，并重编号。
    """
    if int(masks.max()) == 0:
        return masks, pd.DataFrame()

    props = regionprops_table(
        masks,
        properties=("label", "area", "equivalent_diameter", "centroid")
    )
    df = pd.DataFrame(props)

    if min_area_floor and min_area_floor > 0:
        df = df[df["area"] >= int(min_area_floor)].copy()

    df = df[df["equivalent_diameter"] >= float(d_eq_min_px)].copy()

    out = np.zeros_like(masks, dtype=np.int32)
    nid = 1
    for lab in df["label"].values:
        mm = (masks == lab)
        if mm.any():
            out[mm] = nid
            nid += 1

    out = relabel_consecutive(out)
    return out, df


def main():
    ensure_dir(OUT_DIR)

    img_bgr = cv2.imread(IMG_PATH)
    if img_bgr is None:
        raise FileNotFoundError(IMG_PATH)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    corr, grad = preprocess(gray)
    combo_big = build_combo_big(corr, grad)

    cv2.imwrite(os.path.join(OUT_DIR, "debug_corr.png"), corr)
    cv2.imwrite(os.path.join(OUT_DIR, "debug_grad.png"), grad)
    cv2.imwrite(os.path.join(OUT_DIR, "debug_combo_big.png"), combo_big)

    model = models.CellposeModel(model_type=MODEL_TYPE, gpu=USE_GPU)

    # 选择“筛选后超大颗粒数最多”的 diameter
    best = None  # (n_super, d, masks_super, df_super)
    for d in DIAM_SWEEP_BIG:
        masks_raw, flows, styles = eval_cellpose(model, combo_big, d)
        masks_super, df_super = filter_superlarge_by_deq(
            masks_raw, d_eq_min_px=D_EQ_MIN_PX, min_area_floor=MIN_AREA_PX_FLOOR
        )
        n_super = int(masks_super.max())
        print(f"[SWEEP] diameter={d:>3}px -> superlarge_instances={n_super}")
        if best is None or n_super > best[0]:
            best = (n_super, d, masks_super, df_super)

    n_super, best_d, masks_super, df_super = best
    print(f"[BEST] diameter={best_d}px, superlarge_instances={n_super}")

    # 输出 mask / overlay
    cv2.imwrite(os.path.join(OUT_DIR, "masks_super_preview.png"), masks_preview(masks_super))
    cv2.imwrite(os.path.join(OUT_DIR, "overlay_super.png"), overlay_edges(gray, masks_super))

    # 导出表格：用“重筛选后的 regionprops”再计算一次（保证与最终 masks 一致）
    if int(masks_super.max()) == 0:
        print("[WARN] No superlarge particles detected. Try lowering D_EQ_MIN_PX or adjusting thresholds.")
        df_out = pd.DataFrame(columns=["area", "equivalent_diameter", "centroid-0", "centroid-1"])
    else:
        props_out = regionprops_table(
            masks_super,
            properties=("area", "equivalent_diameter", "centroid")
        )
        df_out = pd.DataFrame(props_out)

    csv_path = os.path.join(OUT_DIR, "super_particles.csv")
    df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved: {csv_path}")

    # 统计图（只对超大颗粒）
    if len(df_out) > 0:
        x = df_out["equivalent_diameter"].values

        plt.figure()
        plt.hist(x, bins=20)
        plt.xlabel("Equivalent diameter (px)")
        plt.ylabel("Count")
        plt.title(f"Superlarge PSD (d_eq >= {D_EQ_MIN_PX}px)")
        plt.savefig(os.path.join(OUT_DIR, "psd_hist_super.png"), dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure()
        xs = np.sort(x)
        cdf = np.arange(1, len(xs) + 1) / len(xs)
        plt.plot(xs, cdf)
        plt.xlabel("Equivalent diameter (px)")
        plt.ylabel("CDF")
        plt.title(f"Superlarge PSD CDF (d_eq >= {D_EQ_MIN_PX}px)")
        plt.savefig(os.path.join(OUT_DIR, "psd_cdf_super.png"), dpi=300, bbox_inches="tight")
        plt.close()

    print("[DONE] outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
