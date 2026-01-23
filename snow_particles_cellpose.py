# -*- coding: utf-8 -*-
"""
wet_snow_cellpose_3scale_FINAL_408.py
Cellpose 4.0.8：湿雪颗粒三尺度实例分割 + 尺度专属过滤 + 大到小合并 + 粒径分布

依赖：
pip install cellpose opencv-python scikit-image pandas matplotlib

输出（OUT_DIR）：
- debug_corr.png / debug_grad.png
- debug_combo_S1.png / debug_combo_S2.png / debug_combo_S3.png
- masks_S1_preview.png / masks_S2_preview.png / masks_S3_preview.png / masks_merged_preview.png
- overlay_S1.png / overlay_S2.png / overlay_S3.png / overlay_merged.png
- particle_sizes.csv
- psd_hist.png / psd_cdf.png
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
OUT_DIR = "snow_cellpose_out"

# 标定：222 px = 3.75 m
PIXEL_TO_MM = 3750.0 / 222.0  # mm/px

USE_GPU = True
MODEL_TYPE = "cyto3"          # 4.0.8：建议先 cyto3，不行再试 cyto

# ---- 预处理参数（湿雪常用）----
BG_KERNEL = 101               # 多尺度建议别太大：81/101/121 试
CLAHE_CLIP = 2.0
BLUR_K = 3                    # 必须奇数：3/5

# ---- 三尺度直径 sweep（按你想法可继续改）----
DIAM_SWEEPS = [
    ("S1", [10, ]),            # 小颗粒（10px 量级）
    ("S2", [100,]),         # 大颗粒（~100px）
    ("S3", [200, ]),         # 超大块（可选）
]

# ---- 每尺度融合权重：combo = w_corr*corr + w_grad*grad ----
COMBO_WEIGHTS = {
    "S1": (0.75, 0.25),   # 小颗粒更依赖边缘
    "S2": (0.90, 0.10),   # 大颗粒更依赖强度/低频对比
    "S3": (0.95, 0.05),   # 超大块更偏强度
}

# ---- 每尺度阈值：cellprob 越负越“敢分割”（漏检少但噪声多）----
THRESHOLDS = {
    "S1": (-2.5, 0.4),    # (cellprob_threshold, flow_threshold)
    "S2": (-2.0, 0.4),
    "S3": (-1.5, 0.5),
}

# ---- 尺度专属最小面积（px^2）：关键！避免大尺度保留小碎片 ----
# 这组是稳健起点：跑完看 overlay 再微调
MIN_AREA_BY_SCALE = {
    "S1": 20,      # 小颗粒：过滤极小噪点
    "S2": 300,     # 大尺度：强制只保留“大颗粒”级别
    "S3": 1500,    # 超大尺度：强制只保留“块体”级别
}

# （可选）最终 merged 的极小兜底（只杀 1-2 像素噪点；不参与物理解释）
MIN_AREA_GLOBAL = None   # 可设 3；不建议设大

# ---- 合并参数：允许 add 与 base 的最大重叠比例（相对 add 面积）----
OVERLAP_FRAC_MAX = 0.02  # 小颗粒在大颗粒边缘少量重叠可保留；若想强保护大颗粒可设 0.0

# ---- 合并顺序：必须从大到小 ----
MERGE_ORDER = ["S3", "S2", "S1"]


# =========================
# 工具函数
# =========================
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def preprocess(gray: np.ndarray):
    """背景校正 + CLAHE + blur + Sobel 梯度"""
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


def build_combo(corr, grad, w_corr, w_grad):
    return cv2.addWeighted(corr, float(w_corr), grad, float(w_grad), 0)


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


def eval_cellpose(model, img, diameter, cellprob_th, flow_th):
    """兼容不同小版本：不支持阈值参数则退化"""
    try:
        return model.eval(
            img,
            diameter=diameter,
            channels=[0, 0],
            cellprob_threshold=cellprob_th,
            flow_threshold=flow_th
        )
    except TypeError:
        return model.eval(img, diameter=diameter, channels=[0, 0])


def relabel_consecutive(masks: np.ndarray):
    """labels 重编号为 1..N"""
    out = np.zeros_like(masks, dtype=np.int32)
    nid = 1
    for lab in range(1, int(masks.max()) + 1):
        mm = (masks == lab)
        if mm.any():
            out[mm] = nid
            nid += 1
    return out


def filter_by_min_area(masks: np.ndarray, min_area: int):
    """按面积过滤并重编号"""
    if min_area is None or min_area <= 0:
        return relabel_consecutive(masks)

    props = regionprops_table(masks, properties=("label", "area"))
    dfp = pd.DataFrame(props)
    keep = dfp.loc[dfp["area"] >= int(min_area), "label"].values

    out = np.zeros_like(masks, dtype=np.int32)
    nid = 1
    for lab in keep:
        mm = (masks == lab)
        if mm.any():
            out[mm] = nid
            nid += 1
    return out


def pick_best_by_sweep(model, img, diam_list, cellprob_th, flow_th, min_area, tag=""):
    """
    扫直径并选择“通过 min_area 过滤后实例数最多”的结果
    （比直接用 masks.max 更合理：能抑制大尺度碎片化的假增益）
    """
    best = None  # (n_after_filter, d, masks_filtered, masks_raw)
    for d in diam_list:
        masks_raw, flows, styles = eval_cellpose(model, img, d, cellprob_th, flow_th)
        masks_f = filter_by_min_area(masks_raw, min_area)
        n = int(masks_f.max())
        print(f"[SWEEP-{tag}] d={d:>3}px -> kept_instances={n}")
        if best is None or n > best[0]:
            best = (n, d, masks_f, masks_raw)

    print(f"[BEST-{tag}] diameter={best[1]} px, kept_instances={best[0]}")
    return best[2], best[1]


def merge_masks(base: np.ndarray, add: np.ndarray, overlap_frac_max: float):
    """
    base：当前合并结果（大尺度优先）
    add ：待补充的更小尺度结果
    overlap_frac_max：允许 add 与 base 的最大重叠比例（相对 add 面积）
    """
    base = relabel_consecutive(base)
    add = relabel_consecutive(add)

    out = base.copy()
    next_id = int(out.max()) + 1
    base_occ = out > 0

    for sid in range(1, int(add.max()) + 1):
        sm = (add == sid)
        if not sm.any():
            continue
        overlap = (base_occ & sm).sum()
        frac = overlap / float(sm.sum())
        if frac > overlap_frac_max:
            continue
        out[sm] = next_id
        next_id += 1

    return out


# =========================
# 主程序
# =========================
def main():
    ensure_dir(OUT_DIR)

    img_bgr = cv2.imread(IMG_PATH)
    if img_bgr is None:
        raise FileNotFoundError(IMG_PATH)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    corr, grad = preprocess(gray)
    cv2.imwrite(os.path.join(OUT_DIR, "debug_corr.png"), corr)
    cv2.imwrite(os.path.join(OUT_DIR, "debug_grad.png"), grad)

    # 每尺度 combo
    combos = {}
    for tag, _ in DIAM_SWEEPS:
        w_corr, w_grad = COMBO_WEIGHTS.get(tag, (0.85, 0.15))
        combos[tag] = build_combo(corr, grad, w_corr, w_grad)
        cv2.imwrite(os.path.join(OUT_DIR, f"debug_combo_{tag}.png"), combos[tag])

    # 模型
    model = models.CellposeModel(model_type=MODEL_TYPE, gpu=USE_GPU)

    # 每尺度分割（sweep + 尺度专属 min_area 过滤）
    masks_by_scale = {}
    chosen_d = {}
    for tag, diam_list in DIAM_SWEEPS:
        cellprob_th, flow_th = THRESHOLDS.get(tag, (-2.0, 0.4))
        min_area = MIN_AREA_BY_SCALE.get(tag, 0)

        masks_tag, d_tag = pick_best_by_sweep(
            model=model,
            img=combos[tag],
            diam_list=diam_list,
            cellprob_th=cellprob_th,
            flow_th=flow_th,
            min_area=min_area,
            tag=tag
        )

        masks_by_scale[tag] = masks_tag
        chosen_d[tag] = d_tag

        cv2.imwrite(os.path.join(OUT_DIR, f"masks_{tag}_preview.png"), masks_preview(masks_tag))
        cv2.imwrite(os.path.join(OUT_DIR, f"overlay_{tag}.png"), overlay_edges(gray, masks_tag))

    print("[INFO] chosen diameters:", chosen_d)
    print("[INFO] kept instances by scale:",
          {k: int(v.max()) for k, v in masks_by_scale.items()})

    # 合并：从大到小
    merged = np.zeros_like(gray, dtype=np.int32)
    for tag in MERGE_ORDER:
        if tag in masks_by_scale:
            merged = merge_masks(merged, masks_by_scale[tag], overlap_frac_max=OVERLAP_FRAC_MAX)

    masks_merged = relabel_consecutive(merged)

    # （可选）极小兜底去噪
    if MIN_AREA_GLOBAL is not None and MIN_AREA_GLOBAL > 0:
        masks_merged = filter_by_min_area(masks_merged, MIN_AREA_GLOBAL)

    print("[INFO] merged instances:", int(masks_merged.max()))
    cv2.imwrite(os.path.join(OUT_DIR, "masks_merged_preview.png"), masks_preview(masks_merged))
    cv2.imwrite(os.path.join(OUT_DIR, "overlay_merged.png"), overlay_edges(gray, masks_merged))

    # 粒径统计（merged）
    props = regionprops_table(masks_merged, properties=("area", "equivalent_diameter"))
    df = pd.DataFrame(props)
    print("[INFO] merged regions:", len(df))

    if len(df) == 0:
        raise RuntimeError("No particles detected in merged masks. Try loosening thresholds or adjusting BG_KERNEL/diameters.")

    df["d_eq_px"] = df["equivalent_diameter"]
    df["d_eq_mm"] = df["d_eq_px"] * PIXEL_TO_MM

    x = df["d_eq_mm"].values
    d10, d50, d90 = np.percentile(x, [10, 50, 90])
    print(f"[RESULT] d10={d10:.1f} mm, d50={d50:.1f} mm, d90={d90:.1f} mm")

    # 保存 CSV
    df.to_csv(os.path.join(OUT_DIR, "particle_sizes.csv"), index=False, encoding="utf-8-sig")

    # PSD 图
    plt.figure()
    plt.hist(x, bins=30)
    plt.xlabel("Equivalent diameter (mm)")
    plt.ylabel("Count")
    plt.title("PSD (hist) - merged")
    plt.savefig(os.path.join(OUT_DIR, "psd_hist.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    xs = np.sort(x)
    cdf = np.arange(1, len(xs) + 1) / len(xs)
    plt.plot(xs, cdf)
    plt.xlabel("Equivalent diameter (mm)")
    plt.ylabel("CDF")
    plt.title("PSD (CDF) - merged")
    plt.savefig(os.path.join(OUT_DIR, "psd_cdf.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print("[DONE] outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
