# -*- coding: utf-8 -*-
import os, glob, math
import cv2
import numpy as np
from tqdm import tqdm

DATASET_DIR = r"D:\0StudyR\2024-2025-4-Summer\MA463\PokerImage\dataset"
SPLITS = ["train", "valid", "test"]
LABEL_IN = "labels"          # 原 AABB
LABEL_OUT = "labels_obb"     # 输出 OBB
ROI_EXPAND = 2.2             # 🔑 将小框按这个倍数放大后再估角，尽量囊括牌边
CANNY_T1, CANNY_T2 = 50, 150
HOUGH_MIN_LEN = 40           # 线段最短像素
HOUGH_MAX_GAP = 10
ANGLE_SNAP_DEG = 2.0         # 靠近 0/90/180 时吸附，抑制抖动

def clamp(v, lo, hi): return max(lo, min(hi, v))

def yolo_to_xyxy(xc, yc, w, h, W, H):
    x1 = (xc - w/2) * W
    y1 = (yc - h/2) * H
    x2 = (xc + w/2) * W
    y2 = (yc + h/2) * H
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

def expand_xyxy(x1,y1,x2,y2,W,H,ratio):
    cx, cy = (x1+x2)/2, (y1+y2)/2
    ww, hh = (x2-x1)*ratio, (y2-y1)*ratio
    nx1 = clamp(int(round(cx-ww/2)), 0, W-1)
    ny1 = clamp(int(round(cy-hh/2)), 0, H-1)
    nx2 = clamp(int(round(cx+ww/2)), 0, W-1)
    ny2 = clamp(int(round(cy+hh/2)), 0, H-1)
    return nx1, ny1, nx2, ny2

def angle_norm_90(a):
    """ 归一化到 [-90,90) """
    a = ((a + 90) % 180) - 90
    # 吸附到 0/±90
    for t in [0, 90, -90]:
        if abs(a - t) < ANGLE_SNAP_DEG:
            a = float(t)
    return a

def angle_from_hough(roi):
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3,3), 0)
    edges = cv2.Canny(g, CANNY_T1, CANNY_T2)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180.0, threshold=60,
                            minLineLength=HOUGH_MIN_LEN, maxLineGap=HOUGH_MAX_GAP)
    if lines is None: 
        return None

    angles = []
    lengths = []
    for x1,y1,x2,y2 in lines[:,0,:]:
        dx, dy = x2-x1, y2-y1
        if dx==0 and dy==0: 
            continue
        ang = math.degrees(math.atan2(dy, dx))  # (-180,180]
        # 牌边的两组方向相差约90°，我们统一到 [-90,90)
        ang = angle_norm_90(ang)
        L = math.hypot(dx, dy)
        angles.append(ang)
        lengths.append(L)

    if not angles:
        return None

    # 取“最长若干线”的加权中位数/均值更稳
    idx = np.argsort(lengths)[-min(10, len(lengths)):]  # 最长的若干条
    sel_angles = np.array([angles[i] for i in idx], dtype=float)
    sel_lengths = np.array([lengths[i] for i in idx], dtype=float)
    # 将角度折叠到同一簇（避免 +89 和 -89 混在一起）
    base = sel_angles[sel_lengths.argmax()]
    diff = np.array([angle_norm_90(a - base) for a in sel_angles])
    ang = angle_norm_90(base + np.average(diff, weights=sel_lengths))
    return ang

def angle_from_pca(roi):
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3,3), 0)
    edges = cv2.Canny(g, CANNY_T1, CANNY_T2)
    ys, xs = np.nonzero(edges)
    if len(xs) < 50:
        return None
    pts = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    mean, eigvecs = cv2.PCACompute(pts, mean=None, maxComponents=2)
    v = eigvecs[0]  # 主轴
    ang = math.degrees(math.atan2(v[1], v[0]))
    return angle_norm_90(ang)

def angle_from_minarea(roi):
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3,3), 0)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    bw = 255 - bw  # 让前景为白
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)                # angle in [-90,0)
    (rw, rh) = rect[1]
    angle = rect[2]
    if rw < rh: 
        angle += 90.0
    return angle_norm_90(angle)

def estimate_theta_for_roi(img, x1,y1,x2,y2):
    """ 先扩大 ROI，再依次 Hough->PCA->minAreaRect 求角 """
    H, W = img.shape[:2]
    ex1, ey1, ex2, ey2 = expand_xyxy(x1,y1,x2,y2,W,H,ROI_EXPAND)
    roi = img[ey1:ey2, ex1:ex2]
    if roi.size == 0:
        return 0.0
    # 1) Hough
    ang = angle_from_hough(roi)
    if ang is not None:
        return ang
    # 2) PCA
    ang = angle_from_pca(roi)
    if ang is not None:
        return ang
    # 3) minAreaRect
    ang = angle_from_minarea(roi)
    if ang is not None:
        return ang
    return 0.0

def process_split(split):
    img_dir = os.path.join(DATASET_DIR, split, "images")
    in_dir  = os.path.join(DATASET_DIR, split, LABEL_IN)
    out_dir = os.path.join(DATASET_DIR, split, LABEL_OUT)
    os.makedirs(out_dir, exist_ok=True)

    label_files = glob.glob(os.path.join(in_dir, "*.txt"))
    print(f"\n[INFO] Split {split}: images={os.path.abspath(img_dir)}  labels={len(label_files)}")
    for lbl_path in tqdm(label_files, desc=f"{split}"):
        stem = os.path.splitext(os.path.basename(lbl_path))[0]
        img_path = None
        for ext in [".jpg",".jpeg",".png",".bmp",".webp"]:
            p = os.path.join(img_dir, stem+ext)
            if os.path.exists(p):
                img_path = p; break
        if img_path is None:
            continue

        img = cv2.imread(img_path)
        if img is None: 
            continue
        H, W = img.shape[:2]
        out_lines = []

        with open(lbl_path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln: 
                    continue
                parts = ln.split()
                if len(parts) < 5:
                    continue
                cls, xc, yc, w, h = parts[:5]
                cls = int(cls)
                xc, yc, w, h = map(float, (xc, yc, w, h))
                x1,y1,x2,y2 = yolo_to_xyxy(xc, yc, w, h, W, H)

                theta = estimate_theta_for_roi(img, x1,y1,x2,y2)

                # 输出 OBB（归一化）
                cx_pix = (x1+x2)/2.0; cy_pix = (y1+y2)/2.0
                w_pix  = (x2-x1);      h_pix  = (y2-y1)
                xc_n = cx_pix / W; yc_n = cy_pix / H
                w_n  = w_pix  / W; h_n  = h_pix  / H

                out_lines.append(f"{cls} {xc_n:.6f} {yc_n:.6f} {w_n:.6f} {h_n:.6f} {theta:.2f}")

        with open(os.path.join(out_dir, stem + ".txt"), "w", encoding="utf-8") as fo:
            fo.write("\n".join(out_lines))

if __name__ == "__main__":
    for sp in SPLITS:
        process_split(sp)
    print("\n[DONE] AABB→OBB 转换完成（扩大 ROI + Hough/PCA 回退）。")
