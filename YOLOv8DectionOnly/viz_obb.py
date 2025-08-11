# -*- coding: utf-8 -*-
import os, glob
import cv2
import numpy as np
from tqdm import tqdm

# ================== 配置区 ==================
DATASET_DIR = r"D:\0StudyR\2024-2025-4-Summer\MA463\PokerImage\dataset"
SPLITS = ["train", "valid", "test"]
LABEL_SUBDIR = "labels_obb"   # 如果你是覆盖到原 labels，则改成 "labels"
MAX_IMAGES_PER_SPLIT = None   # 例如 50；None 表示全部
THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
# ==========================================

def draw_rotated_box(img, xc_n, yc_n, w_n, h_n, theta_deg, color, thickness=2):
    H, W = img.shape[:2]
    cx = xc_n * W
    cy = yc_n * H
    ww = w_n  * W
    hh = h_n  * H
    rect = ((cx, cy), (ww, hh), theta_deg)
    box = cv2.boxPoints(rect)            # 4x2 float
    box = box.astype(int)                 # ✅ 旧 np.int0 改成 astype(int)
    cv2.polylines(img, [box], isClosed=True, color=color, thickness=thickness)
    return box


def list_images(dir_images):
    # 支持常见后缀
    return sorted(sum([glob.glob(os.path.join(dir_images, f"*.{ext}"))
                       for ext in ["jpg","jpeg","png","bmp","webp"]], []))

def main():
    print(f"[INFO] DATASET_DIR = {os.path.abspath(DATASET_DIR)}")
    for split in SPLITS:
        img_dir = os.path.join(DATASET_DIR, split, "images")
        lbl_dir = os.path.join(DATASET_DIR, split, LABEL_SUBDIR)
        out_dir = os.path.join(DATASET_DIR, split, "viz_obb")
        os.makedirs(out_dir, exist_ok=True)

        img_files = list_images(img_dir)
        if MAX_IMAGES_PER_SPLIT:
            img_files = img_files[:MAX_IMAGES_PER_SPLIT]

        print(f"\n[INFO] Split: {split}")
        print(f"       images: {len(img_files)}  dir={os.path.abspath(img_dir)}")
        print(f"       labels: dir={os.path.abspath(lbl_dir)}")
        print(f"       output: dir={os.path.abspath(out_dir)}")

        for img_path in tqdm(img_files, desc=f"{split}"):
            stem = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(lbl_dir, stem + ".txt")
            img = cv2.imread(img_path)
            if img is None:
                continue

            H, W = img.shape[:2]
            if not os.path.exists(lbl_path):
                # 没有标签也保存一份，便于排查
                cv2.imwrite(os.path.join(out_dir, stem + "_no_label.jpg"), img)
                continue

            with open(lbl_path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]

            # 给不同类别分配颜色（简单映射）
            def color_for_class(c):
                # 固定几种颜色循环
                palette = [(0,255,0),(255,0,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]
                return palette[c % len(palette)]

            for ln in lines:
                parts = ln.split()
                if len(parts) < 6:
                    # 不是 OBB 行，跳过
                    continue
                cls = int(parts[0])
                xc, yc, w, h, theta = map(float, parts[1:6])

                color = color_for_class(cls)
                box = draw_rotated_box(img, xc, yc, w, h, theta, color, THICKNESS)

                # 画类别与角度
                cx = int(xc * W); cy = int(yc * H)
                cv2.putText(img, f"id:{cls} th:{theta:.1f}", (max(0,cx-40), max(15,cy-10)),
                            FONT, 0.5, color, 1, cv2.LINE_AA)

            out_path = os.path.join(out_dir, stem + ".jpg")
            cv2.imwrite(out_path, img)

    print("\n[DONE] 可视化完成。到各 split 的 viz_obb 目录查看效果。")

if __name__ == "__main__":
    main()
