from pathlib import Path
import cv2
from tqdm import tqdm

ROOT = Path("/home/fanh/Poker/dataset")     # 解压后的多类数据
OUT  = Path("/home/fanh/Poker/dataset_cls") # 分类用ROI输出
PAD  = 0.10                                  # 四周留白比例
LABEL_DIRNAME = "labels"                      # 用多类标注来裁剪

def clip(v, lo, hi): return max(lo, min(hi, v))
def yolo_to_xyxy(xc, yc, w, h, W, H):
    x1 = (xc - w/2) * W; y1 = (yc - h/2) * H
    x2 = (xc + w/2) * W; y2 = (yc + h/2) * H
    return x1, y1, x2, y2
def expand(x1,y1,x2,y2,W,H,ratio):
    w, h = x2-x1, y2-y1
    x1 -= w*ratio; y1 -= h*ratio; x2 += w*ratio; y2 += h*ratio
    x1 = int(clip(x1,0,W-1)); y1 = int(clip(y1,0,H-1))
    x2 = int(clip(x2,0,W-1)); y2 = int(clip(y2,0,H-1))
    return x1,y1,x2,y2

def process_split(split):
    img_dir   = ROOT / split / "images"
    label_dir = ROOT / split / LABEL_DIRNAME
    out_split = OUT / split
    out_split.mkdir(parents=True, exist_ok=True)

    txts = sorted(label_dir.glob("*.txt"))
    print(f"[{split}] labels={len(txts)} from {label_dir}")
    for p in tqdm(txts, desc=f"crop {split}"):
        stem = p.stem
        # 找图
        img_path = None
        for ext in (".jpg",".jpeg",".png",".bmp",".webp"):
            q = img_dir / f"{stem}{ext}"
            if q.exists(): img_path = q; break
        if not img_path:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]

        with p.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) < 5:  # 跳过空/坏行
                    continue
                cls = int(float(parts[0]))
                xc, yc, w, h = map(float, parts[1:5])
                x1,y1,x2,y2 = yolo_to_xyxy(xc,yc,w,h,W,H)
                x1,y1,x2,y2 = expand(x1,y1,x2,y2,W,H,PAD)
                if x2<=x1 or y2<=y1: 
                    continue
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                # 按类别建文件夹
                cls_dir = out_split / str(cls)
                cls_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(cls_dir / f"{stem}_{i:02d}.jpg"), crop)

if __name__ == "__main__":
    for sp in ["train","valid","test"]:
        process_split(sp)
    print("✅ Done. 输出在 /home/fanh/Poker/dataset_cls/（train/valid/test 下按类分文件夹）")
