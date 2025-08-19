# -*- coding: utf-8 -*-
"""
poker_yolo_detect_and_crop.py
- 生成 data.yaml
- 训练 YOLOv8（水平框）
- 推理并裁剪 ROI（支持 train/valid/test 任意子集）
用法示例：
  python poker_yolo_detect_and_crop.py \
    --dataset /home/fanh/Poker/dataset \
    --epochs 20 --imgsz 640 --batch 128 --device 0,1 \
    --splits test --pad 0.10 --conf 0.25 --iou 0.45
如已训练好，直接裁剪：
  python poker_yolo_detect_and_crop.py --dataset /home/fanh/Poker/dataset \
    --weights /home/fanh/Poker/runs_detect/yolov8n_train/weights/best.pt \
    --skip-train --splits test
"""
import os
import sys
import argparse
from pathlib import Path
import yaml
import cv2
from tqdm import tqdm

# ---------------- Utilities ----------------
def guess_num_classes(lbl_dir: Path) -> int:
    import glob
    max_id = -1
    files = glob.glob(str(lbl_dir / "*.txt"))
    for p in files[:1000]:  # 采样避免太慢
        try:
            with open(p, "r", encoding="utf-8") as f:
                for ln in f:
                    ps = ln.strip().split()
                    if len(ps) >= 5:
                        cid = int(float(ps[0]))
                        max_id = max(max_id, cid)
        except Exception:
            continue
    return max_id + 1 if max_id >= 0 else 1

def ensure_yaml(dataset_dir: Path, out_yaml: Path) -> Path:
    train_dir = (dataset_dir / "train").as_posix()
    val_dir   = (dataset_dir / "valid").as_posix()
    test_dir  = (dataset_dir / "test").as_posix()
    nc = guess_num_classes(dataset_dir / "train" / "labels")
    names = [str(i) for i in range(nc)]
    data = {"path": dataset_dir.as_posix(),
            "train": train_dir, "val": val_dir, "test": test_dir,
            "names": names}
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    print(f"[INFO] data.yaml 已生成: {out_yaml}")
    print(f"[INFO] 估计类别数: {nc}")
    return out_yaml

def expand_box(xyxy, pad_ratio, W, H):
    x1, y1, x2, y2 = map(float, xyxy)
    w, h = x2 - x1, y2 - y1
    px, py = w * pad_ratio, h * pad_ratio
    nx1 = int(max(0, min(W - 1, x1 - px)))
    ny1 = int(max(0, min(H - 1, y1 - py)))
    nx2 = int(max(0, min(W - 1, x2 + px)))
    ny2 = int(max(0, min(H - 1, y2 + py)))
    return nx1, ny1, nx2, ny2

# ---------------- Core ----------------
def train_model(data_yaml: Path, runs_dir: Path, model_size: str,
                epochs: int, imgsz: int, batch, device, workers: int, cache):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] 未安装 ultralytics，请先: pip install ultralytics")
        sys.exit(1)
    weights = f"yolov8{model_size}.pt"
    runs_dir.mkdir(parents=True, exist_ok=True)
    print("[INFO] 开始训练 YOLOv8 ...")
    model = YOLO(weights)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch if batch else 0,   # 0=自动探测
        project=str(runs_dir),
        name=f"yolov8{model_size}_train",
        exist_ok=True,
        device=str(device),            # "0" 或 "0,1"
        workers=workers,
        cache=cache
    )
    best = runs_dir / f"yolov8{model_size}_train" / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError("训练完成但未找到 best.pt，请检查日志。")
    print(f"[INFO] 最优权重: {best}")
    return best

def predict_and_crop(weights: Path, dataset_dir: Path, splits, crop_dir: Path,
                     conf: float, iou: float, imgsz: int, device, workers: int, pad_ratio: float):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] 未安装 ultralytics，请先: pip install ultralytics")
        sys.exit(1)
    model = YOLO(str(weights))
    names = model.names if hasattr(model, "names") else {}
    crop_dir.mkdir(parents=True, exist_ok=True)

    for sp in splits:
        img_dir = dataset_dir / sp / "images"
        if not img_dir.exists():
            print(f"[WARN] 跳过 {sp}: 不存在 {img_dir}")
            continue
        out_dir = crop_dir / sp
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] 推理并裁剪：split={sp}")
        results = model.predict(
            source=str(img_dir),
            conf=conf, iou=iou, imgsz=imgsz,
            stream=True, device=str(device), workers=workers, verbose=False
        )
        for r in tqdm(results, desc=f"crop {sp}"):
            if r.path is None or r.orig_img is None:
                continue
            img = r.orig_img  # BGR numpy
            H, W = img.shape[:2]
            stem = Path(r.path).stem
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for i, box in enumerate(r.boxes):
                xyxy = box.xyxy[0].tolist()
                confv = float(box.conf[0]) if box.conf is not None else 0.0
                clsid = int(box.cls[0]) if box.cls is not None else -1
                x1,y1,x2,y2 = expand_box(xyxy, pad_ratio, W, H)
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                cls_name = names.get(clsid, str(clsid))
                save_dir = out_dir / f"class_{cls_name}"
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f"{stem}_det{i:02d}_c{clsid}_p{confv:.2f}.jpg"
                cv2.imwrite(str(save_path), crop)
        print(f"[DONE] {sp} 裁剪输出: {out_dir}")

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="/home/fanh/Poker/dataset",
                        help="数据集根目录，含 train/valid/test")
    parser.add_argument("--project", type=str, default="/home/fanh/Poker",
                        help="项目根目录（保存 runs 和 crops）")
    parser.add_argument("--model-size", type=str, default="n", choices=["n","s","m","l","x"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=64,
                        help="批大小；设 0 让 YOLO 自动探测")
    parser.add_argument("--device", type=str, default=None,
                        help='GPU 设备，如 "0" 或 "0,1"; 留空则自动选择 0 或 "cpu"')
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--cache", action="store_true", help="训练时缓存数据到内存/磁盘")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--splits", nargs="+", default=["test"], help="要推理并裁剪的集合")
    parser.add_argument("--pad", type=float, default=0.10, help="裁剪 padding 比例")
    parser.add_argument("--skip-train", action="store_true", help="跳过训练，直接推理裁剪")
    parser.add_argument("--weights", type=str, default=None, help="已训练好的权重路径（跳过训练时必填或已有历史 best.pt）")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    assert dataset_dir.exists(), f"数据集不存在: {dataset_dir}"
    project_dir = Path(args.project)
    runs_dir = project_dir / "runs_detect"
    crop_dir = project_dir / "crops"
    data_yaml = project_dir / "data.yaml"

    # 自动选择 device
    if args.device is None:
        try:
            import torch
            if torch.cuda.is_available():
                args.device = "0"
            else:
                args.device = "cpu"
        except Exception:
            args.device = "cpu"
    print(f"[INFO] 使用 device = {args.device}")

    # 生成 data.yaml
    ensure_yaml(dataset_dir, data_yaml)

    # 训练 or 使用现有权重
    best = None
    if args.skip_train:
        if args.weights and Path(args.weights).exists():
            best = Path(args.weights)
        else:
            # 尝试复用最近的 best.pt
            cands = sorted(runs_dir.rglob("weights/best.pt"), key=os.path.getmtime, reverse=True)
            if not cands:
                raise FileNotFoundError("skip-train 模式下未提供 --weights，且找不到历史 best.pt")
            best = Path(cands[0])
        print(f"[INFO] 使用现有权重: {best}")
    else:
        best = train_model(
            data_yaml, runs_dir,
            model_size=args.model_size,
            epochs=args.epochs, imgsz=args.imgsz,
            batch=args.batch, device=args.device,
            workers=args.workers, cache=args.cache
        )

    # 推理 + 裁剪
    predict_and_crop(
        weights=best, dataset_dir=dataset_dir, splits=args.splits,
        crop_dir=crop_dir, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
        device=args.device, workers=args.workers, pad_ratio=args.pad
    )
    print("\n[ALL DONE] 训练/裁剪完成。")

if __name__ == "__main__":
    main()
