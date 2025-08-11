# -*- coding: utf-8 -*-
"""
训练 YOLOv8（水平框）并在 test/images 上推理，自动裁剪 ROI 到 crops/ 下。
- 会自动生成 data.yaml
- 训练完成后，用 best.pt 做推理
- 每个检测保存一张裁剪图（可设置边缘留白 padding）
"""
import os, sys, yaml, math
from pathlib import Path
import cv2
from tqdm import tqdm

# ===== 配置区（按需改） =====
# 项目根：脚本所在目录
BASE_DIR = Path(__file__).resolve().parent
# 数据集根：与当前脚本同级的 dataset 目录
DATASET_DIR = BASE_DIR / "dataset"

# 训练输出目录
RUNS_DIR = BASE_DIR / "runs_detect"
# 模型大小：n/s/m/l/x
MODEL_SIZE = "n"
# 训练超参
EPOCHS = 100
IMGSZ = 640
BATCH = 16
# 推理阈值
CONF = 0.25
IOU = 0.45
# 裁剪图输出目录
CROP_DIR = BASE_DIR / "crops"
# 裁剪时给 bbox 的相对 padding（例如 0.10 = 四周各加10%）
PAD_RATIO = 0.10
# 只对 test 集推理；如需对 train/valid 也裁剪可加到列表里
PRED_SPLITS = ["test"]
# =========================

def ensure_yaml(dataset_dir: Path) -> Path:
    """自动生成 data.yaml（names 如果不知道，就按 0..N-1 占位）"""
    yaml_path = BASE_DIR / "data.yaml"
    train_dir = (dataset_dir / "train").as_posix()
    val_dir = (dataset_dir / "valid").as_posix()
    test_dir = (dataset_dir / "test").as_posix()

    # 估计类别数：扫描一个 labels 目录，取最大 class_id+1
    def guess_num_classes(lbl_dir: Path) -> int:
        import glob
        import re
        max_id = -1
        files = glob.glob(str(lbl_dir / "*.txt"))
        for p in files[:500]:  # 采样防止太慢
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for ln in f:
                        parts = ln.strip().split()
                        if len(parts) >= 5:
                            cid = int(float(parts[0]))
                            max_id = max(max_id, cid)
            except:
                pass
        return max_id + 1 if max_id >= 0 else 1

    nc = guess_num_classes(dataset_dir / "train" / "labels")
    names = [str(i) for i in range(nc)]

    data = {
        "path": dataset_dir.as_posix(),
        "train": train_dir,
        "val": val_dir,
        "test": test_dir,
        "names": names
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    print(f"[INFO] data.yaml 已生成：{yaml_path}")
    print(f"       类别数估计：{nc}（names={names[:10]}{'...' if len(names)>10 else ''}）")
    return yaml_path

def train_if_needed(data_yaml: Path, model_size: str = "n") -> Path:
    """如无已训好的 best.pt，则训练；返回 best.pt 路径"""
    from ultralytics import YOLO
    weights = f"yolov8{model_size}.pt"
    runs_dir = RUNS_DIR
    runs_dir.mkdir(parents=True, exist_ok=True)

    # 若已有最近一次训练的 best.pt，直接复用
    existed_best = sorted(runs_dir.rglob("weights/best.pt"), key=os.path.getmtime, reverse=True)
    if existed_best:
        print(f"[INFO] 发现已训练模型：{existed_best[0]}")
        return Path(existed_best[0])

    print("[INFO] 开始训练 YOLOv8...")
    model = YOLO(weights)
    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        project=str(runs_dir),
        name=f"yolov8{model_size}_train",
        exist_ok=True,
        device=0 if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
    )
    best = runs_dir / f"yolov8{model_size}_train" / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError("未找到 best.pt，请检查训练日志。")
    return best

def clip(v, lo, hi):
    return max(lo, min(hi, v))

def expand_box(xyxy, pad_ratio, W, H):
    x1, y1, x2, y2 = map(float, xyxy)
    w, h = x2 - x1, y2 - y1
    px, py = w * pad_ratio, h * pad_ratio
    nx1 = int(clip(x1 - px, 0, W - 1))
    ny1 = int(clip(y1 - py, 0, H - 1))
    nx2 = int(clip(x2 + px, 0, W - 1))
    ny2 = int(clip(y2 + py, 0, H - 1))
    return nx1, ny1, nx2, ny2

def run_predict_and_crop(weights: Path, dataset_dir: Path, splits, crop_dir: Path):
    from ultralytics import YOLO
    model = YOLO(str(weights))
    crop_dir.mkdir(parents=True, exist_ok=True)

    for sp in splits:
        img_dir = dataset_dir / sp / "images"
        out_dir = crop_dir / sp
        out_dir.mkdir(parents=True, exist_ok=True)

        # 用 Ultralytics 内置推理拿到检测框
        print(f"[INFO] 推理 split={sp} ...")
        results = model.predict(
            source=str(img_dir),
            conf=CONF,
            iou=IOU,
            imgsz=IMGSZ,
            stream=True,     # 流式避免一次性加载全部
            device=0 if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu',
            verbose=False
        )

        # 类别名
        names = model.names if hasattr(model, "names") else {}
        # 遍历每张图的结果
        for r in tqdm(results, desc=f"cropping {sp}"):
            if r.path is None or r.orig_img is None:
                continue
            img_path = Path(r.path)
            img = r.orig_img  # numpy array, BGR
            H, W = img.shape[:2]
            stem = img_path.stem

            if r.boxes is None or len(r.boxes) == 0:
                continue

            for i, box in enumerate(r.boxes):
                xyxy = box.xyxy[0].tolist()    # [x1,y1,x2,y2]
                conf = float(box.conf[0]) if box.conf is not None else 0.0
                cls_id = int(box.cls[0]) if box.cls is not None else -1
                # 扩张一下 bbox 再裁剪
                x1,y1,x2,y2 = expand_box(xyxy, PAD_RATIO, W, H)
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # 以类别建子文件夹（可选）
                cls_name = names.get(cls_id, str(cls_id))
                save_dir = out_dir / f"class_{cls_name}"
                save_dir.mkdir(parents=True, exist_ok=True)

                save_name = f"{stem}_det{i:02d}_c{cls_id}_p{conf:.2f}.jpg"
                save_path = save_dir / save_name
                cv2.imwrite(str(save_path), crop)

        print(f"[DONE] {sp} 裁剪完成，输出在：{out_dir}")

def main():
    assert DATASET_DIR.exists(), f"DATASET_DIR 不存在: {DATASET_DIR}"
    data_yaml = ensure_yaml(DATASET_DIR)
    best = train_if_needed(data_yaml, MODEL_SIZE)
    run_predict_and_crop(best, DATASET_DIR, PRED_SPLITS, CROP_DIR)
    print("\n[ALL DONE] 训练 + 推理 + 裁剪 完成。")

if __name__ == "__main__":
    main()
