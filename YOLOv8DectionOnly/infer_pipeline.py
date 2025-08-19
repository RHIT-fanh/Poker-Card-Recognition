# -*- coding: utf-8 -*-
"""
一键推理：YOLO(只检测牌) + EfficientNet-B0(牌面分类)
- 输入：单张图片路径 或 文件夹
- 输出：可视化图、JSON（可选CSV），每个目标包含 det_conf、cls_prob、combined_conf=两者乘积
"""

import os, json, csv, time
from pathlib import Path
import argparse
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

# ---------- YOLO ----------
from ultralytics import YOLO

# ---------- EfficientNet-B0 ----------
import torch.nn as nn
from torchvision import models

def build_efficientnet_b0(num_classes: int):
    m = models.efficientnet_b0(weights=None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    return m

# ---------- Utils ----------
def load_classifier(weights_path: Path, meta_path: Path = None, device="cuda"):
    """
    兼容两种保存方式：
    A) 新推荐：weights_only + meta.json
       - weights_path: *.weights.pth
       - meta_path:    *.meta.json
    B) 旧方式：一个 .pth 里包含 state_dict 与元数据
    """
    if meta_path and meta_path.exists():
        meta = json.load(open(meta_path, "r"))
        num_classes = int(meta["num_classes"])
        class_names = meta.get("class_names", [str(i) for i in range(num_classes)])
        imgsz = int(meta.get("imgsz", 224))
        model = build_efficientnet_b0(num_classes).to(device)
        state = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return model, class_names, imgsz

    # 旧：单 .pth
    ckpt = torch.load(weights_path, map_location='cpu')
    num_classes = int(ckpt["num_classes"])
    class_names = ckpt.get("class_names", [str(i) for i in range(num_classes)])
    imgsz = int(ckpt.get("imgsz", 224))
    model = build_efficientnet_b0(num_classes).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, class_names, imgsz

def cls_preprocess(imgsz):
    return transforms.Compose([
        transforms.Resize(int(imgsz*1.15)),
        transforms.CenterCrop(imgsz),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def draw_box(img, xyxy, label, color=(0, 200, 0)):
    x1,y1,x2,y2 = map(int, xyxy)
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
    # 文本背景
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, max(0, y1- th - 6)), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img, label, (x1+3, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

def yolo_to_xyxy_det(box):
    # box.xyxy: tensor([[x1,y1,x2,y2]])
    return box.xyxy[0].tolist()

def expand_xyxy(xyxy, pad_ratio, W, H):
    x1,y1,x2,y2 = map(float, xyxy)
    w,h = x2-x1, y2-y1
    x1 -= w*pad_ratio; y1 -= h*pad_ratio
    x2 += w*pad_ratio; y2 += h*pad_ratio
    x1 = int(max(0, min(W-1, x1))); y1 = int(max(0, min(H-1, y1)))
    x2 = int(max(0, min(W-1, x2))); y2 = int(max(0, min(H-1, y2)))
    return [x1,y1,x2,y2]

def load_names_map(path):
    """
    可选：把分类ID映射成牌名（如 0->AH, 1->2H ...）
    文件格式支持两种：
      - JSON: {"0":"AH","1":"2H",...}
      - TXT/CSV: 每行 id,name
    """
    if path is None: return None
    p = Path(path)
    if not p.exists(): return None
    if p.suffix.lower()==".json":
        return {int(k):v for k,v in json.load(open(p, "r")).items()}
    # 简单CSV/TXT
    mapping = {}
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith("#"): continue
            parts = [x.strip() for x in line.replace(",", " ").split()]
            if len(parts) >= 2:
                mapping[int(parts[0])] = parts[1]
    return mapping or None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo",   type=str, required=True, help="YOLO 检测权重 .pt 路径")
    ap.add_argument("--cls_w",  type=str, required=True, help="分类器权重（.weights.pth 或 旧版 .pth）")
    ap.add_argument("--cls_meta", type=str, default=None, help="分类器 meta.json（若采用weights_only保存）")
    ap.add_argument("--source", type=str, required=True, help="图片/文件夹路径")
    ap.add_argument("--outdir", type=str, default="/home/fanh/Poker/out_infer", help="输出目录")
    ap.add_argument("--device", type=str, default=None, help='如 "0" 或 "cpu"')
    ap.add_argument("--imgsz",  type=int, default=640, help="YOLO 输入尺寸")
    ap.add_argument("--conf",   type=float, default=0.25, help="YOLO 置信度阈值")
    ap.add_argument("--iou",    type=float, default=0.45, help="YOLO NMS IoU 阈值")
    ap.add_argument("--pad",    type=float, default=0.10, help="裁剪ROI的留白比例")
    ap.add_argument("--names_map", type=str, default=None, help="可选：分类ID到牌名映射文件")
    ap.add_argument("--save_csv", action="store_true", help="同时保存CSV")
    args = ap.parse_args()

    # 设备
    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = args.device

    # 加载模型
    det_model = YOLO(args.yolo)
    cls_model, cls_names, cls_imgsz = load_classifier(Path(args.cls_w), Path(args.cls_meta) if args.cls_meta else None, device=device)
    tf_cls = cls_preprocess(cls_imgsz)
    id2name_map = load_names_map(args.names_map)

    # 输入集合
    src = Path(args.source)
    if src.is_dir():
        imgs = sorted([p for p in src.iterdir() if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp",".webp"]])
    else:
        imgs = [src]

    out_dir = ensure_dir(Path(args.outdir))
    vis_dir = ensure_dir(out_dir / "vis")
    json_dir = ensure_dir(out_dir / "json")
    csv_dir  = ensure_dir(out_dir / "csv") if args.save_csv else None

    for img_path in imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] skip unreadable: {img_path}")
            continue
        H, W = img.shape[:2]
        t0 = time.time()

        # ---------- 检测 ----------
        results = det_model.predict(
            source=str(img_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False
        )
        dets = []
        if results and len(results)>0 and results[0].boxes is not None:
            for box in results[0].boxes:
                xyxy = yolo_to_xyxy_det(box)
                det_conf = float(box.conf[0]) if box.conf is not None else 0.0
                # 单类检测，cls恒为0；这里保留字段
                dets.append({"xyxy": xyxy, "det_conf": det_conf})

        # ---------- 分类 ----------
        records = []
        vis_img = img.copy()
        for i, d in enumerate(dets):
            # 裁剪 + padding
            ex_xyxy = expand_xyxy(d["xyxy"], args.pad, W, H)
            x1,y1,x2,y2 = map(int, ex_xyxy)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0: 
                continue

            # 预处理并分类
            with torch.no_grad():
                pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                x = tf_cls(pil).unsqueeze(0).to(device)
                logits = cls_model(x)
                prob = torch.softmax(logits, dim=1)[0]
                cls_id = int(prob.argmax().item())
                cls_prob = float(prob[cls_id].item())

            name_show = id2name_map.get(cls_id, cls_names[cls_id] if cls_id < len(cls_names) else str(cls_id)) if id2name_map else (cls_names[cls_id] if cls_id < len(cls_names) else str(cls_id))
            combined = d["det_conf"] * cls_prob

            label = f"{name_show}  det:{d['det_conf']:.2f}  cls:{cls_prob:.2f}  p:{combined:.2f}"
            draw_box(vis_img, ex_xyxy, label)

            rec = {
                "index": i,
                "image": str(img_path),
                "xyxy": [int(x1),int(y1),int(x2),int(y2)],
                "det_conf": round(d["det_conf"], 6),
                "cls_id": cls_id,
                "cls_name": name_show,
                "cls_prob": round(cls_prob, 6),
                "combined_conf": round(combined, 6)
            }
            records.append(rec)

        # ---------- 保存 ----------
        stem = img_path.stem
        cv2.imwrite(str(vis_dir / f"{stem}_vis.jpg"), vis_img)
        with open(json_dir / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump({"image": str(img_path), "results": records}, f, ensure_ascii=False, indent=2)
        if csv_dir:
            with open(csv_dir / f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["index","x1","y1","x2","y2","det_conf","cls_id","cls_name","cls_prob","combined_conf"])
                for r in records:
                    x1,y1,x2,y2 = r["xyxy"]
                    w.writerow([r["index"], x1,y1,x2,y2, r["det_conf"], r["cls_id"], r["cls_name"], r["cls_prob"], r["combined_conf"]])
        dt = time.time()-t0
        print(f"[DONE] {img_path.name}: {len(records)} objs, saved, {dt:.2f}s")

if __name__ == "__main__":
    main()
