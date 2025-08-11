
import glob
import csv
from ultralytics import YOLO

# path
IMG_DIR  = r"D:\OneDrive - Rose-Hulman Institute of Technology\Rose-Hulman\course\CSSE\CSSE463\final project\git\Poker-Card-Recognition\YOLOv8\dataset\test\images"
GT_DIR   = r"D:\OneDrive - Rose-Hulman Institute of Technology\Rose-Hulman\course\CSSE\CSSE463\final project\git\Poker-Card-Recognition\YOLOv8\dataset\test\labels"
CSV_PATH = r"D:\OneDrive - Rose-Hulman Institute of Technology\Rose-Hulman\course\CSSE\CSSE463\final project\git\Poker-Card-Recognition\YOLOv8\evaluation_model.csv"

MODEL_PATH = r"D:\OneDrive - Rose-Hulman Institute of Technology\Rose-Hulman\course\CSSE\CSSE463\final project\git\Poker-Card-Recognition\YOLOv8\Model\weights\best.pt"
CONF_THRES = 0.50
IMG_EXTS = (".jpg")

# read ground truthes
def load_gt_classes(txt_path):
    
    classes = set()
    if not os.path.exists(txt_path):
        return classes
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                cid = int(float(parts[0]))
                classes.add(cid)
            except:
                pass
    return classes

def ids_to_names(id_set, id2name):
    names = []
    for cid in sorted(id_set):
        names.append(str(id2name.get(cid, cid)))
    return "|".join(names)


model = YOLO(MODEL_PATH)
id2name = model.names if hasattr(model, "names") else {}

rows = []
img_paths = sorted([p for p in glob.glob(os.path.join(IMG_DIR, "*")) if p.lower().endswith(IMG_EXTS)])

for img_path in img_paths:
    stem = os.path.splitext(os.path.basename(img_path))[0]
    gt_txt = os.path.join(GT_DIR, f"{stem}.txt")

    gt_classes = load_gt_classes(gt_txt)

    # predicting with threshold 50%
    res = model.predict(img_path, conf=CONF_THRES, verbose=False)[0]
    pred_classes = set()
    if res.boxes is not None and len(res.boxes) > 0:
        data = res.boxes.data.cpu().numpy()
        for x1, y1, x2, y2, conf, cls_id in data:
            if conf >= CONF_THRES:
                pred_classes.add(int(cls_id))

    
    inter = gt_classes & pred_classes
    missing = gt_classes - pred_classes
    extra   = pred_classes - gt_classes # detecting false positives

    has_false_positive = len(extra) > 0

    recall = len(inter) / len(gt_classes) if len(gt_classes) > 0 else (1.0 if len(pred_classes) == 0 else 0.0)
    precision = len(inter) / len(pred_classes) if len(pred_classes) > 0 else (1.0 if len(gt_classes) == 0 else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # col headers 
    rows.append({
        "image": os.path.basename(img_path),
        "gt_classes": ids_to_names(gt_classes, id2name),
        "pred_classes": ids_to_names(pred_classes, id2name),
        "missing_classes": ids_to_names(missing, id2name),
        "extra_classes": ids_to_names(extra, id2name),
        "has_false_positive": has_false_positive,
        "class_recall": f"{recall:.4f}",
        "class_precision": f"{precision:.4f}",
        "class_f1": f"{f1:.4f}",
    })

fieldnames = ["image","gt_classes","pred_classes","missing_classes","extra_classes","has_false_positive","class_recall","class_precision","class_f1"]
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Done. CSV -> {CSV_PATH}")
