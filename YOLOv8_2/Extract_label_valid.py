import json
from pathlib import Path
from PIL import Image


JSON_FILE = Path(r"D:\OneDrive - Rose-Hulman Institute of Technology\Rose-Hulman\course\CSSE\CSSE463\final project\git\Poker-Card-Recognition\YOLOv8_2\dataset\splits\three_card_poker_class_level\val.json")
IMG_DIR = Path(r"D:\OneDrive - Rose-Hulman Institute of Technology\Rose-Hulman\course\CSSE\CSSE463\final project\git\Poker-Card-Recognition\YOLOv8_2\dataset\imgs\three_card_poker_class_level\val")
LABEL_OUT_DIR = Path(r"D:\OneDrive - Rose-Hulman Institute of Technology\Rose-Hulman\course\CSSE\CSSE463\final project\git\Poker-Card-Recognition\YOLOv8_2\dataset\labels\three_card_poker_class_level\val")
LABEL_OUT_DIR.mkdir(parents=True, exist_ok=True)


with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f" Processing {JSON_FILE.name} with {len(data)} entries...")

for key, item in data.items():
    img_path = item["img_path"]
    card_infos = item["card_points"]  # List of [ [points], class_id ]


    img_file = IMG_DIR / Path(img_path).name
    if not img_file.exists():
        print(f"Image not found: {img_file}")
        continue

    with Image.open(img_file) as img:
        W, H = img.size

    label_lines = []
    for card in card_infos:
        points, class_id = card
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_center = (x_min + x_max) / 2 / W
        y_center = (y_min + y_max) / 2 / H
        width = (x_max - x_min) / W
        height = (y_max - y_min) / H

        label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    label_file = LABEL_OUT_DIR / (Path(img_path).stem + ".txt")
    with open(label_file, "w") as f:
        f.write("\n".join(label_lines))

print("finished")
