import json
from pathlib import Path
import shutil

# === ✅ 自定义路径 ===
IMG_DIR = Path(r"D:\OneDrive - Rose-Hulman Institute of Technology\Rose-Hulman\course\CSSE\CSSE463\final project\git\Poker-Card-Recognition\YOLOv8\dataset\special\imgs\three_card_poker_class_level")
JSON_DIR = Path(r"D:\OneDrive - Rose-Hulman Institute of Technology\Rose-Hulman\course\CSSE\CSSE463\final project\git\Poker-Card-Recognition\YOLOv8\dataset\special\splits\three_card_poker_class_level")

# 输出目录
TRAIN_DIR = IMG_DIR / "train"
VAL_DIR = IMG_DIR / "val"

# 创建输出文件夹
TRAIN_DIR.mkdir(parents=True, exist_ok=True)
VAL_DIR.mkdir(parents=True, exist_ok=True)

# 处理函数
def move_images(json_path, target_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)

    for sample in data.values():
        img_path = sample['img_path']  # 相对路径，例如 imgs/single/1234.png
        img_name = Path(img_path).name  # 提取文件名 1234.png
        src_path = IMG_DIR / img_name
        dst_path = target_dir / img_name

        if src_path.exists():
            shutil.copy2(src_path, dst_path)
        else:
            print(f"❌ 未找到图片: {src_path}")

# 执行分类
move_images(JSON_DIR / "train.json", TRAIN_DIR)
move_images(JSON_DIR / "val.json", VAL_DIR)

print("✅ 图片分类完成！")
