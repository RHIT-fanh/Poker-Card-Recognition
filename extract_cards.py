import os
import json
from PIL import Image
from tqdm import tqdm  # 进度条

# 牌面类别映射
suit_map = {
    'H': 'heart',
    'D': 'diamond',
    'S': 'spade',
    'C': 'club',
    'JOKER': 'joker'
}

rank_map = {
    'A': 'A', '2': '2', '3': '3', '4': '4', '5': '5',
    '6': '6', '7': '7', '8': '8', '9': '9', '10': '10',
    'J': 'J', 'Q': 'Q', 'K': 'K', 'JOKER': 'joker'
}

# 路径设置（相对路径）
image_root = "../PokerImage/Images/Images"  # 原始图片所在目录
json_path = "../PokerImage/annotation.json"
output_root = "../PokerImage/dataset"

# 创建输出目录
for suit in suit_map.values():
    os.makedirs(os.path.join(output_root, 'suit', suit), exist_ok=True)
for rank in rank_map.values():
    os.makedirs(os.path.join(output_root, 'rank', rank), exist_ok=True)

# 读取 JSON
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 建立 image_id → 文件名映射
id_to_filename = {img['id']: img['file_name'] for img in data['images']}

# 遍历标注（加进度条）
for ann in tqdm(data['annotations'], desc="裁剪扑克中", unit="张"):
    image_id = ann['image_id']
    filename = id_to_filename.get(image_id)
    bbox = ann['bbox']  # [x, y, w, h]
    label = ann['category_id']

    # 从 categories 获取牌名
    category_name = None
    for cat in data['categories']:
        if cat['id'] == label:
            category_name = cat['name']
            break
    if not category_name:
        continue

    # Joker 特殊处理
    if 'JOKER' in category_name.upper():
        suit_name = 'joker'
        rank_name = 'joker'
    else:
        rank_part = category_name[:-1]
        suit_part = category_name[-1]
        suit_name = suit_map.get(suit_part)
        rank_name = rank_map.get(rank_part)

    if not suit_name or not rank_name:
        continue

    # 打开原图
    img_path = os.path.join(image_root, filename)
    if not os.path.exists(img_path):
        print(f"图片不存在: {img_path}")
        continue
    img = Image.open(img_path)

    # 裁剪整张牌
    x, y, w, h = bbox
    cropped = img.crop((x, y, x + w, y + h))

    # 保存到 suit 目录
    cropped.save(os.path.join(output_root, 'suit', suit_name, filename))

    # 保存到 rank 目录
    cropped.save(os.path.join(output_root, 'rank', rank_name, filename))

print("✅ 裁剪完成！所有图片已保存到 dataset/ 下。")
