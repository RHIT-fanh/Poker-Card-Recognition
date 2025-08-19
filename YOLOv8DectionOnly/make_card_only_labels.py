from pathlib import Path

DATASET = Path("/home/fanh/Poker/dataset")  # 你的数据集根目录
SPLITS = ["train", "valid", "test"]

for sp in SPLITS:
    src = DATASET / sp / "labels"
    dst = DATASET / sp / "labels_card"
    dst.mkdir(parents=True, exist_ok=True)

    for txt_file in src.glob("*.txt"):
        lines_out = []
        for line in txt_file.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            parts[0] = "0"  # 强制改为类别0
            lines_out.append(" ".join(parts))
        (dst / txt_file.name).write_text("\n".join(lines_out))

print("✅ 单类标签已生成到 labels_card/")
