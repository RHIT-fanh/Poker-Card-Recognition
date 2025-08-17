# This code is for generating the train.txt and val.txt image path file YOLO needs

from pathlib import Path


base_path = Path("D:/OneDrive - Rose-Hulman Institute of Technology/Rose-Hulman/course/CSSE/CSSE463/final project/git/Poker-Card-Recognition/YOLOv8_2/dataset")
train_images_path = base_path / "train" / "images"
val_images_path = base_path / "val" / "images"


train_txt = base_path / "train.txt"
val_txt = base_path / "val.txt"


def generate_txt(images_dir: Path, output_txt: Path):
    image_files = sorted(images_dir.glob("*.*"))  
    with open(output_txt, "w") as f:
        for img in image_files:
            f.write(str(img.resolve()) + "\n")  

generate_txt(train_images_path, train_txt)
generate_txt(val_images_path, val_txt)

print("Successfully generated")
