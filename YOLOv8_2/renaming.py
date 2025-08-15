# this code is for renaming the images and label. In the original format different folder has same name, which will cause problem in yolo format

import os
from pathlib import Path

TARGET_DIR = Path(r"D:\OneDrive - Rose-Hulman Institute of Technology\Rose-Hulman\course\CSSE\CSSE463\final project\git\Poker-Card-Recognition\YOLOv8_2\dataset\labels\single\train")


for file in TARGET_DIR.iterdir():
    if file.is_file():
        new_name = f"{file.stem}_single{file.suffix}"
        new_path = file.with_name(new_name)
        file.rename(new_path)
        print(f"Renamed: {file.name} → {new_name}")

print("All the files have been renamed")
