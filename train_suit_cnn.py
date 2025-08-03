import os
import random
import shutil
from glob import glob
import numpy as np
from PIL import Image, UnidentifiedImageError

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm.auto import tqdm  # 更智能的进度条（Windows/终端/Notebook均可）

# ========== 基本设置 ==========
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# 当前脚本目录（和你之前的代码同目录）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 源数据（四花色整牌图像）与划分后数据集目录（相对路径）
SRC_ROOT = os.path.normpath(os.path.join(BASE_DIR, "../PokerImage/dataset/suit"))
DST_ROOT = os.path.normpath(os.path.join(BASE_DIR, "../PokerImage/dataset_suit_split"))

ALLOWED_CLASSES = ["heart", "diamond", "spade", "club"]  # 只做四花色
SPLITS = (0.7, 0.15, 0.15)  # train/val/test

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# 训练参数
BATCH_SIZE = 32
NUM_EPOCHS = 15
LR = 1e-4
NUM_WORKERS = 0 if os.name == "nt" else 4  # Windows 下用 0，避免多进程问题
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 是否强制重建划分目录（设为 True 会删除并重建 dataset_suit_split/）
FORCE_RESPLIT = False


def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)


def collect_images_by_class(src_root, allowed_classes):
    """逐类收集文件路径，并过滤坏图像"""
    class_to_files = {}
    for cls in allowed_classes:
        cls_dir = os.path.join(src_root, cls)
        if not os.path.isdir(cls_dir):
            print(f"⚠️ 警告：找不到类别目录 {cls_dir}，将跳过该类。")
            class_to_files[cls] = []
            continue
        files = []
        for ext in IMG_EXTS:
            files.extend(glob(os.path.join(cls_dir, f"*{ext}")))
            files.extend(glob(os.path.join(cls_dir, f"*{ext.upper()}")))
        # 过滤损坏图片
        good_files = []
        for fp in files:
            try:
                with Image.open(fp) as im:
                    im.verify()
                good_files.append(fp)
            except (UnidentifiedImageError, OSError):
                print(f"坏图像或无法读取，已跳过：{fp}")
        random.shuffle(good_files)
        class_to_files[cls] = good_files
    return class_to_files


def split_and_copy(class_to_files, dst_root, splits=(0.7, 0.15, 0.15)):
    """按类别分层划分并复制到 train/val/test（带总进度条 + 每类小进度）"""
    if FORCE_RESPLIT and os.path.isdir(dst_root):
        shutil.rmtree(dst_root)

    train_root = os.path.join(dst_root, "train")
    val_root   = os.path.join(dst_root, "val")
    test_root  = os.path.join(dst_root, "test")
    for root in [train_root, val_root, test_root]:
        for cls in ALLOWED_CLASSES:
            safe_mkdir(os.path.join(root, cls))

    t_ratio, v_ratio, _ = splits

    # 计算整体文件数，用一个总进度条
    total_files = sum(len(v) for v in class_to_files.values())
    pbar_total = tqdm(total=total_files, desc="📦 划分并复制到 dataset_suit_split", unit="图", dynamic_ncols=True)

    for cls, files in class_to_files.items():
        n = len(files)
        n_train = int(n * t_ratio)
        n_val   = int(n * v_ratio)
        n_test  = n - n_train - n_val
        idx_train = list(range(0, n_train))
        idx_val   = list(range(n_train, n_train + n_val))
        idx_test  = list(range(n_train + n_val, n))

        # 每一类再给一个小进度条
        for split_name, idxs, root in [
            ("train", idx_train, train_root),
            ("val",   idx_val,   val_root),
            ("test",  idx_test,  test_root),
        ]:
            desc = f"{cls:7s} → {split_name:5s}"
            for i in tqdm(idxs, desc=desc, leave=False, dynamic_ncols=True):
                src_fp = files[i]
                basename = os.path.basename(src_fp)
                dst_fp = os.path.join(root, cls, basename)
                if not os.path.exists(dst_fp):
                    shutil.copy2(src_fp, dst_fp)
                pbar_total.update(1)

    pbar_total.close()

    # 打印各划分数量
    def count_images(root):
        total = 0
        per_cls = {}
        for cls in ALLOWED_CLASSES:
            c = sum(1 for e in os.scandir(os.path.join(root, cls)) if e.is_file())
            per_cls[cls] = c
            total += c
        return total, per_cls

    for split in ["train", "val", "test"]:
        total, per_cls = count_images(os.path.join(dst_root, split))
        print(f"{split.upper():5s} 总数: {total} | 逐类: {per_cls}")


def get_dataloaders(dst_root, img_size=224, batch_size=32, num_workers=0):
    # 数据增强与预处理（ImageNet 归一化）
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dir = os.path.join(dst_root, "train")
    val_dir   = os.path.join(dst_root, "val")
    test_dir  = os.path.join(dst_root, "test")

    train_set = datasets.ImageFolder(train_dir, transform=train_tf)
    val_set   = datasets.ImageFolder(val_dir,   transform=eval_tf)
    test_set  = datasets.ImageFolder(test_dir,  transform=eval_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_set.classes


def build_model(num_classes):
    # ResNet18 迁移学习
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"🧠 Train {epoch}/{total_epochs}", dynamic_ncols=True)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        avg_loss = running_loss / total if total > 0 else 0.0
        acc = correct / total if total > 0 else 0.0
        pbar.set_postfix(avg_loss=f"{avg_loss:.4f}", acc=f"{acc:.4f}")
    return running_loss / total, correct / total


def eval_one_epoch(model, loader, criterion, device, epoch, total_epochs, phase="Val"):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"🔎 {phase} {epoch}/{total_epochs}", dynamic_ncols=True)
    with torch.no_grad():
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            avg_loss = running_loss / total if total > 0 else 0.0
            acc = correct / total if total > 0 else 0.0
            pbar.set_postfix(avg_loss=f"{avg_loss:.4f}", acc=f"{acc:.4f}")
    return running_loss / total, correct / total


def plot_confusion_matrix(cm, classes, save_path):
    fig = plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest')  # 默认配色
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main():
    # 1) 收集 & 划分 & 复制（带总进度条）
    class_to_files = collect_images_by_class(SRC_ROOT, ALLOWED_CLASSES)
    split_and_copy(class_to_files, DST_ROOT, SPLITS)

    # 2) DataLoader
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        DST_ROOT, img_size=IMG_SIZE, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )
    num_classes = len(classes)
    print(f"类别顺序：{classes}")

    # 3) 模型与优化器
    model = build_model(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0
    best_model_path = os.path.join(BASE_DIR, "best_suit_resnet18.pth")

    # 4) 训练（外层 epoch 进度条）
    epoch_bar = tqdm(range(1, NUM_EPOCHS + 1), desc="📈 Epochs", dynamic_ncols=True)
    for epoch in epoch_bar:
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch, NUM_EPOCHS)
        va_loss, va_acc = eval_one_epoch(model, val_loader, criterion, DEVICE, epoch, NUM_EPOCHS, phase="Val")

        # 在 epoch 进度条上显示汇总指标
        epoch_bar.set_postfix(tr_acc=f"{tr_acc:.4f}", va_acc=f"{va_acc:.4f}")

        # 保存最优模型
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), best_model_path)

    print(f"✅ 训练完成。最佳验证准确率：{best_val_acc:.4f}，模型已保存：{best_model_path}")

    # 5) 测试集评估
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="🧪 Test", dynamic_ncols=True):
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(list(preds))
            all_labels.extend(list(labels.numpy()))

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    test_acc = (all_preds == all_labels).mean()

    print("\n======== 测试集结果 ========")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=classes, digits=4))

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    cm_path = os.path.join(BASE_DIR, "confusion_matrix_suit.png")
    plot_confusion_matrix(cm, classes, cm_path)
    print(f"混淆矩阵已保存：{cm_path}")


if __name__ == "__main__":
    main()
