# -*- coding: utf-8 -*-
import time, json, random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ===== 配置 =====
DATA_ROOT = Path("/home/fanh/Poker/dataset_cls")  # 分类数据根目录
OUT_DIR   = Path("/home/fanh/Poker/runs_cls")     # 输出目录
IMGSZ     = 224                                    # 可换 256/320
BATCH     = 128
EPOCHS    = 40
LR        = 1e-3
WD        = 1e-4
NUM_WORKERS = 16
SEED      = 42
DEVICE    = "cuda:0" if torch.cuda.is_available() else "cpu"
# ===============

def set_seed(seed=42):
    import numpy as np
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True  # 更快

def get_loaders():
    # 训练增强：适度即可（ROI已裁好）
    train_tf = transforms.Compose([
        transforms.Resize(int(IMGSZ*1.15)),
        transforms.RandomResizedCrop(IMGSZ, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.15, 0.15, 0.15, 0.05),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(IMGSZ*1.15)),
        transforms.CenterCrop(IMGSZ),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])

    train_ds = datasets.ImageFolder((DATA_ROOT/"train").as_posix(), transform=train_tf)
    valid_ds = datasets.ImageFolder((DATA_ROOT/"valid").as_posix(), transform=eval_tf)
    test_ds  = datasets.ImageFolder((DATA_ROOT/"test").as_posix(),  transform=eval_tf)

    train_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
    valid_ld = DataLoader(valid_ds, batch_size=BATCH, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
    test_ld  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
    return train_ld, valid_ld, test_ld, train_ds.classes

def build_model(num_classes: int):
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    return m

def one_epoch(model, loader, criterion, optimizer=None):
    train_mode = optimizer is not None
    model.train(mode=train_mode)
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        if train_mode:
            loss.backward()
            optimizer.step()
        loss_sum += loss.item() * y.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return loss_sum/total, correct/total

def main():
    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_ld, valid_ld, test_ld, classes = get_loaders()
    num_classes = len(classes)
    print(f"[INFO] num_classes={num_classes} | example map: {dict(list(enumerate(classes))[:5])}")

    model = build_model(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    best_path = OUT_DIR / "efficientnet_b0_best.pth"

    for epoch in range(1, EPOCHS+1):
        t0 = time.time()
        tr_loss, tr_acc = one_epoch(model, train_ld, criterion, optimizer)
        va_loss, va_acc = one_epoch(model, valid_ld, criterion, optimizer=None)
        scheduler.step()

        print(f"Epoch {epoch:03d}/{EPOCHS} | "
              f"train {tr_loss:.4f}/{tr_acc:.4f} | "
              f"valid {va_loss:.4f}/{va_acc:.4f} | "
              f"{time.time()-t0:.1f}s")

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({
                "state_dict": model.state_dict(),
                "num_classes": num_classes,
                "class_names": classes,
                "imgsz": IMGSZ,
                "acc": best_acc,
            }, best_path)
            print(f"[SAVE] best -> {best_path} (acc={best_acc:.4f})")

    # Test
    ckpt = torch.load(best_path, map_location=DEVICE)
    model = build_model(ckpt["num_classes"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    te_loss, te_acc = one_epoch(model, test_ld, criterion, optimizer=None)
    print(f"[TEST] loss={te_loss:.4f} acc={te_acc:.4f}")

if __name__ == "__main__":
    main()
