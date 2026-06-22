import os
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

import timm
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.utils import ModelEmaV2

from sklearn.model_selection import train_test_split
from collections import Counter

# =========================================================
# CONFIG
# =========================================================

DATA_DIR = "Celeb_face_data_aligned_V3"

CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best.pt")
LAST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "last.pt")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

IMG_SIZE = 192
EMBEDDING_DIM = 512

BATCH_SIZE = 12
EPOCHS = 60

LR = 1e-4
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# SEED
# =========================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)

# =========================================================
# DATA AUGMENTATION
# =========================================================

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomApply([
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        )
    ], p=0.5),

    transforms.RandomApply([
        transforms.GaussianBlur(3)
    ], p=0.15),

    transforms.RandomAffine(
        degrees=8,
        translate=(0.03, 0.03),
        scale=(0.95, 1.05)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# =========================================================
# LOAD DATASET
# =========================================================

full_dataset = datasets.ImageFolder(DATA_DIR)

class_names = full_dataset.classes
num_classes = len(class_names)

targets = full_dataset.targets

indices = list(range(len(full_dataset)))

train_idx, val_idx = train_test_split(
    indices,
    test_size=0.1,
    stratify=targets,
    random_state=SEED
)

train_dataset = torch.utils.data.Subset(
    datasets.ImageFolder(DATA_DIR, transform=train_transform),
    train_idx
)

val_dataset = torch.utils.data.Subset(
    datasets.ImageFolder(DATA_DIR, transform=val_transform),
    val_idx
)

# =========================================================
# BALANCED SAMPLER
# =========================================================

train_targets = [targets[i] for i in train_idx]

class_count = Counter(train_targets)

weights = [
    1.0 / class_count[t]
    for t in train_targets
]

sampler = WeightedRandomSampler(
    weights,
    num_samples=len(weights),
    replacement=True
)

# =========================================================
# DATALOADERS
# =========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    
    
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    
    
)

# =========================================================
# ARC FACE
# =========================================================

class ArcMarginProduct(nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        s=30.0,
        m=0.35
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.FloatTensor(out_features, in_features)
        )

        nn.init.xavier_uniform_(self.weight)

        self.s = s
        self.m = m

    def forward(self, embeddings, labels):

        cosine = F.linear(
            F.normalize(embeddings),
            F.normalize(self.weight)
        )

        theta = torch.acos(
            torch.clamp(cosine, -1 + 1e-7, 1 - 1e-7)
        )

        target_logits = torch.cos(theta + self.m)

        one_hot = torch.zeros_like(cosine)

        one_hot.scatter_(
            1,
            labels.view(-1, 1),
            1
        )

        logits = (
            one_hot * target_logits
            + (1 - one_hot) * cosine
        )

        logits *= self.s

        return logits

# =========================================================
# MODEL
# =========================================================

class FaceModel(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224.ms_in22k",
            pretrained=True,
            num_classes=0,
            img_size=192
        )

        backbone_out = self.backbone.num_features

        self.embedding = nn.Sequential(
            nn.Linear(backbone_out, EMBEDDING_DIM),
            nn.BatchNorm1d(EMBEDDING_DIM),
            nn.PReLU(),
            nn.Dropout(0.3)
        )

        self.arcface = ArcMarginProduct(
            EMBEDDING_DIM,
            num_classes
        )

    def forward(self, x, labels=None):

        feats = self.backbone(x)

        emb = self.embedding(feats)

        emb = F.normalize(emb)

        if labels is not None:
            logits = self.arcface(emb, labels)
            return logits, emb

        return emb

# =========================================================
# INIT
# =========================================================

model = FaceModel(num_classes).to(DEVICE)

model = model.to(memory_format=torch.channels_last)

ema_model = ModelEmaV2(
    model,
    decay=0.999
)

criterion = nn.CrossEntropyLoss(
    label_smoothing=0.1
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=5,
    T_mult=2
)

scaler = GradScaler("cuda")

# =========================================================
# RESUME
# =========================================================

start_epoch = 0
best_acc = 0.0

if os.path.exists(LAST_MODEL_PATH):

    ckpt = torch.load(
        LAST_MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(ckpt["model"])

    optimizer.load_state_dict(ckpt["optimizer"])

    scheduler.load_state_dict(ckpt["scheduler"])

    scaler.load_state_dict(ckpt["scaler"])

    start_epoch = ckpt["epoch"] + 1
    best_acc = ckpt["best_acc"]

    print(f"\n🔁 Resumed from epoch {start_epoch}")

# =========================================================
# TRAIN
# =========================================================

print("\n🎬 Training Started")
print(f"Device: {DEVICE}")

for epoch in range(start_epoch, EPOCHS):

    # =========================
    # TRAIN
    # =========================

    model.train()

    train_loss = 0.0

    for imgs, labels in tqdm(train_loader):

        imgs = imgs.to(
        DEVICE,
        non_blocking=True,
        memory_format=torch.channels_last
    )
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", dtype=torch.float16):

            logits, _ = model(imgs, labels)

            loss = criterion(logits, labels)

        scaler.scale(loss).backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        scaler.step(optimizer)
        scaler.update()

        ema_model.update(model)

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # =========================
    # VALIDATION
    # =========================

    ema_model.module.eval()

    val_loss = 0.0

    correct = 0
    total = 0

    with torch.no_grad():

        for imgs, labels in val_loader:

            imgs = imgs.to(
            DEVICE,
            non_blocking=True,
            memory_format=torch.channels_last
        )
            labels = labels.to(DEVICE, non_blocking=True)

            with autocast(device_type="cuda", dtype=torch.float16):

                logits, _ = ema_model.module(imgs, labels)

                loss = criterion(logits, labels)

            val_loss += loss.item()

            preds = logits.argmax(dim=1)

            

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(val_loader)

    val_acc = correct / total

    scheduler.step()

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"train_loss={train_loss:.4f} | "
        f"val_loss={val_loss:.4f} | "
        f"val_acc={val_acc:.4f}"
    )

    # =========================
    # SAVE BEST
    # =========================

    if val_acc > best_acc:

        best_acc = val_acc

        torch.save({
            "epoch": epoch,
            "model": ema_model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_acc": best_acc
        }, BEST_MODEL_PATH)

        print("🔥 New best model saved")

    # =========================
    # SAVE LAST
    # =========================

    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_acc": best_acc
    }, LAST_MODEL_PATH)

print("\n✅ Training Complete")
print(f"🏆 Best Accuracy: {best_acc:.4f}")