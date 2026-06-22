import os
import pickle
from pathlib import Path

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

from tqdm import tqdm

# =========================================================
# CONFIG
# =========================================================

DATASET_DIR = "Celeb_face_data_aligned_V3"

MODEL_PATH = "checkpoints/best.pt"

OUTPUT_FILE = "celeb_embeddings.pkl"

IMG_SIZE = 192
EMBEDDING_DIM = 512

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# ARC FACE
# =========================================================

class ArcMarginProduct(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()

        self.weight = nn.Parameter(
            torch.FloatTensor(out_features, in_features)
        )

    def forward(self, x, labels):
        pass

# =========================================================
# MODEL
# =========================================================

class FaceModel(nn.Module):

    def __init__(self, num_classes=363):

        super().__init__()

        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224.ms_in22k",
            pretrained=False,
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

    def forward(self, x):

        feats = self.backbone(x)

        emb = self.embedding(feats)

        emb = F.normalize(emb)

        return emb

# =========================================================
# LOAD MODEL
# =========================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model = FaceModel()

model.load_state_dict(
    checkpoint["model"],
    strict=False
)

model.to(DEVICE)
model.eval()

print("✅ Model Loaded")

# =========================================================
# IMAGE → EMBEDDING
# =========================================================

def get_embedding(img_path):

    img = cv2.imread(str(img_path))

    if img is None:
        return None

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    img = cv2.resize(
        img,
        (IMG_SIZE, IMG_SIZE)
    )

    img = img.astype(np.float32) / 255.0

    img = (img - 0.5) / 0.5

    img = np.transpose(
        img,
        (2, 0, 1)
    )

    img = torch.tensor(
        img,
        dtype=torch.float32
    ).unsqueeze(0)

    img = img.to(DEVICE)

    with torch.no_grad():

        emb = model(img)

    emb = emb.cpu().numpy()[0]

    emb = emb / np.linalg.norm(emb)

    return emb

# =========================================================
# BUILD DATABASE
# =========================================================

database = {}

root = Path(DATASET_DIR)

celebs = sorted([
    d for d in root.iterdir()
    if d.is_dir()
])

print(f"\nCelebrities Found: {len(celebs)}\n")

for celeb_dir in tqdm(celebs):

    embeddings = []

    images = (
        list(celeb_dir.glob("*.jpg"))
        + list(celeb_dir.glob("*.jpeg"))
        + list(celeb_dir.glob("*.png"))
    )

    for img_path in images:

        emb = get_embedding(img_path)

        if emb is not None:
            embeddings.append(emb)

    if len(embeddings) == 0:
        continue

    mean_embedding = np.mean(
        embeddings,
        axis=0
    )

    mean_embedding /= np.linalg.norm(
        mean_embedding
    )

    database[
        celeb_dir.name
    ] = mean_embedding

print(
    f"\n✅ Celebrities Stored: {len(database)}"
)

with open(
    OUTPUT_FILE,
    "wb"
) as f:

    pickle.dump(
        database,
        f
    )

print(
    f"✅ Saved: {OUTPUT_FILE}"
)