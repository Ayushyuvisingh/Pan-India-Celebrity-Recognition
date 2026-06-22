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

OUTPUT_FILE = "all_embeddings.pkl"

IMG_SIZE = 192
EMBEDDING_DIM = 512

BATCH_SIZE = 64

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# ARC FACE
# =========================================================

class ArcMarginProduct(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()

        self.weight = nn.Parameter(
            torch.FloatTensor(
                out_features,
                in_features
            )
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
            nn.Linear(
                backbone_out,
                EMBEDDING_DIM
            ),
            nn.BatchNorm1d(
                EMBEDDING_DIM
            ),
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

print("Loading model...")

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
print(f"Device: {DEVICE}")

# =========================================================
# PREPROCESS
# =========================================================

def preprocess_image(img_path):

    img = cv2.imread(
        str(img_path)
    )

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

    img = img.astype(
        np.float32
    ) / 255.0

    img = (
        img - 0.5
    ) / 0.5

    img = np.transpose(
        img,
        (2, 0, 1)
    )

    return img


# =========================================================
# BATCH EMBEDDING
# =========================================================

def get_embeddings_batch(images):

    batch = torch.tensor(
        np.stack(images),
        dtype=torch.float32
    ).to(DEVICE)

    with torch.no_grad():

        embeddings = model(batch)

    embeddings = embeddings.cpu().numpy()

    embeddings /= np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    return embeddings


# =========================================================
# BUILD DATABASE
# =========================================================

database = {}

root = Path(DATASET_DIR)

celebs = sorted([
    d for d in root.iterdir()
    if d.is_dir()
])

print(
    f"\nCelebrities Found: {len(celebs)}"
)

total_images = 0

for celeb_dir in tqdm(
    celebs,
    desc="Building Database"
):

    images = (
        list(celeb_dir.glob("*.jpg"))
        + list(celeb_dir.glob("*.jpeg"))
        + list(celeb_dir.glob("*.png"))
    )

    celeb_data = []

    batch_imgs = []
    batch_names = []

    for img_path in images:

        img = preprocess_image(
            img_path
        )

        if img is None:
            continue

        batch_imgs.append(img)
        batch_names.append(
            img_path.name
        )

        if len(batch_imgs) == BATCH_SIZE:

            embeddings = get_embeddings_batch(
                batch_imgs
            )

            for emb, name in zip(
                embeddings,
                batch_names
            ):

                celeb_data.append({

                    "embedding": emb,

                    "image": name

                })

                total_images += 1

            batch_imgs = []
            batch_names = []

    # Remaining images

    if len(batch_imgs) > 0:

        embeddings = get_embeddings_batch(
            batch_imgs
        )

        for emb, name in zip(
            embeddings,
            batch_names
        ):

            celeb_data.append({

                "embedding": emb,

                "image": name

            })

            total_images += 1

    if len(celeb_data) > 0:

        database[
            celeb_dir.name
        ] = celeb_data


# =========================================================
# SAVE
# =========================================================

print(
    f"\n✅ Celebrities Stored: {len(database)}"
)

print(
    f"✅ Total Image Embeddings: {total_images}"
)

with open(
    OUTPUT_FILE,
    "wb"
) as f:

    pickle.dump(
        database,
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )

print(
    f"✅ Saved: {OUTPUT_FILE}"
)