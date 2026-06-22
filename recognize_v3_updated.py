import sys
import pickle
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

from insightface.app import FaceAnalysis
from insightface.utils import face_align
from collections import Counter

# =========================================================
# CONFIG
# =========================================================

MODEL_PATH = "checkpoints/best.pt"

MEAN_DB_PATH = "celeb_embeddings.pkl"
ALL_DB_PATH = "all_embeddings.pkl"

IMG_SIZE = 192
EMBEDDING_DIM = 512

TOP_K = 10
TOP_K_DEEP = 30

UNKNOWN_THRESHOLD = 0.65

CONFIDENCE_THRESHOLD = 0.85
MARGIN_THRESHOLD = 0.05

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# INSIGHTFACE
# =========================================================

print("Loading InsightFace...")

face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

face_app.prepare(
    ctx_id=1,
    det_size=(640,640)
)

print("✅ InsightFace Loaded")

# =========================================================
# MODEL
# =========================================================

class ArcMarginProduct(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()

        self.weight = nn.Parameter(
            torch.FloatTensor(out_features, in_features)
        )

    def forward(self, x, labels):
        pass


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

# =========================================================
# DATABASE
# =========================================================

print("Loading mean embeddings...")

with open(
    MEAN_DB_PATH,
    "rb"
) as f:

    mean_database = pickle.load(f)

print(
    f"✅ Mean DB Loaded ({len(mean_database)} celebrities)"
)

print("Loading all embeddings...")

with open(
    ALL_DB_PATH,
    "rb"
) as f:

    all_database = pickle.load(f)

print(
    f"✅ All DB Loaded ({len(all_database)} celebrities)"
)

ALL_EMBEDDINGS = []
ALL_LABELS = []
ALL_IMAGES = []

for celeb_name, items in all_database.items():

    for item in items:

        ALL_EMBEDDINGS.append(
            item["embedding"]
        )

        ALL_LABELS.append(
            celeb_name
        )

        ALL_IMAGES.append(
            item["image"]
        )

ALL_EMBEDDINGS = np.array(
    ALL_EMBEDDINGS,
    dtype=np.float32
)

print(
    f"✅ Flattened {len(ALL_EMBEDDINGS)} embeddings"
)

# =========================================================
# DETECT + ALIGN
# =========================================================

def align_face(image_path):

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(
            f"Cannot read image: {image_path}"
        )

    working_img = img

    faces = face_app.get(working_img)

    if len(faces) == 0:

        print("Trying larger image...")

        working_img = cv2.resize(
            img,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC
        )

        faces = face_app.get(working_img)

    if len(faces) == 0:

        raise ValueError(
            f"No face detected in {image_path}"
        )

    face = max(
        faces,
        key=lambda f:
        (f.bbox[2] - f.bbox[0]) *
        (f.bbox[3] - f.bbox[1])
    )

    aligned = face_align.norm_crop(
        working_img,
        landmark=face.kps,
        image_size=224
    )

    aligned = cv2.resize(
        aligned,
        (192, 192)
    )

    print("✅ Face detected and aligned")

    return aligned

    

# =========================================================
# EMBEDDING
# =========================================================

def get_embedding(image_path):

    try:

        aligned = align_face(image_path)

    except Exception as e:
        print(f"\nALIGN ERROR: {e}\n")

        print(
            "⚠ Face detection failed."
        )

        print(
            "⚠ Using full image fallback..."
        )

        aligned = cv2.imread(image_path)

        if aligned is None:

            raise ValueError(
                f"Cannot read image: {image_path}"
            )

        aligned = cv2.resize(
            aligned,
            (192, 192)
        )

    rgb = cv2.cvtColor(
        aligned,
        cv2.COLOR_BGR2RGB
    )

    rgb = rgb.astype(np.float32) / 255.0

    rgb = (rgb - 0.5) / 0.5

    rgb = np.transpose(
        rgb,
        (2, 0, 1)
    )

    tensor = torch.tensor(
        rgb,
        dtype=torch.float32
    ).unsqueeze(0)

    tensor = tensor.to(DEVICE)

    with torch.no_grad():

        emb = model(tensor)

    emb = emb.cpu().numpy()[0]

    emb /= np.linalg.norm(emb)

    return emb

# =========================================================
# SIMILARITY
# =========================================================

def cosine_similarity(a, b):

    return np.dot(a, b)

def fast_search(query_embedding):

    results = []

    for celeb_name, celeb_embedding in mean_database.items():

        score = float(
            np.dot(
                query_embedding,
                celeb_embedding
            )
        )

        results.append(
            (celeb_name, score)
        )

    results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return results

def deep_verify(query_embedding):

    scores = np.dot(
        ALL_EMBEDDINGS,
        query_embedding
    )

    winner_guess = ALL_LABELS[
        np.argmax(scores)
    ]

    effective_k = min(
        TOP_K_DEEP,
        len(all_database[winner_guess])
    )

    top_idx = np.argsort(
        scores
    )[::-1][:effective_k]

    top_scores = scores[top_idx]

    top_similarity = float(
        top_scores[0]
    )

    avg_similarity = float(
        np.mean(top_scores)
    )

    vote_names = []

    for idx in top_idx:

        vote_names.append(
            ALL_LABELS[idx]
        )

    votes = Counter(
        vote_names
    )

    winner_name, winner_votes = (
        votes.most_common(1)[0]
    )

    # NEW
    

    confidence = avg_similarity

    runner_name = None
    runner_votes = 0

    if len(votes) > 1:

        runner_name, runner_votes = (
            votes.most_common(2)[1]
        )

    return {

        "name": winner_name,

        "confidence": confidence,

        "votes": winner_votes,

        "runner_name": runner_name,

        "runner_votes": runner_votes,

        "top_similarity": top_similarity,

        "avg_similarity": avg_similarity,

        "effective_k": effective_k
    }

# =========================================================
# MAIN
# =========================================================

if len(sys.argv) < 2:

    print(
        "\nUsage:\n"
        "python recognize_v2.py image.jpg"
    )

    sys.exit()

image_path = sys.argv[1]

query_embedding = get_embedding(
    image_path
)

fast_results = fast_search(
    query_embedding
)

top1_name, top1_score = fast_results[0]
top2_name, top2_score = fast_results[1]

margin = (
    top1_score -
    top2_score
)

print("\n==============================")
print("FAST SEARCH")
print("==============================\n")

for i, (name, score) in enumerate(
    fast_results[:TOP_K],
    start=1
):

    print(
        f"{i}. "
        f"{name:<30} "
        f"{score*100:.2f}%"
    )

# UNKNOWN PERSON

if top1_score < UNKNOWN_THRESHOLD:

    print("\n------------------------------")

    print(
        "Prediction : UNKNOWN PERSON"
    )

    print(
        f"Confidence : "
        f"{top1_score*100:.2f}%"
    )

    sys.exit()

# FAST ACCEPT

if (
    top1_score >= 0.98
    and margin >= 0.20
):

    print("\n------------------------------")

    print(
        f"Prediction : {top1_name}"
    )

    print(
        f"Confidence : "
        f"{top1_score*100:.2f}%"
    )

    print(
        "Mode       : FAST"
    )

    sys.exit()

# DEEP VERIFY

else:

    print(
        "\n⚠ Running Deep Verification..."
    )

    result = deep_verify(
        query_embedding

    )

    if (
        result["avg_similarity"] < 0.93
        and top1_score < 0.98
    ):

        print(
            "\nPrediction : UNKNOWN PERSON"
        )

        print(
            f"Similarity : "
            f"{result['avg_similarity']*100:.2f}%"
        )

        sys.exit()

    print(
        "\n=============================="
    )

    print(
        "VERIFIED RESULT"
    )

    print(
        "==============================\n"
    )

    print(
        f"Prediction : "
        f"{result['name']}"
    )

    print(
        f"Confidence : "
        f"{result['confidence']*100:.2f}%"
    )

    print(
        "Mode       : VERIFIED"
    )

    print()

    print(
        f"Evidence   : "
        f"{result['votes']} / "
        f"{result['effective_k']} nearest matches"
    )

    print(
        f"Top Match  : "
        f"{result['top_similarity']*100:.2f}%"
    )

    print(
        f"Average    : "
        f"{result['avg_similarity']*100:.2f}%"
    )

    if result["runner_name"]:

        print(
            f"Runner-up  : "
            f"{result['runner_name']} "
            f"({result['runner_votes']} votes)"
        )