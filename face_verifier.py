import pickle
import numpy as np
from collections import Counter

from insightface.app import FaceAnalysis

# -------------------------
# CONFIG
# -------------------------

ALL_EMBEDDINGS_PATH = r"D:\Projects\ML\bollywood_cleb\all_embeddings.pkl"

TOP_K = 20

# -------------------------
# LOAD INSIGHTFACE
# -------------------------

print("Loading InsightFace...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

print("InsightFace Ready")

# -------------------------
# LOAD DATABASE
# -------------------------

print("Loading embeddings...")

with open(
    ALL_EMBEDDINGS_PATH,
    "rb"
) as f:

    all_database = pickle.load(f)

ALL_EMBEDDINGS = []
ALL_LABELS = []

for celeb, entries in all_database.items():

    for item in entries:

        emb = item["embedding"]

        emb = emb / np.linalg.norm(emb)

        ALL_EMBEDDINGS.append(emb)

        ALL_LABELS.append(celeb)

ALL_EMBEDDINGS = np.array(
    ALL_EMBEDDINGS,
    dtype=np.float32
)

print(
    f"Loaded {len(ALL_LABELS)} embeddings"
)

# -------------------------
# FACE EMBEDDING
# -------------------------

def get_embedding(image_bgr):

    faces = app.get(image_bgr)

    if len(faces) != 1:

        return None

    emb = faces[0].embedding

    emb = emb / np.linalg.norm(emb)

    return emb

# -------------------------
# TARGET VERIFICATION
# -------------------------

def verify_target_identity(
    query_embedding,
    target_name,
    min_votes=12,
    min_similarity=0.60
):

    scores = np.dot(
        ALL_EMBEDDINGS,
        query_embedding
    )

    top_idx = np.argsort(
        scores
    )[::-1][:TOP_K]

    votes = []
    sims = []

    for idx in top_idx:

        votes.append(
            ALL_LABELS[idx]
        )

        sims.append(
            float(scores[idx])
        )

    target_votes = sum(
        1
        for v in votes
        if v == target_name
    )

    target_sims = [

        sims[i]

        for i, v in enumerate(votes)

        if v == target_name

    ]

    avg_sim = (

        float(np.mean(target_sims))

        if len(target_sims)

        else 0

    )

    accepted = (

        target_votes >= min_votes

        and

        avg_sim >= min_similarity

    )

    return {

        "accepted": accepted,

        "target_votes": target_votes,

        "avg_similarity": avg_sim

    }