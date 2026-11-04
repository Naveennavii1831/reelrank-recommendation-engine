
from pathlib import Path
import json

import numpy as np
import pandas as pd
import faiss


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

TRAIN_FILE = DATA / "train.csv"
VALIDATION_FILE = DATA / "validation.csv"
EMBEDDINGS_FILE = MODELS / "retrieval_embeddings.npz"
INDEX_FILE = MODELS / "movie_faiss.index"
MOVIE_IDS_FILE = MODELS / "faiss_movie_ids.npy"

REPORTS.mkdir(parents=True, exist_ok=True)

MAX_USERS = 1000
SEED = 42
RELEVANT_RATING = 4.0
K_VALUES = (10, 20)


def recall_at_k(recommended, relevant, k):
    if not relevant:
        return 0.0
    return len(set(recommended[:k]) & relevant) / len(relevant)


def ndcg_at_k(recommended, relevant, k):
    if not relevant:
        return 0.0

    dcg = sum(
        1.0 / np.log2(rank + 1)
        for rank, movie_id in enumerate(recommended[:k], start=1)
        if movie_id in relevant
    )

    ideal_count = min(k, len(relevant))
    ideal_dcg = sum(
        1.0 / np.log2(rank + 1)
        for rank in range(1, ideal_count + 1)
    )

    return dcg / ideal_dcg if ideal_dcg else 0.0


print("Loading embeddings and FAISS index...")

with np.load(EMBEDDINGS_FILE, allow_pickle=False) as data:
    user_ids = data["user_ids"].astype(str)
    user_embeddings = data["user_embeddings"].astype(np.float32)

index = faiss.read_index(str(INDEX_FILE))
movie_ids = np.load(MOVIE_IDS_FILE, allow_pickle=False).astype(str)

user_position = {
    user_id: position
    for position, user_id in enumerate(user_ids)
}

known_users = set(user_ids)
known_movies = set(movie_ids)

print(f"Known users: {len(user_ids):,}")
print(f"Indexed movies: {index.ntotal:,}")

# Load only the columns needed from the full chronological training set.
print("\nLoading full training history...")
train = pd.read_csv(
    TRAIN_FILE,
    usecols=["userId", "movieId"]
    if "rating" not in pd.read_csv(TRAIN_FILE, nrows=0).columns
    else ["userId", "movieId", "rating"]
)

train["userId"] = train["userId"].astype(str)
train["movieId"] = train["movieId"].astype(str)

# Exclude all movies appearing in the full training history.
# This is the correct history filter for a chronological split.
seen_by_user = train.groupby("userId")["movieId"].apply(set).to_dict()

del train

print("Full training history loaded.")

print("\nLoading validation positives...")
validation = pd.read_csv(
    VALIDATION_FILE,
    usecols=["userId", "movieId", "rating"]
)

validation["userId"] = validation["userId"].astype(str)
validation["movieId"] = validation["movieId"].astype(str)

validation = validation[validation["rating"] >= RELEVANT_RATING]

# Evaluate only warm-start users and movies that the retrieval model knows.
validation = validation[
    validation["userId"].isin(known_users)
    & validation["movieId"].isin(known_movies)
]

relevant_by_user = (
    validation.groupby("userId")["movieId"].apply(set).to_dict()
)

eligible_users = [
    user_id
    for user_id, relevant in relevant_by_user.items()
    if relevant
]

rng = np.random.default_rng(SEED)

if len(eligible_users) > MAX_USERS:
    selected_users = sorted(
        rng.choice(
            eligible_users,
            size=MAX_USERS,
            replace=False
        ).tolist()
    )
else:
    selected_users = sorted(eligible_users)

print(f"Eligible users: {len(eligible_users):,}")
print(f"Selected users: {len(selected_users):,}")

if not selected_users:
    raise RuntimeError(
        "No eligible users. Check the dataset and model ID mappings."
    )

totals = {
    f"{metric}@{k}": 0.0
    for k in K_VALUES
    for metric in ("Recall", "NDCG")
}

evaluated = 0
n_movies = index.ntotal

print("\nEvaluating retrieval...")

for i, user_id in enumerate(selected_users, start=1):
    query = user_embeddings[
        user_position[user_id]:user_position[user_id] + 1
    ].copy()

    if not np.isfinite(query).all():
        continue

    query = np.ascontiguousarray(query, dtype=np.float32)

    if np.linalg.norm(query) == 0:
        continue

    faiss.normalize_L2(query)

    # Search the entire index so history filtering does not cause
    # insufficient recommendations when enough unseen items exist.
    scores, positions = index.search(query, n_movies)

    seen = seen_by_user.get(user_id, set())
    relevant = relevant_by_user[user_id]

    recommended = []

    for position in positions[0]:
        if position < 0:
            continue

        movie_id = movie_ids[position]

        if movie_id in seen:
            continue

        recommended.append(movie_id)

        if len(recommended) >= max(K_VALUES):
            break

    if not recommended:
        continue

    for k in K_VALUES:
        totals[f"Recall@{k}"] += recall_at_k(
            recommended, relevant, k
        )
        totals[f"NDCG@{k}"] += ndcg_at_k(
            recommended, relevant, k
        )

    evaluated += 1

    if i % 100 == 0:
        print(f"Processed {i:,}/{len(selected_users):,} users")

if evaluated == 0:
    raise RuntimeError("No users were successfully evaluated.")

metrics = {
    key: round(value / evaluated, 6)
    for key, value in totals.items()
}

report = {
    "model": "Two-tower retrieval + FAISS",
    "dataset": "MovieLens 25M",
    "validation_split": "Chronological",
    "history_filter": "Full chronological training interactions",
    "relevant_rating_threshold": RELEVANT_RATING,
    "users_evaluated": evaluated,
    "indexed_movies": int(index.ntotal),
    "metrics": metrics,
    "limitations": [
        "The model was trained on a random 500,000-interaction sample.",
        "Evaluation is restricted to users and movies known to the model.",
        "Recall and NDCG are offline ranking metrics, not online CTR.",
        "The sampled training model may have limited coverage of the full dataset."
    ]
}

output_file = REPORTS / "retrieval_evaluation_v2.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\n" + "=" * 50)
print("CORRECTED TWO-TOWER EVALUATION")
print("=" * 50)
print(f"Users evaluated: {evaluated:,}")
print(f"Movies indexed: {index.ntotal:,}")

for key, value in metrics.items():
    print(f"{key}: {value:.6f}")

print(f"\nReport saved: {output_file}")
