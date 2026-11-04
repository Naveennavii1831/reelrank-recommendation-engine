

from pathlib import Path
import json

import numpy as np
import pandas as pd
import faiss


# --------------------------------------------------
# 1. Paths and evaluation settings
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

TRAIN_FILE = DATA_DIR / "retrieval_train_sample.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"
EMBEDDINGS_FILE = MODELS_DIR / "retrieval_embeddings.npz"
INDEX_FILE = MODELS_DIR / "movie_faiss.index"
MOVIE_IDS_FILE = MODELS_DIR / "faiss_movie_ids.npy"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

K_VALUES = [10, 20]
MAX_USERS = 1000
RELEVANT_RATING = 4.0
RANDOM_SEED = 42


# --------------------------------------------------
# 2. Load model embeddings and FAISS index
# --------------------------------------------------

print("Loading model embeddings and FAISS index...")

with np.load(EMBEDDINGS_FILE, allow_pickle=False) as data:
    model_user_ids = data["user_ids"].astype(str)
    user_embeddings = data["user_embeddings"].astype(np.float32)

index = faiss.read_index(str(INDEX_FILE))
movie_ids = np.load(MOVIE_IDS_FILE, allow_pickle=False).astype(str)

print(f"Users with embeddings: {len(model_user_ids):,}")
print(f"Movies in FAISS index: {index.ntotal:,}")


# --------------------------------------------------
# 3. Load interaction data
# --------------------------------------------------

print("\nLoading sampled training interactions...")

train = pd.read_csv(
    TRAIN_FILE,
    usecols=["userId", "movieId", "rating"]
)

train["userId"] = train["userId"].astype(str)
train["movieId"] = train["movieId"].astype(str)

print(f"Sampled training interactions: {len(train):,}")

print("\nLoading validation interactions...")

validation = pd.read_csv(
    VALIDATION_FILE,
    usecols=["userId", "movieId", "rating"]
)

validation["userId"] = validation["userId"].astype(str)
validation["movieId"] = validation["movieId"].astype(str)

validation = validation[validation["rating"] >= RELEVANT_RATING].copy()

print(f"Validation positive interactions: {len(validation):,}")


# --------------------------------------------------
# 4. Create a consistent warm-start evaluation cohort
# --------------------------------------------------

# Only evaluate users with a learned user embedding.
# Only score movies that exist in the FAISS index.
known_users = set(model_user_ids)
known_movies = set(movie_ids)

validation = validation[
    validation["userId"].isin(known_users)
    & validation["movieId"].isin(known_movies)
].copy()

# The training sample is a random subset of the full training data.
# It does not necessarily contain every interaction used to construct
# the chronological train/validation split.
#
# We exclude sampled training items from recommendations. This gives
# a useful prototype evaluation, but the partial history can introduce
# leakage or incomplete filtering. Results must be interpreted with
# that limitation in mind.

sampled_history = train.groupby("userId")["movieId"].apply(set).to_dict()

validation_relevant = (
    validation.groupby("userId")["movieId"].apply(set).to_dict()
)

eligible_users = sorted(
    user_id
    for user_id, relevant_movies in validation_relevant.items()
    if relevant_movies
    and user_id in sampled_history
)

rng = np.random.default_rng(RANDOM_SEED)

if len(eligible_users) > MAX_USERS:
    selected_users = sorted(
        rng.choice(eligible_users, size=MAX_USERS, replace=False).tolist()
    )
else:
    selected_users = eligible_users

print(f"\nEligible warm-start users: {len(eligible_users):,}")
print(f"Users selected for evaluation: {len(selected_users):,}")

if not selected_users:
    raise RuntimeError(
        "No eligible users remain. Check the user/movie ID mappings "
        "and the training/validation split."
    )


# --------------------------------------------------
# 5. Prepare lookup tables
# --------------------------------------------------

user_position = {
    user_id: position
    for position, user_id in enumerate(model_user_ids)
}

movie_position = {
    movie_id: position
    for position, movie_id in enumerate(movie_ids)
}

train_grouped = sampled_history
relevant_grouped = validation_relevant


# --------------------------------------------------
# 6. Ranking metrics
# --------------------------------------------------

def recall_at_k(recommended, relevant, k):
    if not relevant:
        return 0.0

    hits = len(set(recommended[:k]) & relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended, relevant, k):
    if not relevant:
        return 0.0

    dcg = 0.0

    for rank, movie_id in enumerate(recommended[:k], start=1):
        if movie_id in relevant:
            dcg += 1.0 / np.log2(rank + 1)

    ideal_hits = min(len(relevant), k)

    ideal_dcg = sum(
        1.0 / np.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    return dcg / ideal_dcg if ideal_dcg else 0.0


# --------------------------------------------------
# 7. Generate recommendations for each user
# --------------------------------------------------

print("\nGenerating recommendations...")

metric_totals = {
    f"{metric}@{k}": 0.0
    for k in K_VALUES
    for metric in ["Recall", "NDCG"]
}

evaluated_users = 0
candidate_count = index.ntotal

for counter, user_id in enumerate(selected_users, start=1):
    position = user_position[user_id]

    query = user_embeddings[position:position + 1].copy()
    query = np.ascontiguousarray(query, dtype=np.float32)

    if not np.isfinite(query).all():
        continue

    norm = np.linalg.norm(query)

    if norm == 0:
        continue

    faiss.normalize_L2(query)

    # Search all indexed movies so filtering does not leave too few.
    scores, positions = index.search(query, candidate_count)

    already_rated = train_grouped.get(user_id, set())
    relevant = relevant_grouped.get(user_id, set())

    recommended = []

    for item_position in positions[0]:
        if item_position < 0:
            continue

        movie_id = movie_ids[item_position]

        if movie_id in already_rated:
            continue

        recommended.append(movie_id)

        if len(recommended) >= max(K_VALUES):
            break

    if not recommended:
        continue

    for k in K_VALUES:
        metric_totals[f"Recall@{k}"] += recall_at_k(
            recommended, relevant, k
        )

        metric_totals[f"NDCG@{k}"] += ndcg_at_k(
            recommended, relevant, k
        )

    evaluated_users += 1

    if counter % 100 == 0:
        print(f"Processed {counter:,}/{len(selected_users):,} users")


if evaluated_users == 0:
    raise RuntimeError("No users could be evaluated.")


# --------------------------------------------------
# 8. Calculate and save results
# --------------------------------------------------

results = {
    metric: round(total / evaluated_users, 6)
    for metric, total in metric_totals.items()
}

report = {
    "model": "Two-tower retrieval + FAISS",
    "dataset": "MovieLens 25M",
    "evaluation_split": "Chronological validation split",
    "positive_rating_threshold": RELEVANT_RATING,
    "users_requested": len(selected_users),
    "users_evaluated": evaluated_users,
    "indexed_movies": int(index.ntotal),
    "metrics": results,
    "limitations": [
        "The two-tower model was trained on a random 500,000-interaction "
        "sample of the chronological training split.",
        "Sampled training history is incomplete and may not contain every "
        "training interaction.",
        "Evaluation is restricted to users and movies known to the model.",
        "Recall and NDCG are offline ranking metrics, not online CTR.",
        "This evaluation should be refined before claiming a fair comparison "
        "with the popularity baseline."
    ]
}

report_path = REPORTS_DIR / "retrieval_evaluation.json"

with open(report_path, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=2)

print("\n" + "=" * 55)
print("TWO-TOWER RETRIEVAL EVALUATION")
print("=" * 55)
print(f"Users evaluated: {evaluated_users:,}")
print(f"Movies indexed: {index.ntotal:,}")

for metric, value in results.items():
    print(f"{metric}: {value:.6f}")

print(f"\nReport saved to: {report_path}")
print("=" * 55)
