
import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras as keras

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

TRAIN_FILE = DATA / "train.csv"
VALID_FILE = DATA / "validation.csv"
USER_FILE = DATA / "user_features.csv"
MOVIE_FILE = DATA / "movie_features.csv"

MODEL_FILE = MODELS / "dcn_ranker_compat.h5"
METADATA_FILE = MODELS / "dcn_ranker_metadata.json"
EMBEDDINGS_FILE = MODELS / "retrieval_embeddings.npz"
FAISS_FILE = MODELS / "movie_faiss.index"
FAISS_IDS_FILE = MODELS / "faiss_movie_ids.npy"

N_USERS = 500
N_CANDIDATES = 100
K_VALUES = (10, 20)
SEED = 42

# ---------------------------------------------------------
# Load saved model and metadata
# ---------------------------------------------------------
print("Loading DCN ranker and metadata...")

with open(METADATA_FILE, encoding="utf-8") as f:
    metadata = json.load(f)

model = keras.models.load_model(MODEL_FILE, compile=False)

user_vocabulary = set(metadata["user_vocabulary"])
movie_vocabulary = set(metadata["movie_vocabulary"])
genre_to_index = metadata["genre_to_index"]
genre_count = len(metadata["genre_vocabulary"])
numeric_features = metadata["numeric_features"]

scaler_mean = np.asarray(
    metadata["numeric_scaler_mean"], dtype=np.float32
)
scaler_scale = np.asarray(
    metadata["numeric_scaler_scale"], dtype=np.float32
)

# ---------------------------------------------------------
# Load retrieval assets
# ---------------------------------------------------------
print("Loading FAISS index and retrieval embeddings...")

index = faiss.read_index(str(FAISS_FILE))
faiss_movie_ids = np.load(FAISS_IDS_FILE)

with np.load(EMBEDDINGS_FILE) as archive:
    embedding_users = archive["user_ids"]
    embedding_movies = archive["movie_ids"]

embedding_user_set = set(map(int, embedding_users))
embedding_movie_set = set(map(int, embedding_movies))

print(f"FAISS candidates available: {index.ntotal:,}")
print(f"DCN user vocabulary: {len(user_vocabulary):,}")
print(f"DCN movie vocabulary: {len(movie_vocabulary):,}")

# ---------------------------------------------------------
# Load features
# ---------------------------------------------------------
print("Loading feature tables...")

users = pd.read_csv(USER_FILE).rename(
    columns={
        "interaction_count": "user_interaction_count",
        "average_rating": "user_average_rating",
    }
)

movies = pd.read_csv(
    MOVIE_FILE,
    usecols=[
        "movieId",
        "interaction_count",
        "average_rating",
        "unique_users",
        "genres",
        "title",
    ],
).rename(
    columns={
        "interaction_count": "movie_interaction_count",
        "average_rating": "movie_average_rating",
        "unique_users": "movie_unique_users",
    }
)

users["userId"] = users["userId"].astype(int)
movies["movieId"] = movies["movieId"].astype(int)

# ---------------------------------------------------------
# Load training history and validation positives
# ---------------------------------------------------------
print("Loading full training history...")

train_history = pd.read_csv(
    TRAIN_FILE, usecols=["userId", "movieId"]
)

seen_by_user = train_history.groupby("userId")["movieId"].apply(
    lambda values: set(map(int, values))
).to_dict()

del train_history

print("Loading chronological validation interactions...")

validation = pd.read_csv(
    VALID_FILE,
    usecols=["userId", "movieId", "rating"],
)

validation["userId"] = validation["userId"].astype(int)
validation["movieId"] = validation["movieId"].astype(int)

# Relevant items are held-out ratings >= 4.0.
validation = validation[validation["rating"] >= 4.0]

# Require users and relevant movies to be representable by the DCN.
eligible = validation[
    validation["userId"].isin(user_vocabulary)
    & validation["movieId"].isin(movie_vocabulary)
]

relevant_by_user = eligible.groupby("userId")["movieId"].apply(
    lambda values: set(map(int, values))
).to_dict()

eligible_users = [
    user_id
    for user_id, items in relevant_by_user.items()
    if items and user_id in seen_by_user
]

rng = np.random.default_rng(SEED)
if len(eligible_users) > N_USERS:
    selected_users = sorted(
        rng.choice(eligible_users, size=N_USERS, replace=False).tolist()
    )
else:
    selected_users = sorted(eligible_users)

print(f"Validation users with at least one relevant item: {len(eligible_users):,}")
print(f"Users selected for evaluation: {len(selected_users):,}")

if not selected_users:
    raise RuntimeError(
        "No eligible validation users. Check saved vocabularies and data split."
    )

# ---------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------
def retrieve_candidates(user_id, n_candidates=N_CANDIDATES):
    """Retrieve FAISS candidates using the saved user embedding."""
    if user_id not in embedding_user_set:
        return []

    # Locate user vector in saved embeddings.
    user_position = int(np.searchsorted(embedding_users, user_id))

    # The embedding arrays may not be sorted, so use an explicit lookup.
    # This dictionary is created once below for reliable indexing.
    user_vector = user_embedding_map.get(user_id)
    if user_vector is None:
        return []

    query = np.asarray(user_vector, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(query)

    search_count = min(index.ntotal, n_candidates + 200)
    _, indices = index.search(query, search_count)

    seen = seen_by_user.get(user_id, set())
    result = []

    for idx in indices[0]:
        if idx < 0:
            continue

        movie_id = int(faiss_movie_ids[idx])

        if movie_id in seen:
            continue
        if movie_id not in movie_vocabulary:
            continue

        result.append(movie_id)

        if len(result) >= n_candidates:
            break

    return result


with np.load(EMBEDDINGS_FILE) as archive:
    user_vectors = archive["user_embeddings"]

user_embedding_map = {
    int(user_id): user_vectors[i]
    for i, user_id in enumerate(embedding_users)
}

# ---------------------------------------------------------
# Prepare DCN feature rows
# ---------------------------------------------------------
user_feature_map = users.set_index("userId").to_dict("index")
movie_feature_map = movies.set_index("movieId").to_dict("index")

def make_feature_batch(user_id, candidate_ids):
    rows = []
    dense_rows = []

    for movie_id in candidate_ids:
        user_feature = user_feature_map.get(user_id, {})
        movie_feature = movie_feature_map.get(movie_id, {})

        raw_numeric = [
            user_feature.get("user_interaction_count", np.nan),
            user_feature.get("user_average_rating", np.nan),
            movie_feature.get("movie_interaction_count", np.nan),
            movie_feature.get("movie_average_rating", np.nan),
            movie_feature.get("movie_unique_users", np.nan),
        ]

        cleaned = []
        for value, mean in zip(raw_numeric, scaler_mean):
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = np.nan
            if not np.isfinite(value):
                value = float(mean)
            cleaned.append(value)

        # Apply the same transformations as training.
        for position in (0, 2, 4):
            cleaned[position] = np.log1p(max(cleaned[position], 0.0))

        numeric = (
            np.asarray(cleaned, dtype=np.float32) - scaler_mean
        ) / np.where(scaler_scale == 0, 1.0, scaler_scale)

        genres = np.zeros(genre_count, dtype=np.float32)
        for genre in str(movie_feature.get("genres", "")).split("|"):
            position = genre_to_index.get(genre)
            if position is not None:
                genres[position] = 1.0

        dense_rows.append(np.concatenate([numeric, genres]))

    inputs = {
        "user_id": np.full(
            (len(candidate_ids), 1), user_id, dtype=np.int32
        ),
        "movie_id": np.asarray(candidate_ids, dtype=np.int32).reshape(-1, 1),
        "dense_features": np.asarray(dense_rows, dtype=np.float32),
    }

    return inputs


# ---------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------
def recall_at_k(ranked_items, relevant_items, k):
    if not relevant_items:
        return 0.0
    return len(set(ranked_items[:k]) & relevant_items) / len(relevant_items)


def ndcg_at_k(ranked_items, relevant_items, k):
    dcg = 0.0
    for rank, movie_id in enumerate(ranked_items[:k], start=1):
        if movie_id in relevant_items:
            dcg += 1.0 / np.log2(rank + 1)

    ideal_hits = min(len(relevant_items), k)
    idcg = sum(
        1.0 / np.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    return dcg / idcg if idcg else 0.0


# ---------------------------------------------------------
# Evaluate retrieval and DCN reranking on the same candidates
# ---------------------------------------------------------
print("\nEvaluating retrieval and DCN reranking...")

metrics = {
    "retrieval": {f"{name}@{k}": [] for name in ("Recall", "NDCG") for k in K_VALUES},
    "dcn_reranked": {f"{name}@{k}": [] for name in ("Recall", "NDCG") for k in K_VALUES},
}

users_with_candidates = 0
users_with_validation_hit_in_candidates = 0

for counter, user_id in enumerate(selected_users, start=1):
    candidates = retrieve_candidates(user_id)

    if not candidates:
        continue

    users_with_candidates += 1
    relevant = relevant_by_user.get(user_id, set())

    if set(candidates) & relevant:
        users_with_validation_hit_in_candidates += 1

    # Retrieval order is the original FAISS order.
    retrieval_order = candidates

    feature_batch = make_feature_batch(user_id, candidates)
    scores = model.predict(feature_batch, verbose=0, batch_size=512).reshape(-1)

    ranked_order = [
        movie_id
        for _, movie_id in sorted(
            zip(scores, candidates), reverse=True
        )
    ]

    for k in K_VALUES:
        metrics["retrieval"][f"Recall@{k}"].append(
            recall_at_k(retrieval_order, relevant, k)
        )
        metrics["retrieval"][f"NDCG@{k}"].append(
            ndcg_at_k(retrieval_order, relevant, k)
        )

        metrics["dcn_reranked"][f"Recall@{k}"].append(
            recall_at_k(ranked_order, relevant, k)
        )
        metrics["dcn_reranked"][f"NDCG@{k}"].append(
            ndcg_at_k(ranked_order, relevant, k)
        )

    if counter % 50 == 0:
        print(f"Processed {counter:,}/{len(selected_users):,} users")

# ---------------------------------------------------------
# Summarize and save report
# ---------------------------------------------------------
summary = {}

for method, metric_dict in metrics.items():
    summary[method] = {}
    for metric_name, values in metric_dict.items():
        summary[method][metric_name] = (
            float(np.mean(values)) if values else 0.0
        )

report = {
    "evaluation": "chronological validation, candidate-constrained",
    "relevant_rating_threshold": 4.0,
    "users_selected": len(selected_users),
    "users_with_candidates": users_with_candidates,
    "users_with_validation_hit_in_candidates": users_with_validation_hit_in_candidates,
    "candidate_count_requested": N_CANDIDATES,
    "candidate_generation": "two-tower user embeddings + FAISS",
    "results": summary,
    "limitations": [
        "Evaluation is restricted to users and movies known to the trained DCN.",
        "Recall is measured over all relevant validation movies for eligible users, while candidates are restricted to the retrieved set.",
        "The model was trained using a random sample from the chronological training partition.",
        "Feature tables are derived from the training partition.",
        "The evaluator compares retrieval order with DCN reranking on the same candidate set.",
        "This is an educational prototype, not a production or clinical system."
    ],
}

report_path = REPORTS / "ranking_evaluation.json"
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\n" + "=" * 60)
print("CHRONOLOGICAL RANKING EVALUATION")
print("=" * 60)
print(f"Users selected: {len(selected_users):,}")
print(f"Users with candidates: {users_with_candidates:,}")
print(f"Users whose candidates contain a relevant validation movie: {users_with_validation_hit_in_candidates:,}")

for method, metric_dict in summary.items():
    print(f"\n{method.upper()}")
    for metric_name, value in metric_dict.items():
        print(f"{metric_name}: {value:.6f}")

print(f"\nReport saved: {report_path}")
