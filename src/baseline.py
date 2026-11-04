
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import ndcg_score


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
REPORT_DIR = PROJECT_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "train.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"
MOVIE_FILE = DATA_DIR / "movies_clean.csv"


# --------------------------------------------------
# 2. Configuration
# --------------------------------------------------

TOP_K_VALUES = [10, 20]

# Ratings of 4.0 or higher are treated as relevant.
RELEVANCE_THRESHOLD = 4.0

# Evaluate a fixed sample of users to keep execution
# manageable on a CPU-only computer.
MAX_EVALUATION_USERS = 5000

RANDOM_SEED = 42


# --------------------------------------------------
# 3. Load data
# --------------------------------------------------

print("Loading training and validation data...")

train = pd.read_csv(
    TRAIN_FILE,
    usecols=["userId", "movieId", "rating"]
)

validation = pd.read_csv(
    VALIDATION_FILE,
    usecols=["userId", "movieId", "rating"]
)

movies = pd.read_csv(
    MOVIE_FILE,
    usecols=["movieId", "title", "genres"]
)

print(f"Training interactions: {len(train):,}")
print(f"Validation interactions: {len(validation):,}")


# --------------------------------------------------
# 4. Build the popularity baseline
# --------------------------------------------------

print("\nBuilding popularity rankings...")

movie_stats = (
    train.groupby("movieId")
    .agg(
        interaction_count=("rating", "size"),
        average_rating=("rating", "mean")
    )
    .reset_index()
)

# Baseline A: interaction count.
popularity_ranking = movie_stats.sort_values(
    ["interaction_count", "movieId"],
    ascending=[False, True]
).reset_index(drop=True)


# --------------------------------------------------
# 5. Build a weighted-rating baseline
# --------------------------------------------------

# Bayesian-style weighted rating:
#
# Score = (v / (v + m)) * R + (m / (v + m)) * C
#
# v = number of training ratings for the movie
# R = average training rating for the movie
# m = minimum rating count used for stronger weighting
# C = average rating across all training interactions

C = float(train["rating"].mean())

# Use the 90th percentile of movie rating counts as m.
m = float(movie_stats["interaction_count"].quantile(0.90))

movie_stats["weighted_score"] = (
    (
        movie_stats["interaction_count"]
        / (movie_stats["interaction_count"] + m)
    )
    * movie_stats["average_rating"]
    +
    (
        m
        / (movie_stats["interaction_count"] + m)
    )
    * C
)

weighted_ranking = movie_stats.sort_values(
    ["weighted_score", "interaction_count", "movieId"],
    ascending=[False, False, True]
).reset_index(drop=True)


# --------------------------------------------------
# 6. Select evaluation users
# --------------------------------------------------

# Users are selected from the validation set.
# Only users who have at least one relevant validation
# interaction are included.

relevant_validation = validation[
    validation["rating"] >= RELEVANCE_THRESHOLD
].copy()

users_with_relevant_items = (
    relevant_validation["userId"].unique()
)

rng = np.random.default_rng(RANDOM_SEED)

if len(users_with_relevant_items) > MAX_EVALUATION_USERS:
    evaluation_users = rng.choice(
        users_with_relevant_items,
        size=MAX_EVALUATION_USERS,
        replace=False
    )
else:
    evaluation_users = users_with_relevant_items

evaluation_users = set(evaluation_users.tolist())

relevant_validation = relevant_validation[
    relevant_validation["userId"].isin(evaluation_users)
]

# Build the set of relevant validation movies for each user.
relevant_items_by_user = (
    relevant_validation.groupby("userId")["movieId"]
    .apply(set)
    .to_dict()
)

# Use only movies known to the training dataset.
train_movie_ids = set(train["movieId"].unique())

relevant_items_by_user = {
    user_id: items.intersection(train_movie_ids)
    for user_id, items in relevant_items_by_user.items()
}

relevant_items_by_user = {
    user_id: items
    for user_id, items in relevant_items_by_user.items()
    if items
}

evaluation_users = list(relevant_items_by_user.keys())

print(f"Evaluation users: {len(evaluation_users):,}")
print(f"Relevance threshold: {RELEVANCE_THRESHOLD}")


# --------------------------------------------------
# 7. Build candidate lists
# --------------------------------------------------

# A user should not receive a movie they already rated
# in the training data.

seen_by_user = (
    train.groupby("userId")["movieId"]
    .apply(set)
    .to_dict()
)

def recommend_for_user(user_id, ranking, k):
    seen = seen_by_user.get(user_id, set())

    recommendations = []

    for movie_id in ranking["movieId"]:
        if movie_id not in seen:
            recommendations.append(int(movie_id))

        if len(recommendations) >= k:
            break

    return recommendations


# --------------------------------------------------
# 8. Evaluation metrics
# --------------------------------------------------

def evaluate_ranking(ranking, name):
    results = {}

    maximum_k = max(TOP_K_VALUES)

    recommendations_by_user = {}

    for user_id in evaluation_users:
        recommendations_by_user[user_id] = recommend_for_user(
            user_id,
            ranking,
            maximum_k
        )

    for k in TOP_K_VALUES:
        recalls = []
        ndcgs = []

        for user_id in evaluation_users:
            relevant = relevant_items_by_user[user_id]

            recommendations = recommendations_by_user[user_id][:k]

            hits = sum(
                movie_id in relevant
                for movie_id in recommendations
            )

            recall = hits / len(relevant)
            recalls.append(recall)

            # Binary relevance for the recommended list.
            relevance_vector = np.array(
                [
                    1.0 if movie_id in relevant else 0.0
                    for movie_id in recommendations
                ],
                dtype=float
            )

            if len(relevance_vector) == 0:
                ndcg = 0.0
            else:
                ideal = np.sort(relevance_vector)[::-1]

                discounts = 1.0 / np.log2(
                    np.arange(2, len(relevance_vector) + 2)
                )

                dcg = np.sum(relevance_vector * discounts)
                idcg = np.sum(ideal * discounts)

                ndcg = dcg / idcg if idcg > 0 else 0.0

            ndcgs.append(ndcg)

        results[f"Recall@{k}"] = float(np.mean(recalls))
        results[f"NDCG@{k}"] = float(np.mean(ndcgs))

    results["model"] = name

    return results


# --------------------------------------------------
# 9. Run both baselines
# --------------------------------------------------

print("\nEvaluating popularity baseline...")

popularity_results = evaluate_ranking(
    popularity_ranking,
    "Popularity"
)

print("Evaluating weighted-rating baseline...")

weighted_results = evaluate_ranking(
    weighted_ranking,
    "WeightedRating"
)

results_df = pd.DataFrame(
    [popularity_results, weighted_results]
)

results_df = results_df[
    [
        "model",
        "Recall@10",
        "NDCG@10",
        "Recall@20",
        "NDCG@20"
    ]
]

print("\n" + "=" * 65)
print("BASELINE EVALUATION RESULTS")
print("=" * 65)

print(results_df.to_string(index=False))

# Save metrics.
results_path = REPORT_DIR / "baseline_metrics.csv"

results_df.to_csv(
    results_path,
    index=False
)

# Save the ranked movie lists.
popularity_ranking.merge(
    movies,
    on="movieId",
    how="left"
).to_csv(
    REPORT_DIR / "popularity_ranking.csv",
    index=False
)

weighted_ranking.merge(
    movies,
    on="movieId",
    how="left"
).to_csv(
    REPORT_DIR / "weighted_rating_ranking.csv",
    index=False
)

print(f"\nMetrics saved to: {results_path}")
print("Baseline evaluation completed.")
