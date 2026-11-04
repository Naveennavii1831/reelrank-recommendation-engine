
import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import tf_keras as keras


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

TRAIN_FILE = DATA / "train.csv"
VALID_FILE = DATA / "validation.csv"
USER_FILE = DATA / "user_features.csv"
MOVIE_FILE = DATA / "movie_features.csv"

MODEL_FILE = MODELS / "dcn_ranker.keras"
METADATA_FILE = MODELS / "dcn_ranker_metadata.json"
EMBEDDINGS_FILE = MODELS / "retrieval_embeddings.npz"
FAISS_FILE = MODELS / "movie_faiss.index"
FAISS_IDS_FILE = MODELS / "faiss_movie_ids.npy"

N_USERS = 500
N_CANDIDATES = 100
K_VALUES = (10, 20)
SEED = 42
BOOTSTRAP_SAMPLES = 2000


def recall_at_k(ranked_items, relevant_items, k):
    if not relevant_items:
        return 0.0

    hits = len(set(ranked_items[:k]) & relevant_items)
    return hits / len(relevant_items)


def ndcg_at_k(ranked_items, relevant_items, k):
    dcg = sum(
        1.0 / np.log2(rank + 1)
        for rank, movie_id in enumerate(ranked_items[:k], start=1)
        if movie_id in relevant_items
    )

    ideal_hits = min(len(relevant_items), k)

    idcg = sum(
        1.0 / np.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    return dcg / idcg if idcg else 0.0


def prepare_features(
    user_id,
    candidate_ids,
    user_feature_map,
    movie_feature_map,
    metadata,
):
    genre_to_index = metadata["genre_to_index"]
    genre_count = len(metadata["genre_vocabulary"])

    scaler_mean = np.asarray(
        metadata["numeric_scaler_mean"],
        dtype=np.float32,
    )

    scaler_scale = np.asarray(
        metadata["numeric_scaler_scale"],
        dtype=np.float32,
    )

    scaler_scale = np.where(
        scaler_scale == 0,
        1.0,
        scaler_scale,
    )

    user_row = user_feature_map.get(user_id, {})
    dense_rows = []

    for movie_id in candidate_ids:
        movie_row = movie_feature_map.get(movie_id, {})

        raw_numeric = [
            user_row.get("user_interaction_count", np.nan),
            user_row.get("user_average_rating", np.nan),
            movie_row.get("movie_interaction_count", np.nan),
            movie_row.get("movie_average_rating", np.nan),
            movie_row.get("movie_unique_users", np.nan),
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

        # Match the transformations used during model training.
        for position in (0, 2, 4):
            cleaned[position] = np.log1p(
                max(cleaned[position], 0.0)
            )

        numeric = (
            np.asarray(cleaned, dtype=np.float32) - scaler_mean
        ) / scaler_scale

        genres = np.zeros(
            genre_count,
            dtype=np.float32,
        )

        for genre in str(
            movie_row.get("genres", "")
        ).split("|"):
            index = genre_to_index.get(genre)

            if index is not None:
                genres[index] = 1.0

        dense_rows.append(
            np.concatenate([numeric, genres])
        )

    return {
        "user_id": np.full(
            (len(candidate_ids), 1),
            user_id,
            dtype=np.int32,
        ),
        "movie_id": np.asarray(
            candidate_ids,
            dtype=np.int32,
        ).reshape(-1, 1),
        "dense_features": np.asarray(
            dense_rows,
            dtype=np.float32,
        ),
    }


def bootstrap_mean_difference(
    differences,
    rng,
    samples=BOOTSTRAP_SAMPLES,
):
    """Percentile bootstrap CI for paired per-user differences."""

    differences = np.asarray(
        differences,
        dtype=np.float64,
    )

    if len(differences) == 0:
        return {
            "mean_difference": None,
            "ci_95_lower": None,
            "ci_95_upper": None,
        }

    bootstrap_means = np.empty(
        samples,
        dtype=np.float64,
    )

    for i in range(samples):
        sampled = rng.choice(
            differences,
            size=len(differences),
            replace=True,
        )
        bootstrap_means[i] = sampled.mean()

    return {
        "mean_difference": float(differences.mean()),
        "ci_95_lower": float(
            np.percentile(bootstrap_means, 2.5)
        ),
        "ci_95_upper": float(
            np.percentile(bootstrap_means, 97.5)
        ),
    }


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    print("Loading DCN model and metadata...")

    with open(
        METADATA_FILE,
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    model = keras.models.load_model(MODEL_FILE)

    user_vocabulary = set(
        map(int, metadata["user_vocabulary"])
    )
    movie_vocabulary = set(
        map(int, metadata["movie_vocabulary"])
    )

    print("Loading FAISS index...")

    index = faiss.read_index(str(FAISS_FILE))
    faiss_movie_ids = np.load(
        FAISS_IDS_FILE
    ).astype(np.int64)

    with np.load(
        EMBEDDINGS_FILE,
        allow_pickle=False,
    ) as archive:
        embedding_users = archive["user_ids"].astype(np.int64)
        user_vectors = archive[
            "user_embeddings"
        ].astype(np.float32)

    user_embedding_map = {
        int(user_id): user_vectors[i]
        for i, user_id in enumerate(embedding_users)
    }

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

    user_feature_map = users.set_index(
        "userId"
    ).to_dict("index")

    movie_feature_map = movies.set_index(
        "movieId"
    ).to_dict("index")

    print("Loading training history...")

    history = pd.read_csv(
        TRAIN_FILE,
        usecols=["userId", "movieId"],
        dtype={
            "userId": "int32",
            "movieId": "int32",
        },
    )

    seen_by_user = (
        history.groupby("userId")["movieId"]
        .apply(lambda values: set(map(int, values)))
        .to_dict()
    )

    del history

    print("Loading chronological validation data...")

    validation = pd.read_csv(
        VALID_FILE,
        usecols=["userId", "movieId", "rating"],
    )

    validation = validation[
        validation["rating"] >= 4.0
    ].copy()

    validation["userId"] = validation["userId"].astype(int)
    validation["movieId"] = validation["movieId"].astype(int)

    validation = validation[
        validation["userId"].isin(user_vocabulary)
        & validation["movieId"].isin(movie_vocabulary)
    ]

    relevant_by_user = (
        validation.groupby("userId")["movieId"]
        .apply(lambda values: set(map(int, values)))
        .to_dict()
    )

    eligible_users = [
        user_id
        for user_id, relevant in relevant_by_user.items()
        if relevant and user_id in seen_by_user
    ]

    rng = np.random.default_rng(SEED)

    if len(eligible_users) > N_USERS:
        selected_users = sorted(
            rng.choice(
                eligible_users,
                size=N_USERS,
                replace=False,
            ).tolist()
        )
    else:
        selected_users = sorted(eligible_users)

    if not selected_users:
        raise RuntimeError(
            "No eligible users found for evaluation."
        )

    print(f"Selected users: {len(selected_users):,}")
    print(f"Requested candidates per user: {N_CANDIDATES}")

    # Per-user metrics preserve the pairing between rankers.
    metric_names = [
        f"{name}@{k}"
        for k in K_VALUES
        for name in ("Recall", "NDCG")
    ]

    per_user = {
        "retrieval": {name: [] for name in metric_names},
        "dcn": {name: [] for name in metric_names},
    }

    users_with_candidates = 0
    users_with_relevant_candidate = 0

    print("\nRunning offline A/B evaluation...")

    for counter, user_id in enumerate(
        selected_users,
        start=1,
    ):
        user_vector = user_embedding_map.get(user_id)

        if user_vector is None:
            continue

        query = np.asarray(
            user_vector,
            dtype=np.float32,
        ).reshape(1, -1).copy()

        faiss.normalize_L2(query)

        search_count = min(
            index.ntotal,
            N_CANDIDATES + 200,
        )

        _, positions = index.search(
            query,
            search_count,
        )

        seen = seen_by_user.get(user_id, set())
        candidates = []

        for position in positions[0]:
            if position < 0:
                continue

            movie_id = int(faiss_movie_ids[position])

            if movie_id in seen:
                continue

            if movie_id not in movie_vocabulary:
                continue

            if movie_id not in movie_feature_map:
                continue

            candidates.append(movie_id)

            if len(candidates) >= N_CANDIDATES:
                break

        if not candidates:
            continue

        users_with_candidates += 1

        relevant = relevant_by_user.get(user_id, set())

        if set(candidates) & relevant:
            users_with_relevant_candidate += 1

        # Ranker A: original FAISS retrieval order.
        retrieval_order = list(candidates)

        # Ranker B: DCN reranking of the exact same candidates.
        inputs = prepare_features(
            user_id,
            candidates,
            user_feature_map,
            movie_feature_map,
            metadata,
        )

        scores = model.predict(
            inputs,
            verbose=0,
            batch_size=512,
        ).reshape(-1)

        ranked_positions = np.argsort(-scores)
        dcn_order = [
            candidates[int(position)]
            for position in ranked_positions
        ]

        for k in K_VALUES:
            per_user["retrieval"][f"Recall@{k}"].append(
                recall_at_k(
                    retrieval_order,
                    relevant,
                    k,
                )
            )

            per_user["retrieval"][f"NDCG@{k}"].append(
                ndcg_at_k(
                    retrieval_order,
                    relevant,
                    k,
                )
            )

            per_user["dcn"][f"Recall@{k}"].append(
                recall_at_k(
                    dcn_order,
                    relevant,
                    k,
                )
            )

            per_user["dcn"][f"NDCG@{k}"].append(
                ndcg_at_k(
                    dcn_order,
                    relevant,
                    k,
                )
            )

        if counter % 50 == 0:
            print(
                f"Processed {counter:,}/"
                f"{len(selected_users):,} users"
            )

    summary = {}

    for ranker_name, metrics in per_user.items():
        summary[ranker_name] = {
            metric: (
                float(np.mean(values))
                if values
                else 0.0
            )
            for metric, values in metrics.items()
        }

    comparisons = {}

    for metric in metric_names:
        baseline = summary["retrieval"][metric]
        treatment = summary["dcn"][metric]
        absolute_difference = treatment - baseline

        relative_improvement = (
            (absolute_difference / baseline) * 100.0
            if baseline != 0
            else None
        )

        comparisons[metric] = {
            "retrieval_baseline": baseline,
            "dcn_treatment": treatment,
            "absolute_difference": absolute_difference,
            "relative_improvement_percent": relative_improvement,
        }

    ndcg10_differences = (
        np.asarray(per_user["dcn"]["NDCG@10"])
        - np.asarray(per_user["retrieval"]["NDCG@10"])
    )

    confidence_interval = bootstrap_mean_difference(
        ndcg10_differences,
        rng,
    )

    report = {
        "experiment_type": "offline paired ranking comparison",
        "dataset_split": "chronological validation",
        "relevance_threshold": 4.0,
        "selected_users": len(selected_users),
        "users_with_candidates": users_with_candidates,
        "users_with_relevant_candidate": users_with_relevant_candidate,
        "requested_candidates_per_user": N_CANDIDATES,
        "ranker_a": "FAISS retrieval order",
        "ranker_b": "DCN reranking on the same retrieved candidates",
        "metrics": summary,
        "comparison": comparisons,
        "ndcg_at_10_paired_bootstrap_95_percent_ci": confidence_interval,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "seed": SEED,
        "limitations": [
            "This is an offline experiment, not a live online A/B test.",
            "Both rankers use the same retrieved candidates.",
            "Recall and NDCG are measured against held-out ratings >= 4.0.",
            "Users and movies are restricted to those represented by the trained model.",
            "CTR cannot be calculated without actual impression and click logs.",
            "The DCN is evaluated on historical data and may not generalize to live users.",
            "Offline metric improvements do not establish statistical or business significance by themselves.",
        ],
    }

    report_path = REPORTS / "ab_test_results.json"

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, indent=2)

    print("\n" + "=" * 64)
    print("OFFLINE A/B RANKER COMPARISON")
    print("=" * 64)

    print(f"Users selected: {len(selected_users):,}")
    print(f"Users with candidates: {users_with_candidates:,}")
    print(
        "Users with a relevant candidate: "
        f"{users_with_relevant_candidate:,}"
    )

    for metric in metric_names:
        result = comparisons[metric]

        print(f"\n{metric}")
        print(
            f"  Retrieval: {result['retrieval_baseline']:.6f}"
        )
        print(
            f"  DCN:       {result['dcn_treatment']:.6f}"
        )
        print(
            f"  Difference: {result['absolute_difference']:+.6f}"
        )

        improvement = result["relative_improvement_percent"]

        if improvement is not None:
            print(
                f"  Relative improvement: {improvement:+.2f}%"
            )
        else:
            print(
                "  Relative improvement: undefined "
                "(retrieval baseline is zero)"
            )

    print("\nPaired bootstrap CI for NDCG@10 difference:")
    print(
        json.dumps(
            confidence_interval,
            indent=2,
        )
    )

    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
