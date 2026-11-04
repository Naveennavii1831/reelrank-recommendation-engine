import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import faiss
import numpy as np
import pandas as pd
import redis
import requests

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"

ONNX_MODEL_FILE = ROOT / "triton" / "models" / "dcn_ranker" / "1" / "model.onnx"
TRITON_URL = os.getenv(
    "TRITON_URL",
    "http://127.0.0.1:8003",
)

TRITON_MODEL_NAME = "dcn_ranker"
TRITON_INFER_URL = (
    f"{TRITON_URL}/v2/models/{TRITON_MODEL_NAME}/infer"
)
TRITON_READY_URL = (
    f"{TRITON_URL}/v2/models/{TRITON_MODEL_NAME}/ready"
)
METADATA_FILE = MODELS / "dcn_ranker_metadata.json"
EMBEDDINGS_FILE = MODELS / "retrieval_embeddings.npz"
FAISS_FILE = MODELS / "movie_faiss.index"
FAISS_IDS_FILE = MODELS / "faiss_movie_ids.npy"

USER_FILE = DATA / "user_features.csv"
MOVIE_FILE = DATA / "movie_features.csv"
TRAIN_FILE = DATA / "train.csv"


# Redis cache configuration.
# Recommendations are cached for 5 minutes by default.
REDIS_URL = os.getenv(
    "REDIS_URL",
    "redis://127.0.0.1:6379/0",
)

CACHE_TTL_SECONDS = int(
    os.getenv("RECOMMENDATION_CACHE_TTL", "300")
)

redis_client = redis.Redis.from_url(
    REDIS_URL,
    decode_responses=True,
    socket_connect_timeout=1,
    socket_timeout=1,
)


# ============================================================
# 2. GLOBAL MODEL AND DATA VARIABLES
# ============================================================

ranker = None
triton_model_ready = False
metadata: dict[str, Any] = {}

faiss_index = None
faiss_movie_ids = np.array([], dtype=np.int64)

user_embedding_map: dict[int, np.ndarray] = {}
user_features: dict[int, dict] = {}
movie_features: dict[int, dict] = {}

seen_by_user: dict[int, set[int]] = {}

movie_vocabulary: set[int] = set()
user_vocabulary: set[int] = set()

genre_to_index: dict[str, int] = {}
genre_count = 0

scaler_mean = np.array([], dtype=np.float32)
scaler_scale = np.array([], dtype=np.float32)

movie_lookup: dict[int, dict] = {}
popular_movie_ids: list[int] = []


# ============================================================
# 3. LOAD MODELS AND DATA
# ============================================================

def load_assets():
    global ranker, triton_model_ready, metadata, faiss_index, faiss_movie_ids
    global user_embedding_map, user_features, movie_features
    global seen_by_user, movie_vocabulary, user_vocabulary
    global genre_to_index, genre_count, scaler_mean, scaler_scale
    global movie_lookup, popular_movie_ids
    global user_id_to_index, movie_id_to_index

    required_files = [
        ONNX_MODEL_FILE,
        METADATA_FILE,
        EMBEDDINGS_FILE,
        FAISS_FILE,
        FAISS_IDS_FILE,
        USER_FILE,
        MOVIE_FILE,
        TRAIN_FILE,
    ]

    missing = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Required project files are missing: "
            + ", ".join(missing)
        )

        print("Connecting to Triton Inference Server...")

    with open(METADATA_FILE, encoding="utf-8") as file:
        metadata = json.load(file)

    try:
        triton_response = requests.get(
            TRITON_READY_URL,
            timeout=5,
        )
        triton_response.raise_for_status()

        triton_model_ready = True

    except requests.RequestException as error:
        raise RuntimeError(
            f"Triton model is not ready at {TRITON_READY_URL}: {error}"
        ) from error

    print("Triton model loaded successfully.")
    print("Triton endpoint:", TRITON_INFER_URL)

    user_id_to_index = {
        int(user_id): index + 1
        for index, user_id in enumerate(metadata["user_vocabulary"])
    }

    movie_id_to_index = {
        int(movie_id): index + 1
        for index, movie_id in enumerate(metadata["movie_vocabulary"])
    }

    user_vocabulary = set(user_id_to_index.keys())
    movie_vocabulary = set(movie_id_to_index.keys())

    genre_to_index = metadata["genre_to_index"]
    genre_count = len(metadata["genre_vocabulary"])

    scaler_mean = np.asarray(
        metadata["numeric_scaler_mean"],
        dtype=np.float32,
    )

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

    print("Loading FAISS index and retrieval embeddings...")

    faiss_index = faiss.read_index(str(FAISS_FILE))

    faiss_movie_ids = np.load(
        FAISS_IDS_FILE
    ).astype(np.int64)

    with np.load(
        EMBEDDINGS_FILE,
        allow_pickle=False,
    ) as archive:
        user_ids = archive["user_ids"].astype(np.int64)
        user_vectors = archive[
            "user_embeddings"
        ].astype(np.float32)

    user_embedding_map = {
        int(user_id): user_vectors[i]
        for i, user_id in enumerate(user_ids)
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

    user_features = users.set_index(
        "userId"
    ).to_dict("index")

    movie_features = movies.set_index(
        "movieId"
    ).to_dict("index")

    movie_lookup = {
        int(row["movieId"]): {
            "movie_id": int(row["movieId"]),
            "title": str(row["title"]),
            "genres": str(row["genres"]),
        }
        for row in movies.to_dict("records")
    }

    print("Loading interaction history for seen-item filtering...")

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

    # Popularity fallback based on training interaction counts.
    popularity = movies.sort_values(
        [
            "movie_interaction_count",
            "movie_average_rating",
        ],
        ascending=[False, False],
    )

    popular_movie_ids = [
        int(movie_id)
        for movie_id in popularity["movieId"].tolist()
    ]

    if faiss_index.ntotal != len(faiss_movie_ids):
        raise RuntimeError(
            "FAISS index size does not match "
            "the movie-ID mapping."
        )

    print(f"DCN user vocabulary: {len(user_vocabulary):,}")
    print(f"DCN movie vocabulary: {len(movie_vocabulary):,}")
    print(f"User embeddings: {len(user_embedding_map):,}")
    print(f"Movies in FAISS index: {faiss_index.ntotal:,}")
    print(f"Users with training history: {len(seen_by_user):,}")


# ============================================================
# 4. RETRIEVE UNSEEN MOVIE CANDIDATES
# ============================================================

def retrieve_candidates(
    user_id: int,
    count: int,
) -> list[tuple[int, float]]:
    """Retrieve unseen movie candidates using FAISS."""

    vector = user_embedding_map.get(user_id)

    if vector is None:
        return []

    query = np.asarray(
        vector,
        dtype=np.float32,
    ).reshape(1, -1).copy()

    faiss.normalize_L2(query)

    search_count = min(
        faiss_index.ntotal,
        max(count * 5, 200),
    )

    similarities, positions = faiss_index.search(
        query,
        search_count,
    )

    seen = seen_by_user.get(user_id, set())
    candidates = []

    for similarity, position in zip(
        similarities[0],
        positions[0],
    ):
        if position < 0:
            continue

        movie_id = int(faiss_movie_ids[position])

        if movie_id in seen:
            continue

        if movie_id not in movie_vocabulary:
            continue

        if movie_id not in movie_features:
            continue

        candidates.append(
            (movie_id, float(similarity))
        )

        if len(candidates) >= count:
            break

    return candidates


# ============================================================
# 5. PREPARE DCN RANKING FEATURES
# ============================================================

def prepare_ranker_inputs(
    user_id: int,
    candidate_ids: list[int],
) -> dict[str, np.ndarray]:
    """Prepare numeric, genre, user-ID, and movie-ID inputs."""

    dense_rows = []
    user_row = user_features.get(user_id, {})

    for movie_id in candidate_ids:
        movie_row = movie_features.get(movie_id, {})

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

        # Match the transformations used during training.
        for position in (0, 2, 4):
            cleaned[position] = np.log1p(
                max(cleaned[position], 0.0)
            )

        numeric = (
            np.asarray(cleaned, dtype=np.float32)
            - scaler_mean
        ) / scaler_scale

        genres = np.zeros(
            genre_count,
            dtype=np.float32,
        )

        genre_string = str(
            movie_row.get("genres", "")
        )

        for genre in genre_string.split("|"):
            index = genre_to_index.get(genre)

            if index is not None:
                genres[index] = 1.0

        dense = np.concatenate(
            [numeric, genres]
        ).astype(np.float32)

        dense_rows.append(dense)

    dense_features = np.asarray(
        dense_rows,
        dtype=np.float32,
    )
    movie_ids = np.asarray(
        [
            movie_id_to_index[int(movie_id)]
            for movie_id in candidate_ids
        ],
        dtype=np.int32,
    ).reshape(-1, 1)

    user_index = user_id_to_index[int(user_id)]

    user_ids = np.full(
        (len(candidate_ids), 1),
        user_index,
        dtype=np.int32,
    )

    return {
        "dense_features": dense_features,
        "movie_id": movie_ids,
        "user_id": user_ids,
    }

# ============================================================
# 6. POPULARITY FALLBACK
# ============================================================

def popularity_fallback(
    user_id: int,
    count: int,
) -> list[dict[str, Any]]:
    """Return popular unseen movies if personalized retrieval fails."""

    seen = seen_by_user.get(user_id, set())
    results = []

    for movie_id in popular_movie_ids:
        if movie_id in seen:
            continue

        movie = movie_lookup.get(movie_id)

        if movie is None:
            continue

        results.append(
            {
                **movie,
                "score": None,
                "retrieval_similarity": None,
                "recommendation_type": "popularity_fallback",
            }
        )

        if len(results) >= count:
            break

    return results


# ============================================================
# 7. REDIS CACHE HELPERS
# ============================================================

def get_cached_recommendations(
    cache_key: str,
) -> dict[str, Any] | None:
    """Return cached JSON data, or None if unavailable."""

    try:
        cached_value = redis_client.get(cache_key)

        if cached_value is None:
            return None

        return json.loads(cached_value)

    except (
        redis.RedisError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ):
        return None


def cache_recommendations(
    cache_key: str,
    result: dict[str, Any],
) -> None:
    """Store recommendation results with a TTL."""

    try:
        redis_client.setex(
            cache_key,
            CACHE_TTL_SECONDS,
            json.dumps(
                result,
                ensure_ascii=False,
                allow_nan=False,
            ),
        )

    except (
        redis.RedisError,
        TypeError,
        ValueError,
    ):
        pass


# ============================================================
# 8. FASTAPI LIFESPAN AND APPLICATION
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_assets()

    try:
        redis_client.ping()
        print("Redis cache: connected")
    except redis.RedisError:
        print(
            "Redis cache: unavailable; "
            "recommendations will still work without caching"
        )

    yield

    try:
        redis_client.close()
    except redis.RedisError:
        pass


app = FastAPI(
    title="Personalized Movie Recommendation API",
    description=(
        "Educational two-stage recommender using two-tower retrieval, "
        "FAISS candidate search, DCN reranking, and Redis caching."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 9. ROOT AND HEALTH ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {
        "project": "Large-Scale Personalized Recommendation Engine",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    try:
        redis_available = bool(redis_client.ping())
    except redis.RedisError:
        redis_available = False

    return {
        "status": (
            "healthy"
            if triton_model_ready and faiss_index is not None
            else "not_ready"
        ),
        "ranking_model_loaded": triton_model_ready,
        "ranking_model_runtime": "NVIDIA Triton Inference Server",
        "triton_url": TRITON_URL,

        "faiss_index_loaded": faiss_index is not None,
        "faiss_movie_count": (
            int(faiss_index.ntotal)
            if faiss_index is not None
            else 0
        ),
        "user_embedding_count": len(user_embedding_map),
        "redis_connected": redis_available,
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
        "note": "Educational prototype; not a production service.",
    }
# ============================================================
# 10. RECOMMENDATION ENDPOINT
# ============================================================

@app.get("/recommendations/{user_id}")
def get_recommendations(
    user_id: int,
    count: int = Query(default=10, ge=1, le=50),
    candidate_count: int = Query(default=100, ge=10, le=500),
    response: Response = None,
):
    """
    Generate personalized recommendations.

    Pipeline:
    1. Check Redis cache.
    2. Retrieve candidates using FAISS.
    3. Rerank candidates using the DCN ONNX model.
    4. Return popularity fallback if personalized retrieval fails.
    """

    if not triton_model_ready or faiss_index is None:
        raise HTTPException(
            status_code=503,
            detail="Recommendation models are not ready.",
        )

    cache_key = (
        f"recommendations:v1:"
        f"{user_id}:{count}:{candidate_count}"
    )

    cached = get_cached_recommendations(cache_key)

    if cached is not None:
        cached["cache_status"] = "HIT"
        return cached

    if user_id not in user_vocabulary:
        fallback = popularity_fallback(user_id, count)

        result = {
            "user_id": user_id,
            "recommendations": fallback,
            "count": len(fallback),
            "recommendation_method": "popularity_fallback",
            "cache_status": "MISS",
            "message": (
                "Unknown user. Recommendations use popular unseen movies."
            ),
        }

        cache_recommendations(cache_key, result)
        return result

    candidates = retrieve_candidates(
        user_id,
        candidate_count,
    )

    if not candidates:
        fallback = popularity_fallback(user_id, count)

        result = {
            "user_id": user_id,
            "recommendations": fallback,
            "count": len(fallback),
            "recommendation_method": "popularity_fallback",
            "cache_status": "MISS",
            "message": (
                "Personalized retrieval returned no candidates. "
                "Popularity fallback was used."
            ),
        }

        cache_recommendations(cache_key, result)
        return result

    candidate_ids = [
        movie_id
        for movie_id, _ in candidates
    ]

    retrieval_scores = {
        movie_id: similarity
        for movie_id, similarity in candidates
    }

    inputs = prepare_ranker_inputs(
        user_id,
        candidate_ids,
    )

    try:
        # Prepare Triton inference inputs.
        dense_features = inputs["dense_features"].astype(
            np.float32
        )

        movie_index = inputs["movie_id"].astype(
            np.int32
        )

        user_index = inputs["user_id"].astype(
            np.int32
        )

        # Build Triton HTTP inference request.
        triton_payload = {
            "inputs": [
                {
                    "name": "dense_features",
                    "shape": list(dense_features.shape),
                    "datatype": "FP32",
                    "data": dense_features.tolist(),
                },
                {
                    "name": "movie_index",
                    "shape": list(movie_index.shape),
                    "datatype": "INT32",
                    "data": movie_index.tolist(),
                },
                {
                    "name": "user_index",
                    "shape": list(user_index.shape),
                    "datatype": "INT32",
                    "data": user_index.tolist(),
                },
            ],
            "outputs": [
                {
                    "name": "relevance_v2"
                }
            ],
        }

        # Send request to Triton Inference Server.
        triton_response = requests.post(
            TRITON_INFER_URL,
            json=triton_payload,
            timeout=30,
        )

        triton_response.raise_for_status()

        triton_result = triton_response.json()

        # Extract DCN ranking scores.
        output_data = next(
            output["data"]
            for output in triton_result["outputs"]
            if output["name"] == "relevance_v2"
        )

        scores = np.asarray(
            output_data,
            dtype=np.float32,
        ).reshape(-1)

    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Triton ranking inference failed: {error}",
        ) from error

    if len(scores) != len(candidate_ids):
        raise HTTPException(
            status_code=500,
            detail="Ranking model returned an unexpected number of scores.",
        )

    ranked_positions = np.argsort(scores)[::-1][:count]

    recommendations = []

    for position in ranked_positions:
        movie_id = candidate_ids[int(position)]
        movie = movie_lookup.get(movie_id)

        if movie is None:
            continue

        recommendations.append(
            {
                **movie,
                "score": float(scores[position]),
                "retrieval_similarity": float(
                    retrieval_scores[movie_id]
                ),
                "recommendation_type": "personalized_dcn",
            }
        )

    result = {
        "user_id": user_id,
        "recommendations": recommendations,
        "count": len(recommendations),
        "recommendation_method": "two_tower_faiss_plus_dcn",
        "candidate_count": len(candidates),
        "cache_status": "MISS",
        "message": (
            "Recommendations were retrieved using FAISS "
            "and reranked using the DCN model."
        ),
    }

    cache_recommendations(
        cache_key,
        result,
    )

    return result


# ============================================================
# 11. MOVIE DETAILS ENDPOINT
# ============================================================

@app.get("/movies/{movie_id}")
def get_movie(movie_id: int):
    """Return movie information by movie ID."""

    movie = movie_lookup.get(movie_id)

    if movie is None:
        raise HTTPException(
            status_code=404,
            detail=f"Movie ID {movie_id} was not found.",
        )

    return movie


# ============================================================
# 12. POPULAR MOVIES ENDPOINT
# ============================================================

@app.get("/popular")
def get_popular_movies(
    count: int = Query(default=20, ge=1, le=100),
):
    """Return globally popular movies."""

    results = []

    for movie_id in popular_movie_ids[:count]:
        movie = movie_lookup.get(movie_id)

        if movie is not None:
            results.append(movie)

    return {
        "count": len(results),
        "movies": results,
        "method": "popularity",
    }


# ============================================================
# 13. CACHE MANAGEMENT ENDPOINT
# ============================================================

@app.delete("/cache/{user_id}")
def clear_user_cache(user_id: int):
    """Clear cached recommendation results for one user."""

    pattern = f"recommendations:v1:{user_id}:*"

    try:
        keys = list(
            redis_client.scan_iter(
                match=pattern,
                count=100,
            )
        )

        if keys:
            deleted = redis_client.delete(*keys)
        else:
            deleted = 0

        return {
            "user_id": user_id,
            "deleted_keys": int(deleted),
            "status": "success",
        }

    except redis.RedisError as error:
        raise HTTPException(
            status_code=503,
            detail=f"Redis cache is unavailable: {error}",
        )


# ============================================================
# 14. APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
    )
