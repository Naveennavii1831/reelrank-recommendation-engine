
from pathlib import Path

import faiss
import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "processed"

EMBEDDINGS_FILE = MODELS_DIR / "retrieval_embeddings.npz"
INDEX_FILE = MODELS_DIR / "movie_faiss.index"
MOVIE_IDS_FILE = MODELS_DIR / "faiss_movie_ids.npy"

SAMPLE_FILE = DATA_DIR / "retrieval_train_sample.csv"
MOVIES_FILE = DATA_DIR / "movies_clean.csv"


# --------------------------------------------------
# 2. Load saved embeddings
# --------------------------------------------------

print("Loading two-tower embeddings...")

if not EMBEDDINGS_FILE.exists():
    raise FileNotFoundError(
        f"Embedding file not found: {EMBEDDINGS_FILE}"
    )

with np.load(EMBEDDINGS_FILE, allow_pickle=False) as data:
    movie_ids = data["movie_ids"].astype(str)
    movie_embeddings = data["movie_embeddings"].astype(np.float32)

    user_ids = data["user_ids"].astype(str)
    user_embeddings = data["user_embeddings"].astype(np.float32)

print(f"Movies with embeddings: {len(movie_ids):,}")
print(f"Users with embeddings: {len(user_ids):,}")
print(f"Embedding dimension: {movie_embeddings.shape[1]}")


# --------------------------------------------------
# 3. Validate and normalize movie embeddings
# --------------------------------------------------

if movie_embeddings.ndim != 2 or len(movie_ids) != len(movie_embeddings):
    raise ValueError("Movie IDs and embeddings have incompatible shapes.")

if not np.isfinite(movie_embeddings).all():
    raise ValueError("Movie embeddings contain NaN or infinite values.")

valid_movies = np.linalg.norm(movie_embeddings, axis=1) > 0

movie_ids = movie_ids[valid_movies]
movie_embeddings = movie_embeddings[valid_movies]

movie_embeddings = np.ascontiguousarray(movie_embeddings, dtype=np.float32)
faiss.normalize_L2(movie_embeddings)

print(f"Valid movie embeddings: {len(movie_ids):,}")


# --------------------------------------------------
# 4. Build and save the FAISS index
# --------------------------------------------------

print("\nBuilding FAISS index...")

embedding_dimension = movie_embeddings.shape[1]

# Inner-product search on normalized vectors produces
# cosine similarity scores.
index = faiss.IndexFlatIP(embedding_dimension)
index.add(movie_embeddings)

faiss.write_index(index, str(INDEX_FILE))
np.save(MOVIE_IDS_FILE, movie_ids)

print(f"FAISS index saved: {INDEX_FILE}")
print(f"Movie-ID mapping saved: {MOVIE_IDS_FILE}")
print(f"Vectors indexed: {index.ntotal:,}")


# --------------------------------------------------
# 5. Load data needed to demonstrate recommendations
# --------------------------------------------------

print("\nLoading sample interaction history...")

interactions = pd.read_csv(
    SAMPLE_FILE,
    usecols=["userId", "movieId", "rating"]
)

interactions["userId"] = interactions["userId"].astype(str)
interactions["movieId"] = interactions["movieId"].astype(str)

# Select an available user with a learned embedding.
example_user_id = user_ids[0]

user_position = np.flatnonzero(user_ids == example_user_id)[0]
query = user_embeddings[user_position:user_position + 1].copy()
query = np.ascontiguousarray(query, dtype=np.float32)

if not np.isfinite(query).all():
    raise ValueError("Selected user embedding contains invalid values.")

if np.linalg.norm(query) == 0:
    raise ValueError("Selected user embedding has zero length.")

faiss.normalize_L2(query)

# Exclude movies this user rated in the sampled interaction data.
rated_movie_ids = set(
    interactions.loc[
        interactions["userId"] == example_user_id,
        "movieId"
    ]
)

# Search broadly enough to allow for filtering previously rated movies.
search_k = min(index.ntotal, 1000)
scores, positions = index.search(query, search_k)

recommendations = []

for score, position in zip(scores[0], positions[0]):
    if position < 0:
        continue

    movie_id = movie_ids[position]

    if movie_id in rated_movie_ids:
        continue

    recommendations.append(
        {
            "movieId": movie_id,
            "cosine_similarity": float(score)
        }
    )

    if len(recommendations) == 10:
        break


# --------------------------------------------------
# 6. Display movie titles when available
# --------------------------------------------------

recommendations_df = pd.DataFrame(recommendations)

if recommendations_df.empty:
    print(
        "\nNo unseen recommendations found in the first "
        f"{search_k} search results."
    )
else:
    if MOVIES_FILE.exists():
        movies = pd.read_csv(MOVIES_FILE)

        if "movieId" in movies.columns and "title" in movies.columns:
            movies["movieId"] = movies["movieId"].astype(str)

            recommendations_df = recommendations_df.merge(
                movies[["movieId", "title"]],
                on="movieId",
                how="left"
            )

    print(f"\nExample user ID: {example_user_id}")
    print("\nTop recommendations:")

    print(recommendations_df.to_string(index=False))


# --------------------------------------------------
# 7. Summary
# --------------------------------------------------

print("\nFAISS index creation completed successfully.")
print(f"Index file: {INDEX_FILE}")
print(f"Movie mapping: {MOVIE_IDS_FILE}")
