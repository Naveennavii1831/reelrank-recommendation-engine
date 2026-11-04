
from pathlib import Path

import pandas as pd

# --------------------------------------------------
# 1. Define paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "ml-25m"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

RATINGS_FILE = RAW_DIR / "ratings.csv"
MOVIES_FILE = RAW_DIR / "movies.csv"

# --------------------------------------------------
# 2. Load the data
# --------------------------------------------------

print("Loading MovieLens dataset...")

ratings = pd.read_csv(RATINGS_FILE)
movies = pd.read_csv(MOVIES_FILE)

print(f"Raw ratings: {len(ratings):,}")
print(f"Raw movies: {len(movies):,}")

# --------------------------------------------------
# 3. Validate ratings
# --------------------------------------------------

required_columns = {"userId", "movieId", "rating", "timestamp"}

if not required_columns.issubset(ratings.columns):
    raise ValueError(
        f"Ratings dataset must contain: {required_columns}"
    )

ratings["userId"] = pd.to_numeric(
    ratings["userId"], errors="coerce"
)

ratings["movieId"] = pd.to_numeric(
    ratings["movieId"], errors="coerce"
)

ratings["rating"] = pd.to_numeric(
    ratings["rating"], errors="coerce"
)

ratings["timestamp"] = pd.to_numeric(
    ratings["timestamp"], errors="coerce"
)

# Remove missing and invalid values.
ratings = ratings.dropna(
    subset=["userId", "movieId", "rating", "timestamp"]
).copy()

ratings = ratings[
    (ratings["userId"] > 0)
    & (ratings["movieId"] > 0)
    & (ratings["rating"] >= 0.5)
    & (ratings["rating"] <= 5.0)
    & (ratings["rating"] * 2 % 1 == 0)
    & (ratings["timestamp"] > 0)
].copy()

ratings["userId"] = ratings["userId"].astype("int32")
ratings["movieId"] = ratings["movieId"].astype("int32")
ratings["rating"] = ratings["rating"].astype("float32")
ratings["timestamp"] = ratings["timestamp"].astype("int64")

print(f"Ratings after validation: {len(ratings):,}")

# --------------------------------------------------
# 4. Handle duplicate interactions
# --------------------------------------------------

# MovieLens may contain repeated ratings of the same movie
# by the same user at different times.
#
# We retain distinct timestamped events. Exact duplicate
# rows are removed.

ratings = ratings.drop_duplicates(
    subset=["userId", "movieId", "rating", "timestamp"]
).copy()

print(f"Ratings after exact deduplication: {len(ratings):,}")

# --------------------------------------------------
# 5. Validate movie metadata
# --------------------------------------------------

movies = movies.dropna(
    subset=["movieId", "title"]
).copy()

movies["movieId"] = pd.to_numeric(
    movies["movieId"], errors="coerce"
)

movies = movies.dropna(subset=["movieId"]).copy()

movies["movieId"] = movies["movieId"].astype("int32")

movies = movies.drop_duplicates(
    subset=["movieId"]
).copy()

# Keep only interactions associated with known movies.
known_movie_ids = set(movies["movieId"])

ratings = ratings[
    ratings["movieId"].isin(known_movie_ids)
].copy()

# Convert Unix timestamps into UTC datetime values.
ratings["event_time"] = pd.to_datetime(
    ratings["timestamp"],
    unit="s",
    utc=True
)

# --------------------------------------------------
# 6. Sort chronologically
# --------------------------------------------------

ratings = ratings.sort_values(
    ["timestamp", "userId", "movieId"]
).reset_index(drop=True)

# --------------------------------------------------
# 7. Create a global chronological split
# --------------------------------------------------

# The earliest 80% of interactions are used for training.
# The next 10% are used for validation.
# The latest 10% are reserved for testing.

n = len(ratings)

train_end = int(n * 0.80)
validation_end = int(n * 0.90)

train = ratings.iloc[:train_end].copy()
validation = ratings.iloc[train_end:validation_end].copy()
test = ratings.iloc[validation_end:].copy()

# --------------------------------------------------
# 8. Verify chronological ordering
# --------------------------------------------------

assert len(train) > 0
assert len(validation) > 0
assert len(test) > 0

assert train["timestamp"].max() <= validation["timestamp"].min()
assert validation["timestamp"].max() <= test["timestamp"].min()

# --------------------------------------------------
# 9. Build user and movie features from training data
# --------------------------------------------------

user_features = (
    train.groupby("userId")
    .agg(
        interaction_count=("movieId", "size"),
        average_rating=("rating", "mean"),
        last_interaction=("timestamp", "max")
    )
    .reset_index()
)

movie_features = (
    train.groupby("movieId")
    .agg(
        interaction_count=("userId", "size"),
        average_rating=("rating", "mean"),
        unique_users=("userId", "nunique")
    )
    .reset_index()
)

# Popularity is based on training interactions only.
movie_features = movie_features.sort_values(
    ["interaction_count", "movieId"],
    ascending=[False, True]
)

# Add movie titles and genres.
movie_features = movie_features.merge(
    movies[["movieId", "title", "genres"]],
    on="movieId",
    how="left"
)

# --------------------------------------------------
# 10. Save the processed datasets
# --------------------------------------------------

train.to_csv(PROCESSED_DIR / "train.csv", index=False)
validation.to_csv(PROCESSED_DIR / "validation.csv", index=False)
test.to_csv(PROCESSED_DIR / "test.csv", index=False)

user_features.to_csv(
    PROCESSED_DIR / "user_features.csv",
    index=False
)

movie_features.to_csv(
    PROCESSED_DIR / "movie_features.csv",
    index=False
)

movies.to_csv(
    PROCESSED_DIR / "movies_clean.csv",
    index=False
)

# --------------------------------------------------
# 11. Print the processing report
# --------------------------------------------------

print("\n" + "=" * 55)
print("DATA PREPARATION REPORT")
print("=" * 55)

print(f"Total cleaned interactions: {len(ratings):,}")
print(f"Unique users: {ratings['userId'].nunique():,}")
print(f"Unique rated movies: {ratings['movieId'].nunique():,}")

print("\nSplit sizes:")
print(f"Training:   {len(train):,} ({len(train) / n:.1%})")
print(f"Validation: {len(validation):,} ({len(validation) / n:.1%})")
print(f"Testing:    {len(test):,} ({len(test) / n:.1%})")

print("\nSplit timestamp boundaries:")
print(f"Training ends:   {train['event_time'].max()}")
print(f"Validation ends: {validation['event_time'].max()}")
print(f"Testing ends:    {test['event_time'].max()}")

print("\nTop 10 movies by training interaction count:")

top_movies = movie_features.head(10)[
    ["movieId", "title", "interaction_count", "average_rating"]
]

print(top_movies.to_string(index=False))

print("\nSaved files:")
for file in sorted(PROCESSED_DIR.glob("*.csv")):
    print(f"{file.name}: {file.stat().st_size / (1024 * 1024):.2f} MB")

print("\nData preparation completed successfully.")
