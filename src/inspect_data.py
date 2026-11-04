
from pathlib import Path

import pandas as pd

# Locate the project and dataset directories.
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "ml-25m"

ratings_path = DATA_DIR / "ratings.csv"
movies_path = DATA_DIR / "movies.csv"

# Check that the required files exist.
if not ratings_path.exists():
    raise FileNotFoundError(f"Ratings file not found: {ratings_path}")

if not movies_path.exists():
    raise FileNotFoundError(f"Movies file not found: {movies_path}")

print("=" * 60)
print("MOVIELENS 25M DATASET INSPECTION")
print("=" * 60)

# Load the datasets.
ratings = pd.read_csv(ratings_path)
movies = pd.read_csv(movies_path)

# Display dataset dimensions.
print("\n1. DATASET DIMENSIONS")
print(f"Number of ratings: {len(ratings):,}")
print(f"Number of movies: {len(movies):,}")
print(f"Unique users: {ratings['userId'].nunique():,}")
print(f"Unique rated movies: {ratings['movieId'].nunique():,}")

# Display column names and data types.
print("\n2. RATINGS DATA TYPES")
print(ratings.dtypes)

print("\n3. FIRST FIVE RATINGS")
print(ratings.head())

print("\n4. FIRST FIVE MOVIES")
print(movies.head())

# Check for missing values.
print("\n5. MISSING VALUES")
print("Ratings:")
print(ratings.isnull().sum())

print("\nMovies:")
print(movies.isnull().sum())

# Check for duplicate records.
print("\n6. DUPLICATE RECORDS")
print(f"Duplicate rating rows: {ratings.duplicated().sum()}")
print(f"Duplicate movie IDs: {movies['movieId'].duplicated().sum()}")

# Examine the rating distribution.
print("\n7. RATING DISTRIBUTION")
print(ratings["rating"].value_counts().sort_index())

# Examine the timestamp range.
ratings["timestamp"] = pd.to_datetime(
    ratings["timestamp"], unit="s", utc=True
)

print("\n8. RATING TIMESTAMP RANGE")
print(f"Earliest rating: {ratings['timestamp'].min()}")
print(f"Latest rating: {ratings['timestamp'].max()}")

print("\nDataset inspection completed successfully.")
