
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"

INPUT_FILE = DATA_DIR / "train.csv"
OUTPUT_FILE = DATA_DIR / "retrieval_train_sample.csv"

SAMPLE_SIZE = 500_000
RANDOM_SEED = 42

print("Loading training interactions...")

# Read only the columns required by the retrieval model.
ratings = pd.read_csv(
    INPUT_FILE,
    usecols=["userId", "movieId", "rating", "timestamp"]
)

print(f"Available interactions: {len(ratings):,}")

# Keep a reproducible sample.
if len(ratings) > SAMPLE_SIZE:
    sample = ratings.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_SEED
    )
else:
    sample = ratings.copy()

# Sort chronologically for reproducibility and inspection.
sample = sample.sort_values(
    ["timestamp", "userId", "movieId"]
).reset_index(drop=True)

sample.to_csv(OUTPUT_FILE, index=False)

print(f"Sample interactions: {len(sample):,}")
print(f"Unique users: {sample['userId'].nunique():,}")
print(f"Unique movies: {sample['movieId'].nunique():,}")
print(f"Saved to: {OUTPUT_FILE}")
