
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data" / "processed" / "retrieval_train_sample.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_DIM = 32
BATCH_SIZE = 2048
EPOCHS = 5
LEARNING_RATE = 0.05
RANDOM_SEED = 42
POSITIVE_RATING_THRESHOLD = 4.0

tf.keras.utils.set_random_seed(RANDOM_SEED)

print("=" * 60)
print("TWO-TOWER MOVIE RETRIEVAL MODEL")
print("=" * 60)


# ============================================================
# 2. LOAD AND PREPARE TRAINING DATA
# ============================================================

print("\nLoading interaction sample...")

df = pd.read_csv(
    DATA_FILE,
    usecols=["userId", "movieId", "rating"],
)

original_interactions = len(df)

print(f"Original interactions: {original_interactions:,}")

# Retain positive interactions.
df = df[df["rating"] >= POSITIVE_RATING_THRESHOLD].copy()

df["userId"] = df["userId"].astype(str)
df["movieId"] = df["movieId"].astype(str)

if df.empty:
    raise ValueError("No positive interactions were found.")

print(f"Positive interactions: {len(df):,}")
print(f"Unique users: {df['userId'].nunique():,}")
print(f"Unique movies: {df['movieId'].nunique():,}")

# Build ID vocabularies from the training sample.
user_ids = sorted(df["userId"].unique().tolist())
movie_ids = sorted(df["movieId"].unique().tolist())

print("\nCreating TensorFlow datasets...")

interactions = tf.data.Dataset.from_tensor_slices(
    {
        "userId": df["userId"].to_numpy(),
        "movieId": df["movieId"].to_numpy(),
    }
)

interactions = interactions.shuffle(
    buffer_size=min(len(df), 100_000),
    seed=RANDOM_SEED,
    reshuffle_each_iteration=True,
)

train_dataset = (
    interactions
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


# ============================================================
# 3. DEFINE THE USER TOWER
# ============================================================

class UserTower(tf.keras.Model):
    """Maps a user ID to a learned user embedding."""

    def __init__(self, vocabulary, embedding_dim):
        super().__init__()

        self.lookup = tf.keras.layers.StringLookup(
            vocabulary=vocabulary,
            mask_token=None,
            num_oov_indices=1,
        )

        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.lookup.vocabulary_size(),
            output_dim=embedding_dim,
        )

        self.dense = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(embedding_dim),
            ]
        )

    def call(self, user_ids):
        ids = self.lookup(user_ids)
        embeddings = self.embedding(ids)
        return self.dense(embeddings)


# ============================================================
# 4. DEFINE THE MOVIE TOWER
# ============================================================

class MovieTower(tf.keras.Model):
    """Maps a movie ID to a learned movie embedding."""

    def __init__(self, vocabulary, embedding_dim):
        super().__init__()

        self.lookup = tf.keras.layers.StringLookup(
            vocabulary=vocabulary,
            mask_token=None,
            num_oov_indices=1,
        )

        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.lookup.vocabulary_size(),
            output_dim=embedding_dim,
        )

        self.dense = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(embedding_dim),
            ]
        )

    def call(self, movie_ids):
        ids = self.lookup(movie_ids)
        embeddings = self.embedding(ids)
        return self.dense(embeddings)


# ============================================================
# 5. DEFINE THE RETRIEVAL MODEL
# ============================================================

class MovieRetrievalModel(tfrs.Model):
    """Two-tower retrieval model trained with in-batch negatives."""

    def __init__(self, user_model, movie_model):
        super().__init__()

        self.user_model = user_model
        self.movie_model = movie_model

        self.task = tfrs.tasks.Retrieval(
            metrics=None,
            temperature=0.1,
        )

    def compute_loss(self, features, training=False):
        user_embeddings = self.user_model(features["userId"])
        movie_embeddings = self.movie_model(features["movieId"])

        # Other movies in the batch act as negative candidates.
        return self.task(
            user_embeddings,
            movie_embeddings,
            compute_metrics=False,
        )


user_model = UserTower(user_ids, EMBEDDING_DIM)
movie_model = MovieTower(movie_ids, EMBEDDING_DIM)

model = MovieRetrievalModel(
    user_model=user_model,
    movie_model=movie_model,
)

# IMPORTANT: use the configured learning rate.
model.compile(
    optimizer=tf.keras.optimizers.Adagrad(
        learning_rate=LEARNING_RATE
    )
)


# ============================================================
# 6. TRAIN THE MODEL
# ============================================================

print("\nTraining configuration:")
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Embedding dimension: {EMBEDDING_DIM}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Random seed: {RANDOM_SEED}")
print(f"TensorFlow devices: {tf.config.list_physical_devices()}")

start_time = time.time()

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    verbose=1,
)

training_seconds = time.time() - start_time

loss_values = [
    float(value)
    for value in history.history["loss"]
]

if not loss_values or not np.isfinite(loss_values).all():
    raise RuntimeError(
        "Training did not produce valid finite loss values."
    )


# ============================================================
# 7. GENERATE USER AND MOVIE EMBEDDINGS
# ============================================================

print("\nGenerating user embeddings...")

user_embeddings = user_model(
    tf.constant(user_ids)
).numpy().astype(np.float32)

print("Generating movie embeddings...")

movie_embeddings = movie_model(
    tf.constant(movie_ids)
).numpy().astype(np.float32)

if not np.isfinite(user_embeddings).all():
    raise RuntimeError("User embeddings contain invalid values.")

if not np.isfinite(movie_embeddings).all():
    raise RuntimeError("Movie embeddings contain invalid values.")

print(f"User embedding shape: {user_embeddings.shape}")
print(f"Movie embedding shape: {movie_embeddings.shape}")


# ============================================================
# 8. SAVE EMBEDDINGS AND MODEL WEIGHTS
# ============================================================

embeddings_path = MODEL_DIR / "retrieval_embeddings.npz"
weights_path = MODEL_DIR / "two_tower.weights.h5"
metadata_path = MODEL_DIR / "retrieval_metadata.json"

np.savez_compressed(
    embeddings_path,
    movie_ids=np.asarray(movie_ids),
    movie_embeddings=movie_embeddings,
    user_ids=np.asarray(user_ids),
    user_embeddings=user_embeddings,
)

# This saves weights. Reconstruct the same architecture and vocabularies
# before attempting to load these weights in a future session.
model.save_weights(weights_path)


# ============================================================
# 9. SAVE TRAINING METADATA
# ============================================================

metadata = {
    "model": "Two-tower retrieval",
    "dataset": "MovieLens 25M development sample",
    "original_sample_interactions": int(original_interactions),
    "positive_interactions": int(len(df)),
    "unique_users": int(len(user_ids)),
    "unique_movies": int(len(movie_ids)),
    "positive_rating_threshold": POSITIVE_RATING_THRESHOLD,
    "embedding_dimension": EMBEDDING_DIM,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "random_seed": RANDOM_SEED,
    "training_seconds": round(training_seconds, 2),
    "device": "CPU-compatible TensorFlow execution",
}

with open(metadata_path, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=2)


# ============================================================
# 10. SAVE TRAINING REPORT
# ============================================================

report = {
    "model": "Two-tower retrieval",
    "epochs_completed": len(loss_values),
    "training_loss_by_epoch": loss_values,
    "final_training_loss": loss_values[-1],
    "training_seconds": round(training_seconds, 2),
    "embedding_dimension": EMBEDDING_DIM,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "positive_interactions": int(len(df)),
    "unique_users": int(len(user_ids)),
    "unique_movies": int(len(movie_ids)),
}

report_path = REPORT_DIR / "retrieval_training_report.json"

with open(report_path, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=2)


# ============================================================
# 11. PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)

print(f"Training time: {training_seconds:.2f} seconds")

for epoch, loss in enumerate(loss_values, start=1):
    print(f"Epoch {epoch} recorded loss: {loss:.4f}")

print(f"\nFinal recorded training loss: {loss_values[-1]:.4f}")
print(f"Saved weights: {weights_path}")
print(f"Saved embeddings: {embeddings_path}")
print(f"Saved metadata: {metadata_path}")
print(f"Saved training report: {report_path}")
print("=" * 60)
