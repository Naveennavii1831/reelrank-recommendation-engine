
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras as keras
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

TRAIN_FILE = DATA / "train.csv"
USER_FILE = DATA / "user_features.csv"
MOVIE_FILE = DATA / "movie_features.csv"

SAMPLE_SIZE = 300_000
EPOCHS = 5
BATCH_SIZE = 2048
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

MODELS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Load a reproducible sample from the training interactions
# ---------------------------------------------------------
print("Loading training sample...")

# Read only the columns needed and sample rows reproducibly.
# The sample is drawn from the chronological training partition.
train = pd.read_csv(
    TRAIN_FILE,
    usecols=["userId", "movieId", "rating"],
)

if len(train) > SAMPLE_SIZE:
    train = train.sample(n=SAMPLE_SIZE, random_state=SEED)

print(f"Training interactions sampled: {len(train):,}")

# ---------------------------------------------------------
# Load user and movie features
# ---------------------------------------------------------
print("Loading user and movie features...")

users = pd.read_csv(USER_FILE)
movies = pd.read_csv(
    MOVIE_FILE,
    usecols=[
        "movieId",
        "interaction_count",
        "average_rating",
        "unique_users",
        "genres",
    ],
)

# Prefix feature names to avoid ambiguity after joins.
users = users.rename(
    columns={
        "interaction_count": "user_interaction_count",
        "average_rating": "user_average_rating",
    }
)

movies = movies.rename(
    columns={
        "interaction_count": "movie_interaction_count",
        "average_rating": "movie_average_rating",
        "unique_users": "movie_unique_users",
    }
)

# ---------------------------------------------------------
# Join features
# ---------------------------------------------------------
print("Joining features...")

data = train.merge(users, on="userId", how="left")
data = data.merge(movies, on="movieId", how="left")

# A rating >= 4.0 is treated as a positive relevance label.
data["label"] = (data["rating"] >= 4.0).astype(np.float32)

numeric_features = [
    "user_interaction_count",
    "user_average_rating",
    "movie_interaction_count",
    "movie_average_rating",
    "movie_unique_users",
]

for col in numeric_features:
    data[col] = pd.to_numeric(data[col], errors="coerce")
    data[col] = data[col].replace([np.inf, -np.inf], np.nan)
    data[col] = data[col].fillna(data[col].median())

# Log-transform count features to reduce their scale.
for col in [
    "user_interaction_count",
    "movie_interaction_count",
    "movie_unique_users",
]:
    data[col] = np.log1p(data[col].clip(lower=0))

# Genres are represented using a compact multi-hot vector.
genre_vocabulary = sorted(
    {
        genre
        for value in data["genres"].fillna("")
        for genre in str(value).split("|")
        if genre and genre != "(no genres listed)"
    }
)

genre_to_index = {genre: i for i, genre in enumerate(genre_vocabulary)}
genre_matrix = np.zeros(
    (len(data), len(genre_vocabulary)),
    dtype=np.float32,
)

for row, value in enumerate(data["genres"].fillna("")):
    for genre in str(value).split("|"):
        index = genre_to_index.get(genre)
        if index is not None:
            genre_matrix[row, index] = 1.0

# Use integer identifiers as categorical model inputs.
data["userId"] = data["userId"].astype(np.int32)
data["movieId"] = data["movieId"].astype(np.int32)

# Scale numeric inputs.
scaler = StandardScaler()
numeric_matrix = scaler.fit_transform(
    data[numeric_features].astype(np.float32)
).astype(np.float32)

user_ids = data["userId"].to_numpy(dtype=np.int32)
movie_ids = data["movieId"].to_numpy(dtype=np.int32)
labels = data["label"].to_numpy(dtype=np.float32)

# The genre matrix is appended to numeric features.
dense_matrix = np.concatenate(
    [numeric_matrix, genre_matrix],
    axis=1,
).astype(np.float32)

print(f"Users represented: {len(np.unique(user_ids)):,}")
print(f"Movies represented: {len(np.unique(movie_ids)):,}")
print(f"Genre features: {len(genre_vocabulary)}")
print(f"Positive-label rate: {labels.mean():.4f}")

# ---------------------------------------------------------
# Split sampled interactions into train and internal validation
# ---------------------------------------------------------
indices = np.arange(len(labels))

train_idx, valid_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=SEED,
    stratify=labels,
)

# Note: this internal split is random and is for training diagnostics.
# The separate chronological validation.csv remains untouched for
# the later recommendation-ranking evaluation.

# ---------------------------------------------------------
# Build a DCN-style ranking model
# ---------------------------------------------------------
print("Building DCN ranking model...")

unique_users = np.unique(user_ids)
unique_movies = np.unique(movie_ids)

user_lookup = keras.layers.IntegerLookup(
    vocabulary=unique_users.tolist(),
    mask_token=None,
    num_oov_indices=1,
)

movie_lookup = keras.layers.IntegerLookup(
    vocabulary=unique_movies.tolist(),
    mask_token=None,
    num_oov_indices=1,
)

user_input = keras.Input(shape=(1,), dtype=tf.int32, name="user_id")
movie_input = keras.Input(shape=(1,), dtype=tf.int32, name="movie_id")
dense_input = keras.Input(
    shape=(dense_matrix.shape[1],),
    dtype=tf.float32,
    name="dense_features",
)

user_index = user_lookup(user_input)
movie_index = movie_lookup(movie_input)

user_embedding = keras.layers.Embedding(
    input_dim=unique_users.size + 1,
    output_dim=16,
    name="user_embedding",
)(user_index)

movie_embedding = keras.layers.Embedding(
    input_dim=unique_movies.size + 1,
    output_dim=16,
    name="movie_embedding",
)(movie_index)

user_embedding = keras.layers.Flatten()(user_embedding)
movie_embedding = keras.layers.Flatten()(movie_embedding)

x0 = keras.layers.Concatenate()(
    [user_embedding, movie_embedding, dense_input]
)

# Cross network: explicit feature interactions.
cross = x0
for layer_number in range(2):
    cross_projection = keras.layers.Dense(
        x0.shape[-1],
        use_bias=True,
        name=f"cross_projection_{layer_number + 1}",
    )(cross)

    cross = keras.layers.Add()(
        [
            keras.layers.Multiply()([x0, cross_projection]),
            cross,
        ]
    )

# Deep network: nonlinear feature interactions.
deep = keras.layers.Dense(128, activation="relu")(x0)
deep = keras.layers.Dropout(0.2)(deep)
deep = keras.layers.Dense(64, activation="relu")(deep)
deep = keras.layers.Dropout(0.1)(deep)

combined = keras.layers.Concatenate()([cross, deep])
output = keras.layers.Dense(1, activation="sigmoid", name="relevance")(combined)

model = keras.Model(
    inputs={
        "user_id": user_input,
        "movie_id": movie_input,
        "dense_features": dense_input,
    },
    outputs=output,
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[
        keras.metrics.AUC(name="auc"),
        keras.metrics.BinaryAccuracy(name="accuracy"),
    ],
)

model.summary()

# ---------------------------------------------------------
# Build TensorFlow datasets
# ---------------------------------------------------------
def make_dataset(index_array, shuffle):
    features = {
        "user_id": user_ids[index_array].reshape(-1, 1),
        "movie_id": movie_ids[index_array].reshape(-1, 1),
        "dense_features": dense_matrix[index_array],
    }
    target = labels[index_array]

    ds = tf.data.Dataset.from_tensor_slices((features, target))

    if shuffle:
        ds = ds.shuffle(
            min(len(index_array), 50_000),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    return ds.batch(BATCH_SIZE).prefetch(1)


train_ds = make_dataset(train_idx, shuffle=True)
valid_ds = make_dataset(valid_idx, shuffle=False)

# ---------------------------------------------------------
# Train and evaluate
# ---------------------------------------------------------
print("\nTraining DCN ranking model...")
start = time.time()

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    verbose=1,
)

training_seconds = time.time() - start

print("\nCalculating internal validation metrics...")
valid_features = {
    "user_id": user_ids[valid_idx].reshape(-1, 1),
    "movie_id": movie_ids[valid_idx].reshape(-1, 1),
    "dense_features": dense_matrix[valid_idx],
}

predictions = model.predict(
    valid_features,
    batch_size=BATCH_SIZE,
    verbose=0,
).reshape(-1)

auc = float(roc_auc_score(labels[valid_idx], predictions))
validation_log_loss = float(
    log_loss(labels[valid_idx], np.clip(predictions, 1e-7, 1 - 1e-7))
)

# ---------------------------------------------------------
# Save model and preprocessing metadata
# ---------------------------------------------------------
model_path = MODELS / "dcn_ranker.keras"
model.save(model_path)

# Save model lookup vocabularies and numeric scaling parameters.
metadata = {
    "model_type": "Deep & Cross Network (DCN-style)",
    "label_definition": "rating >= 4.0",
    "sample_size": int(len(data)),
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "seed": SEED,
    "user_vocabulary": [int(x) for x in unique_users],
    "movie_vocabulary": [int(x) for x in unique_movies],
    "numeric_features": numeric_features,
    "numeric_scaler_mean": scaler.mean_.tolist(),
    "numeric_scaler_scale": scaler.scale_.tolist(),
    "genre_vocabulary": genre_vocabulary,
    "genre_to_index": genre_to_index,
    "training_seconds": round(training_seconds, 2),
    "internal_validation_auc": auc,
    "internal_validation_log_loss": validation_log_loss,
    "internal_validation_split": "random 80/20 within sampled training interactions",
    "chronological_validation_used_for_training": False,
}

metadata_path = MODELS / "dcn_ranker_metadata.json"
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

report = {
    "model_type": "DCN-style binary relevance ranker",
    "sample_size": int(len(data)),
    "positive_label_rate": float(labels.mean()),
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "training_seconds": round(training_seconds, 2),
    "internal_validation_auc": auc,
    "internal_validation_log_loss": validation_log_loss,
    "final_training_loss": float(history.history["loss"][-1]),
    "final_internal_validation_loss": float(
        history.history["val_loss"][-1]
    ),
    "limitations": [
        "Internal validation uses a random split of sampled training interactions.",
        "Chronological ranking evaluation has not yet been performed.",
        "Feature tables are based on the training partition, but the random internal split can share users and movies across both sides.",
        "AUC and log loss measure binary relevance prediction, not top-K recommendation quality.",
        "The model is an educational prototype."
    ],
}

report_path = REPORTS / "ranking_training_report.json"
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\n" + "=" * 55)
print("DCN RANKING TRAINING COMPLETED")
print("=" * 55)
print(f"Training interactions: {len(data):,}")
print(f"Training time: {training_seconds:.2f} seconds")
print(f"Internal validation AUC: {auc:.4f}")
print(f"Internal validation log loss: {validation_log_loss:.4f}")
print(f"Saved model: {model_path}")
print(f"Saved metadata: {metadata_path}")
print(f"Saved report: {report_path}")
print("\nNote: top-K chronological evaluation is the next step.")
