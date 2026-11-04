import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
import numpy as np
import tensorflow as tf
import tf_keras as keras
import tf2onnx

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "dcn_ranker_compat.h5"
OUTPUT_PATH = ROOT / "triton" / "models" / "dcn_ranker" / "1" / "model.onnx"


def clone_trained_layer(original, name):
    config = original.get_config()
    config["name"] = name
    return original.__class__.from_config(config)


def main():

    # 1. Load original trained model
    print("Loading original trained DCN model...")
    original = keras.models.load_model(MODEL_PATH, compile=False)
    print("Original model loaded successfully.")

    # 2. Create lookup-free inputs
    print("Creating lookup-free model inputs...")

    dense_input = keras.Input(
        shape=(24,), dtype=tf.float32, name="dense_features"
    )
    movie_index = keras.Input(
        shape=(1,), dtype=tf.int32, name="movie_index"
    )
    user_index = keras.Input(
        shape=(1,), dtype=tf.int32, name="user_index"
    )

    # 3. Create embedding layers
    print("Creating embedding layers...")

    user_embedding = keras.layers.Embedding(
        input_dim=89272, output_dim=16, name="user_embedding_v2"
    )
    movie_embedding = keras.layers.Embedding(
        input_dim=12368, output_dim=16, name="movie_embedding_v2"
    )

    user_embedding.build((None, 1))
    movie_embedding.build((None, 1))

    user_embedding.set_weights(
        original.get_layer("user_embedding").get_weights()
    )
    movie_embedding.set_weights(
        original.get_layer("movie_embedding").get_weights()
    )

    print("User embedding weights:", user_embedding.get_weights()[0].shape)
    print("Movie embedding weights:", movie_embedding.get_weights()[0].shape)

    # 4. Generate embedding vectors
    user_vector = keras.layers.Flatten(name="user_flatten_v2")(
        user_embedding(user_index)
    )
    movie_vector = keras.layers.Flatten(name="movie_flatten_v2")(
        movie_embedding(movie_index)
    )

    combined = keras.layers.Concatenate(name="features_concat_v2")(
        [user_vector, movie_vector, dense_input]
    )

    # 5. Clone trained Dense layers
    print("Copying trained DCN layers...")

    layer_names = [
        "cross_projection_1",
        "cross_projection_2",
        "dense",
        "dense_1",
        "relevance"
    ]

    cloned = {}

    for name in layer_names:
        source = original.get_layer(name)
        target = clone_trained_layer(source, name + "_v2")
        target.build(source.input_shape)
        target.set_weights(source.get_weights())
        cloned[name] = target
        print("Copied:", name)

    # 6. First cross-network layer
    print("Reconstructing DCN architecture...")

    cross_1 = cloned["cross_projection_1"](combined)

    cross_product_1 = keras.layers.Multiply(
        name="cross_multiply_1_v2"
    )([combined, cross_1])

    cross_output_1 = keras.layers.Add(
        name="cross_add_1_v2"
    )([cross_product_1, combined])

    # 7. First deep-network layer
    deep_1 = cloned["dense"](combined)

    deep_1 = keras.layers.Dropout(
        original.get_layer("dropout").rate,
        name="dropout_v2"
    )(deep_1)

    # 8. Second cross-network layer
    cross_2 = cloned["cross_projection_2"](cross_output_1)

    cross_product_2 = keras.layers.Multiply(
        name="cross_multiply_2_v2"
    )([combined, cross_2])

    cross_output_2 = keras.layers.Add(
        name="cross_add_2_v2"
    )([cross_product_2, cross_output_1])

    # 9. Second deep-network layer
    deep_2 = cloned["dense_1"](deep_1)

    deep_2 = keras.layers.Dropout(
        original.get_layer("dropout_1").rate,
        name="dropout_2_v2"
    )(deep_2)

    # 10. Final prediction layer
    merged = keras.layers.Concatenate(
        name="final_concat_v2"
    )([cross_output_2, deep_2])

    output = cloned["relevance"](merged)

    replacement = keras.Model(
        inputs=[dense_input, movie_index, user_index],
        outputs=output,
        name="dcn_ranker_lookup_free"
    )

    print("Replacement model created successfully.")

    # 11. Prepare verification data
    print("Preparing prediction verification data...")

    rng = np.random.default_rng(42)
    batch_size = 8

    dense = rng.normal(
        size=(batch_size, 24)
    ).astype(np.float32)

    movie_ids = np.array(
        [[1], [2], [3], [4], [5], [6], [7], [8]],
        dtype=np.int32
    )

    user_ids = np.array(
        [[1], [2], [3], [4], [5], [6], [7], [8]],
        dtype=np.int32
    )

    user_indices = original.get_layer("integer_lookup")(
        tf.constant(user_ids)
    ).numpy()

    movie_indices = original.get_layer("integer_lookup_1")(
        tf.constant(movie_ids)
    ).numpy()

    # 12. Compare predictions
    print("Comparing original and replacement predictions...")

    original_predictions = original.predict(
        [dense, movie_ids, user_ids],
        verbose=0
    )

    replacement_predictions = replacement.predict(
        [dense, movie_indices, user_indices],
        verbose=0
    )

    difference = np.max(
        np.abs(original_predictions - replacement_predictions)
    )

    print("Maximum prediction difference:", float(difference))

    if not np.allclose(
        original_predictions,
        replacement_predictions,
        rtol=1e-5,
        atol=1e-6
    ):
        raise RuntimeError(
            "Prediction verification FAILED. ONNX export cancelled."
        )

    print("Prediction verification PASSED.")

    # 13. Export verified model to ONNX
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    signature = [
        tf.TensorSpec(
            (None, 24), tf.float32, name="dense_features"
        ),
        tf.TensorSpec(
            (None, 1), tf.int32, name="movie_index"
        ),
        tf.TensorSpec(
            (None, 1), tf.int32, name="user_index"
        )
    ]

    print("Exporting replacement model to ONNX...")

    model_proto, _ = tf2onnx.convert.from_keras(
        replacement,
        input_signature=signature,
        opset=17,
        output_path=str(OUTPUT_PATH)
    )

    print("ONNX export completed successfully.")
    print("Output:", OUTPUT_PATH)
    print("File size:", OUTPUT_PATH.stat().st_size, "bytes")
    print("ONNX opset:", model_proto.opset_import[0].version)


if __name__ == "__main__":
    main()