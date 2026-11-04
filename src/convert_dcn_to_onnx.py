import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from pathlib import Path
import tf2onnx
import tf_keras as keras
import tensorflow as tf
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "dcn_ranker_compat.h5"
OUTPUT_PATH = ROOT / "triton" / "models" / "dcn_ranker" / "1" / "model.onnx"
def main():
    print("Loading original DCN model...")

    model = keras.models.load_model(MODEL_PATH, compile=False)
    print("Original model loaded.")

    print("Model inputs:")
    for tensor in model.inputs:
        print(tensor.name, tensor.shape, tensor.dtype)

    input_signature = [
        tf.TensorSpec(shape=(None, 24), dtype=tf.float32, name="dense_features"),
        tf.TensorSpec(shape=(None, 1), dtype=tf.int32, name="movie_id"),
        tf.TensorSpec(shape=(None, 1), dtype=tf.int32, name="user_id"),
    ]

    print("Converting DCN model to ONNX...")
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=17, output_path=str(OUTPUT_PATH))
    print("ONNX conversion completed.")
    print("Output:", OUTPUT_PATH)
    print("Size:", OUTPUT_PATH.stat().st_size, "bytes")
    print("ONNX opset:", model_proto.opset_import[0].version)

if __name__ == "__main__":
    main()
