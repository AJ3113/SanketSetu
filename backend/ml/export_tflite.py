from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from backend.ml.config import (
    ARTIFACTS_DIR,
    KERAS_MODEL_PATH,
    MODEL_OUTPUT_PATH,
    MODELS_DIR,
)


def verify_tflite_model(model_path: Path) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"TFLite model does not exist at {model_path}")

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    sample = np.random.randn(1, 8, 128).astype(np.float32)
    interpreter.set_tensor(input_details["index"], sample)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])
    if output.shape[1] != 20:
        raise ValueError(f"Unexpected TFLite output shape: {output.shape}")
    print(f"TFLite verification OK: input_shape={input_details['shape']}, output_shape={output.shape}")


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not KERAS_MODEL_PATH.exists():
        raise FileNotFoundError(f"Keras model not found at {KERAS_MODEL_PATH}. Train the model first.")

    model = tf.keras.models.load_model(KERAS_MODEL_PATH)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    MODEL_OUTPUT_PATH.write_bytes(tflite_model)
    print(f"Exported TFLite model to {MODEL_OUTPUT_PATH}")
    verify_tflite_model(MODEL_OUTPUT_PATH)


if __name__ == "__main__":
    main()
