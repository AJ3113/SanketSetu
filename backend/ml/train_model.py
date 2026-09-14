from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from backend.ml.config import (
    ARTIFACTS_DIR,
    CLASS_NAMES,
    CONFUSION_MATRIX_PATH,
    DATASET_PATH,
    FRAME_FEATURE_DIM,
    KERAS_MODEL_PATH,
    LABELS_PATH,
    METRICS_PATH,
    PREPROCESSING_JSON_PATH,
    ROOT_DATASET_PATH,
    ROOT_LABELS_PATH,
    ROOT_MODEL_PATH,
    ROOT_PREPROCESSING_JSON_PATH,
)


def load_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    # Prefer root dataset.npz if present or fallback to DATASET_PATH
    dataset_file = ROOT_DATASET_PATH if ROOT_DATASET_PATH.exists() else DATASET_PATH
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_file}. Run build_dataset first.")

    # Load labels
    labels_file = ROOT_LABELS_PATH if ROOT_LABELS_PATH.exists() else LABELS_PATH
    if labels_file.exists():
        with open(labels_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        labels = raw if isinstance(raw, list) else list(raw.keys())
    else:
        labels = list(CLASS_NAMES)

    data = np.load(dataset_file)
    x = data["x"].astype(np.float32)
    y = data["y"].astype(np.int32)
    split = data["split"].astype(str) if "split" in data else None

    if x.ndim != 2 or x.shape[1] != FRAME_FEATURE_DIM:
        raise ValueError(f"Expected 2D dataset with shape (samples, {FRAME_FEATURE_DIM}), got {x.shape}")

    if split is not None:
        train_mask = split == "train"
        val_mask = split == "val"
        test_mask = split == "test"
    else:
        from sklearn.model_selection import train_test_split
        idx = np.arange(len(x))
        train_idx, temp_idx = train_test_split(idx, test_size=0.3, random_state=42, stratify=y)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=y[temp_idx])
        train_mask = np.isin(idx, train_idx)
        val_mask = np.isin(idx, val_idx)
        test_mask = np.isin(idx, test_idx)

    x_train, y_train = x[train_mask], y[train_mask]
    x_val, y_val = x[val_mask], y[val_mask]
    x_test, y_test = x[test_mask], y[test_mask]

    return x_train, y_train, x_val, y_val, x_test, y_test, labels


def build_model(input_dim: int = 126, num_classes: int = 5, dropout_rate: float = 0.2) -> tf.keras.Model:
    """Architecture:
    Input 126
    Dense 256 ReLU
    Dropout
    Dense 128 ReLU
    Dense 5 Softmax
    """
    inputs = tf.keras.Input(shape=(input_dim,), name="landmarks_input")
    x = tf.keras.layers.Dense(256, activation="relu", name="dense_256")(inputs)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense_128")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="dense_5_softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mlp_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def print_confusion_matrix(cm: np.ndarray, labels: list[str]) -> None:
    title_col = "Actual \\ Pred"
    header = f"{title_col:<16} " + " ".join(f"{lbl[:10]:>12}" for lbl in labels)
    print("\n" + "=" * len(header))
    print("CONFUSION MATRIX")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for i, actual_label in enumerate(labels):
        row = f"{actual_label:<16} " + " ".join(f"{cm[i, j]:>12d}" for j in range(len(labels)))
        print(row)
    print("=" * len(header) + "\n")


def train() -> tuple[tf.keras.Model, float, np.ndarray]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_val, y_val, x_test, y_test, labels = load_dataset()
    num_classes = len(labels)

    print(f"Loaded dataset:")
    print(f"  Train samples: {len(x_train)}")
    print(f"  Val samples  : {len(x_val)}")
    print(f"  Test samples : {len(x_test)}")
    print(f"  Feature dim  : {x_train.shape[1]}")
    print(f"  Classes ({num_classes}): {labels}\n")

    model = build_model(input_dim=x_train.shape[1], num_classes=num_classes, dropout_rate=0.2)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        )
    ]

    print("\nTraining MLP classifier...")
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=60,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate on test set
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    test_accuracy = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=list(range(num_classes)))

    print("\n" + "=" * 50)
    print(f"TEST ACCURACY: {test_accuracy * 100:.2f}% ({test_accuracy:.4f})")
    print("=" * 50)

    print_confusion_matrix(cm, labels)

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=labels, digits=4))

    # Save model.keras
    for p in [ROOT_MODEL_PATH, KERAS_MODEL_PATH]:
        p.parent.mkdir(parents=True, exist_ok=True)
        model.save(p)
        print(f"Saved model to: {p}")

    # Save labels.json
    for p in [ROOT_LABELS_PATH, LABELS_PATH]:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(labels, f, indent=2)
        print(f"Saved labels to: {p}")

    # Save preprocessing.json
    prep_config = {
        "feature_dimension": FRAME_FEATURE_DIM,
        "input_shape": [FRAME_FEATURE_DIM],
        "landmarks_per_hand": 21,
        "coords_per_landmark": 3,
        "num_hands": 2,
        "hand_ordering": ["left", "right"],
        "normalization": "wrist_index_pinky_center_max_scale",
        "missing_hand_fill": "zeros",
        "labels": labels,
        "num_classes": len(labels),
    }
    for p in [ROOT_PREPROCESSING_JSON_PATH, PREPROCESSING_JSON_PATH]:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(prep_config, f, indent=2)
        print(f"Saved preprocessing config to: {p}")

    # Also save metrics
    metrics = {
        "test_accuracy": test_accuracy,
        "confusion_matrix": cm.tolist(),
        "classes": labels,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    np.save(CONFUSION_MATRIX_PATH, cm)

    return model, test_accuracy, cm


def main() -> None:
    train()


if __name__ == "__main__":
    main()
