from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
import tensorflow as tf

from backend.ml.config import (
    FRAME_FEATURE_DIM,
    KERAS_MODEL_PATH,
    LABELS_PATH,
    PROJECT_ROOT,
    ROOT_LABELS_PATH,
    ROOT_MODEL_PATH,
)
from backend.ml.preprocessing import (
    create_hand_landmarker,
    extract_landmarks_from_frame,
)
from mediapipe.tasks.python.vision import RunningMode


DEFAULT_LABELS = [
    "HELLO",
    "THANK YOU",
    "YES",
    "NO",
    "HELP",
]

CONFIDENCE_THRESHOLD = 0.7


class SignClassifier:
    """Inference engine for real-time 126-D hand landmark classification using model.keras."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        labels_path: Optional[Union[str, Path]] = None,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        self.confidence_threshold = confidence_threshold

        # Resolve model path (check root model.keras, then artifacts)
        if model_path is None:
            if ROOT_MODEL_PATH.exists():
                model_path = ROOT_MODEL_PATH
            elif KERAS_MODEL_PATH.exists():
                model_path = KERAS_MODEL_PATH
            else:
                raise FileNotFoundError(f"Model file model.keras not found at {ROOT_MODEL_PATH} or {KERAS_MODEL_PATH}")
        self.model_path = Path(model_path)
        self.model = tf.keras.models.load_model(str(self.model_path))

        # Resolve labels path
        if labels_path is None:
            if ROOT_LABELS_PATH.exists():
                labels_path = ROOT_LABELS_PATH
            elif LABELS_PATH.exists():
                labels_path = LABELS_PATH

        if labels_path and Path(labels_path).exists():
            with open(labels_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.labels = raw if isinstance(raw, list) else list(raw.keys())
        else:
            self.labels = list(DEFAULT_LABELS)

        # Initialize MediaPipe HandLandmarker with exact preprocessing settings (IMAGE mode, up to 2 hands)
        self.landmarker = create_hand_landmarker(running_mode=RunningMode.IMAGE, num_hands=2)

    def extract_features(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Extract 126-D normalized landmark vector using exact preprocessing."""
        return extract_landmarks_from_frame(frame_bgr, self.landmarker)

    def predict_vector(self, feature_vector: np.ndarray) -> Dict:
        """Predict sign label and confidence from a 126-D landmark vector.

        Returns:
            {
                "label": "HELLO" | "THANK YOU" | "YES" | "NO" | "HELP" | "UNKNOWN",
                "confidence": float,
                "probabilities": dict[str, float]
            }
        """
        if feature_vector.shape != (FRAME_FEATURE_DIM,):
            raise ValueError(f"Expected shape ({FRAME_FEATURE_DIM},), got {feature_vector.shape}")

        # Check if no hands were detected in the frame (all 126 zeros)
        if np.allclose(feature_vector, 0.0):
            return {
                "label": "UNKNOWN",
                "predicted_label": "NONE",
                "confidence": 0.0,
                "hand_detected": False,
                "probabilities": {lbl: 0.0 for lbl in self.labels},
            }

        sample = feature_vector[np.newaxis, :].astype(np.float32)
        probs = self.model.predict(sample, verbose=0)[0]
        best_idx = int(np.argmax(probs))
        conf = float(probs[best_idx])
        predicted_class = self.labels[best_idx]

        # Enforce threshold: If confidence < 0.7 return UNKNOWN
        if conf < self.confidence_threshold:
            final_label = "UNKNOWN"
        else:
            final_label = predicted_class

        return {
            "label": final_label,
            "predicted_label": predicted_class,
            "confidence": float(round(conf, 4)),
            "hand_detected": True,
            "probabilities": {lbl: float(round(p, 4)) for lbl, p in zip(self.labels, probs)},
        }

    def predict_frame(self, frame_bgr: np.ndarray) -> Dict:
        """Extract landmarks from a camera/video frame and predict sign."""
        features = self.extract_features(frame_bgr)
        return self.predict_vector(features)

    def predict(self, frame_bgr: np.ndarray) -> Tuple[str, float]:
        """Convenience method returning (label, confidence)."""
        res = self.predict_frame(frame_bgr)
        return res["label"], res["confidence"]

    def close(self) -> None:
        if hasattr(self, "landmarker") and self.landmarker:
            self.landmarker.close()

    def __enter__(self) -> SignClassifier:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# Singleton instance for fast API route reuse
_global_classifier: Optional[SignClassifier] = None


def get_classifier() -> SignClassifier:
    global _global_classifier
    if _global_classifier is None:
        _global_classifier = SignClassifier()
    return _global_classifier


def predict_frame(frame_bgr: np.ndarray) -> Tuple[str, float]:
    """Top-level functional API for single-frame inference."""
    classifier = get_classifier()
    return classifier.predict(frame_bgr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sign language inference using model.keras")
    parser.add_argument("--image", type=str, help="Path to input image file")
    parser.add_argument("--video", type=str, help="Path to input video file")
    parser.add_argument("--test", action="store_true", help="Run self-test on dataset samples")
    args = parser.parse_args()

    print("Loading SignClassifier (model.keras, 126-D landmarks)...")
    classifier = SignClassifier()
    print(f"Supported labels: {classifier.labels}")
    print(f"Confidence threshold: {classifier.confidence_threshold}")

    try:
        if args.image:
            img = cv2.imread(args.image)
            if img is None:
                print(f"Error: Could not read image at {args.image}")
                return
            res = classifier.predict_frame(img)
            print(f"\nResult: Label={res['label']}, Confidence={res['confidence']:.4f}")
            print(f"Probabilities: {json.dumps(res['probabilities'], indent=2)}")

        elif args.video:
            cap = cv2.VideoCapture(args.video)
            if not cap.isOpened():
                print(f"Error: Could not read video at {args.video}")
                return
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                if frame_count % 5 == 0:
                    res = classifier.predict_frame(frame)
                    print(f"Frame {frame_count:4d}: Label={res['label']:<10} Conf={res['confidence']:.4f} (Hand={res['hand_detected']})")
            cap.release()

        elif args.test or True:
            # Self-test on real dataset samples
            from backend.ml.config import ROOT_DATASET_PATH, DATASET_PATH
            dpath = ROOT_DATASET_PATH if ROOT_DATASET_PATH.exists() else DATASET_PATH
            if dpath.exists():
                data = np.load(dpath)
                x = data["x"]
                y = data["y"]
                print(f"\nRunning self-test on 10 random samples from {dpath.name}...")
                indices = np.random.RandomState(42).choice(len(x), size=10, replace=False)
                for idx in indices:
                    feat = x[idx]
                    true_label = classifier.labels[y[idx]]
                    res = classifier.predict_vector(feat)
                    print(f"Sample {idx:4d} | GroundTruth: {true_label:<10} | Predicted: {res['label']:<10} | Conf: {res['confidence']:.4f} | Raw: {res['predicted_label']}")
    finally:
        classifier.close()


if __name__ == "__main__":
    main()
