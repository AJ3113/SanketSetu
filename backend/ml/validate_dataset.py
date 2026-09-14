from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from backend.ml.config import ARTIFACTS_DIR, CLASS_NAMES, DATASET_PATH, LABEL_MAP_PATH, PREPROCESSING_CONFIG_PATH


def validate_dataset(dataset_path: Path = DATASET_PATH, label_map_path: Path = LABEL_MAP_PATH) -> dict:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    if not label_map_path.exists():
        raise FileNotFoundError(f"Label map not found: {label_map_path}")

    dataset = np.load(dataset_path)
    x = dataset["x"]
    y = dataset["y"]
    split = dataset["split"]
    video_id = dataset["video_id"]

    with open(label_map_path, "r", encoding="utf-8") as file:
        raw_labels = json.load(file)
    if isinstance(raw_labels, list):
        label_map = {name: idx for idx, name in enumerate(raw_labels)}
    else:
        label_map = raw_labels
    class_names = list(label_map.keys())

    if set(class_names) != set(CLASS_NAMES):
        missing = [name for name in CLASS_NAMES if name not in class_names]
        extra = [name for name in class_names if name not in CLASS_NAMES]
        raise ValueError(f"Class mismatch. Missing={missing}, extra={extra}")

    if x.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix: (samples, features), got {x.shape}")

    feature_dim = x.shape[1]
    if feature_dim != 126:
        raise ValueError(f"Feature dimension mismatch. Expected 126, found {feature_dim}")

    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Sample count mismatch: x {x.shape[0]} vs y {y.shape[0]}")

    if not np.isfinite(x).all():
        raise ValueError("Dataset contains NaN or infinite values.")

    if y.min() < 0 or y.max() >= len(CLASS_NAMES):
        raise ValueError(f"Y labels out of expected range: {y.min()} to {y.max()}")

    counts_by_class = {cls: int(np.sum(y == label_map[cls])) for cls in CLASS_NAMES}
    if any(count == 0 for count in counts_by_class.values()):
        raise ValueError(f"Missing class data: {counts_by_class}")

    for split_name in ["train", "val", "test"]:
        if split_name not in split:
            raise ValueError(f"Missing {split_name} split in dataset")


    video_to_split = {}
    for sample_id, s in enumerate(split):
        vid = video_id[sample_id]
        prev_split = video_to_split.get(vid)
        if prev_split is not None and prev_split != s:
            raise ValueError(f"Video leakage detected across splits for video {vid}: {prev_split} vs {s}")
        video_to_split[vid] = s

    per_split_videos = {name: set() for name in ["train", "val", "test"]}
    for vid, s in video_to_split.items():
        per_split_videos[s].add(vid)

    imbalance = max(counts_by_class.values()) / min(counts_by_class.values())
    if imbalance > 10:
        raise ValueError(f"Class imbalance too high: {imbalance:.2f}x")

    report = {
        "classes": CLASS_NAMES,
        "shape": list(x.shape),
        "n_samples": int(x.shape[0]),
        "counts_by_class": counts_by_class,
        "splits": {name: int(np.sum(split == name)) for name in ["train", "val", "test"]},
        "videos_per_split": {name: len(per_split_videos[name]) for name in ["train", "val", "test"]},
        "video_overlap": 0,
        "imbalance_ratio": float(imbalance),
        "label_map": label_map,
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS_DIR / "dataset_validation.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, sort_keys=True)
    return report


def main() -> None:
    report = validate_dataset()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
