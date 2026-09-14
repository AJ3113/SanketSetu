from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from backend.ml.config import (
    ARTIFACTS_DIR,
    CLASS_NAMES,
    DATASET_DIR,
    DATASET_PATH,
    FRAME_FEATURE_DIM,
    LABELS_PATH,
    LABEL_MAP_PATH,
    PREPROCESSING_CONFIG_PATH,
    ROOT_DATASET_PATH,
    ROOT_LABELS_PATH,
    TRAINING_DATA_DIR,
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT,
)
from backend.ml.preprocessing import (
    ExtractionReport,
    VideoStats,
    build_label_map,
    create_hand_landmarker,
    get_video_files,
    video_to_frame_features,
)

# MediaPipe Tasks API
from mediapipe.tasks.python.vision import RunningMode


def ensure_directories() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def ensure_training_data_exists() -> None:
    if not TRAINING_DATA_DIR.exists():
        raise FileNotFoundError(f"Training data folder does not exist: {TRAINING_DATA_DIR}")


def split_videos_by_class(class_videos: OrderedDict[str, List[Path]]) -> Dict[str, Dict[str, List[Path]]]:
    split_map: Dict[str, Dict[str, List[Path]]] = {}
    for class_name, videos in class_videos.items():
        if not videos:
            split_map[class_name] = {"train": [], "val": [], "test": []}
            continue
        if len(videos) < 3:
            # Cannot do a 3-way split with fewer than 3 videos.
            # Assign all to train.
            split_map[class_name] = {"train": list(videos), "val": [], "test": []}
            continue
        train, temp = train_test_split(videos, train_size=TRAIN_SPLIT, random_state=42, shuffle=True)
        val, test = train_test_split(temp, train_size=VAL_SPLIT / (VAL_SPLIT + TEST_SPLIT), random_state=42, shuffle=True)
        split_map[class_name] = {"train": list(train), "val": list(val), "test": list(test)}
    return split_map


def collect_dataset(
    target_sample_fps: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, ExtractionReport]:
    """Extract single-frame landmark features (126-D) from training videos for the 5 prototype classes.

    Returns:
        (x, y, splits, video_ids, label_map, report)
    """
    ensure_directories()
    ensure_training_data_exists()

    class_videos = get_video_files()
    label_map = build_label_map()
    split_map = split_videos_by_class(class_videos)

    report = ExtractionReport()

    all_features: List[np.ndarray] = []
    all_labels: List[int] = []
    all_splits: List[str] = []
    all_video_ids: List[str] = []

    for class_name in CLASS_NAMES:
        videos = class_videos.get(class_name, [])
        report.videos_per_class[class_name] = len(videos)
        report.total_videos_found += len(videos)
        report.samples_per_class[class_name] = 0

        if not videos:
            report.missing_classes.append(class_name)
            continue

        for video_path in videos:
            print(f"  Processing: {class_name}/{video_path.name} ...", end=" ", flush=True)
            t0 = time.time()

            # Create a fresh landmarker per video (IMAGE mode)
            landmarker = create_hand_landmarker(
                running_mode=RunningMode.IMAGE,
                num_hands=2,
            )

            try:
                frame_vectors, vstats = video_to_frame_features(
                    video_path,
                    landmarker=landmarker,
                    target_sample_fps=target_sample_fps,
                )
            finally:
                landmarker.close()

            vstats.class_name = class_name
            elapsed = round(time.time() - t0, 1)

            # Aggregate stats
            report.total_frames_read += vstats.total_frames
            report.total_frames_sampled += vstats.sampled_frames
            report.total_valid_frames += vstats.valid_frames
            report.total_no_hand_frames += vstats.no_hand_frames
            report.total_error_frames += vstats.error_frames

            if vstats.skipped:
                report.total_videos_skipped += 1
                report.skipped_videos.append({
                    "path": str(video_path.name),
                    "class": class_name,
                    "reason": vstats.skip_reason,
                })
                print(f"SKIPPED ({vstats.skip_reason}) [{elapsed}s]")
            else:
                report.total_videos_processed += 1
                report.total_sequences += len(frame_vectors)
                report.total_samples += len(frame_vectors)
                report.samples_per_class[class_name] += len(frame_vectors)

                # Determine which split this video belongs to
                selected_split = "train"
                for split_name in ("train", "val", "test"):
                    if video_path in split_map.get(class_name, {}).get(split_name, []):
                        selected_split = split_name
                        break

                video_id = f"{class_name}:{video_path.stem}"
                for feat in frame_vectors:
                    all_features.append(feat)
                    all_labels.append(label_map[class_name])
                    all_splits.append(selected_split)
                    all_video_ids.append(video_id)

                print(f"OK  frames={vstats.valid_frames}  samples={len(frame_vectors)}  split={selected_split}  [{elapsed}s]")

            report.per_video.append({
                "video": video_path.name,
                "class": class_name,
                "total_frames": vstats.total_frames,
                "sampled_frames": vstats.sampled_frames,
                "valid_frames": vstats.valid_frames,
                "no_hand_frames": vstats.no_hand_frames,
                "error_frames": vstats.error_frames,
                "samples": len(frame_vectors) if not vstats.skipped else 0,
                "fps": vstats.fps,
                "duration_sec": vstats.duration_sec,
                "skipped": vstats.skipped,
                "skip_reason": vstats.skip_reason,
            })

    if not all_features:
        raise ValueError("No valid frame landmark vectors were created from the training_data folder.")

    # Build arrays: (samples, 126)
    x = np.stack(all_features).astype(np.float32)
    y = np.asarray(all_labels, dtype=np.int32)
    splits = np.asarray(all_splits)
    video_ids = np.asarray(all_video_ids)

    # Save labels.json: ["HELLO", "THANK YOU", "YES", "NO", "HELP"]
    labels_list = list(CLASS_NAMES)
    for p in [LABELS_PATH, ROOT_LABELS_PATH]:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(labels_list, f, indent=2)

    # Save label_map.json for dictionary lookup compatibility
    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, sort_keys=True)

    # Save preprocessing config
    config = {
        "feature_dim": FRAME_FEATURE_DIM,
        "input_shape": [FRAME_FEATURE_DIM],
        "landmarks_per_hand": 21,
        "coords_per_landmark": 3,
        "num_hands": 2,
        "hand_ordering": ["left", "right"],
        "normalization": "wrist_index_pinky_center_max_scale",
        "missing_hand_fill": "zeros",
        "target_sample_fps": target_sample_fps,
        "classes": CLASS_NAMES,
        "num_classes": len(CLASS_NAMES),
        "labels": labels_list,
        "label_map": label_map,
    }
    with open(PREPROCESSING_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Save dataset.npz to both DATASET_PATH and ROOT_DATASET_PATH
    for p in [DATASET_PATH, ROOT_DATASET_PATH]:
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            x=x,
            y=y,
            split=splits,
            video_id=video_ids,
            X=x,
            Y=y,
            data=x,
            labels=y,
        )

    # Save extraction report
    report_path = ARTIFACTS_DIR / "extraction_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)

    return x, y, splits, video_ids, label_map, report


def print_summary(x: np.ndarray, y: np.ndarray, splits: np.ndarray, report: ExtractionReport) -> None:
    """Print a human-readable extraction summary."""
    print("\n" + "=" * 70)
    print("  STAGE 1 - EXTRACTION REPORT (PROTOTYPE SINGLE-FRAME 126-D)")
    print("=" * 70)
    print(f"  Dataset saved to  : {DATASET_PATH} (and {ROOT_DATASET_PATH})")
    print(f"  Labels saved to   : {LABELS_PATH} (and {ROOT_LABELS_PATH})")
    print(f"  Label map saved to: {LABEL_MAP_PATH}")
    print(f"  Config saved to   : {PREPROCESSING_CONFIG_PATH}")
    print(f"  Report saved to   : {ARTIFACTS_DIR / 'extraction_report.json'}")
    print("-" * 70)
    print(f"  Videos found      : {report.total_videos_found}")
    print(f"  Videos processed  : {report.total_videos_processed}")
    print(f"  Videos skipped    : {report.total_videos_skipped}")
    print("-" * 70)
    print(f"  Total frames read : {report.total_frames_read}")
    print(f"  Frames sampled    : {report.total_frames_sampled}")
    print(f"  Valid frames      : {report.total_valid_frames}")
    print(f"  No-hand frames    : {report.total_no_hand_frames}")
    print(f"  Error frames      : {report.total_error_frames}")
    print("-" * 70)
    print(f"  Total samples     : {x.shape[0]}")
    print(f"  Dataset shape (x) : {x.shape}  ->  (samples, features={FRAME_FEATURE_DIM})")
    print(f"  Labels shape  (y) : {y.shape}")
    print("-" * 70)
    print("  Samples per class:")
    for cls in CLASS_NAMES:
        count = report.samples_per_class.get(cls, 0)
        vid_count = report.videos_per_class.get(cls, 0)
        marker = " << MISSING" if count == 0 else ""
        print(f"    {cls:20s}: {count:6d} samples  ({vid_count} videos){marker}")
    print("-" * 70)
    if report.missing_classes:
        print(f"  WARNING: Missing classes: {report.missing_classes}")
    if report.skipped_videos:
        print(f"  WARNING: Skipped videos ({len(report.skipped_videos)}):")
        for sv in report.skipped_videos:
            print(f"       {sv['class']}/{sv['path']} -- {sv['reason']}")
    print("-" * 70)
    split_counts = {s: int(np.sum(splits == s)) for s in ["train", "val", "test"]}
    print(f"  Split distribution: train={split_counts['train']}  val={split_counts['val']}  test={split_counts['test']}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract 126-D single-frame landmark features for prototype labels")
    parser.add_argument("--target-fps", type=float, default=10.0, help="Target sample FPS (default: 10)")
    args = parser.parse_args()

    print("SanketSetu - Single-Frame 126-D Landmark Extraction")
    print(f"Training data dir: {TRAINING_DATA_DIR}")
    print(f"Feature dim      : {FRAME_FEATURE_DIM}")
    print(f"Target sample FPS: {args.target_fps}")
    print(f"Prototype classes: {CLASS_NAMES}")
    print()

    x, y, splits, video_ids, label_map, report = collect_dataset(
        target_sample_fps=args.target_fps,
    )
    print_summary(x, y, splits, report)


if __name__ == "__main__":
    main()
