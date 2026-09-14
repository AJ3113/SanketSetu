from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import mediapipe as mp

from backend.ml.config import (
    ARTIFACTS_DIR,
    CLASS_DIR_MAP,
    CLASS_NAMES,
    FRAME_FEATURE_DIM,
    TRAINING_DATA_DIR,
    VIDEO_EXTENSIONS,
)

# MediaPipe Tasks API (mediapipe >= 1.0)
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

# Path to the bundled hand_landmarker.task model
HAND_LANDMARKER_MODEL_PATH = ARTIFACTS_DIR / "hand_landmarker.task"


# ---------------------------------------------------------------------------
# Data classes for extraction statistics
# ---------------------------------------------------------------------------

@dataclass
class VideoStats:
    """Statistics for a single video extraction."""
    video_path: str = ""
    class_name: str = ""
    total_frames: int = 0
    sampled_frames: int = 0
    valid_frames: int = 0
    no_hand_frames: int = 0
    error_frames: int = 0
    sequences_created: int = 0
    fps: float = 0.0
    duration_sec: float = 0.0
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class ExtractionReport:
    """Aggregate statistics for the full extraction run."""
    total_videos_found: int = 0
    total_videos_processed: int = 0
    total_videos_skipped: int = 0
    total_frames_read: int = 0
    total_frames_sampled: int = 0
    total_valid_frames: int = 0
    total_no_hand_frames: int = 0
    total_error_frames: int = 0
    total_sequences: int = 0
    total_samples: int = 0
    samples_per_class: Dict[str, int] = field(default_factory=dict)
    videos_per_class: Dict[str, int] = field(default_factory=dict)
    missing_classes: List[str] = field(default_factory=list)
    feature_dim: int = FRAME_FEATURE_DIM
    sequence_length: int = 1
    skipped_videos: List[Dict[str, str]] = field(default_factory=list)
    per_video: List[Dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_videos_found": self.total_videos_found,
            "total_videos_processed": self.total_videos_processed,
            "total_videos_skipped": self.total_videos_skipped,
            "total_frames_read": self.total_frames_read,
            "total_frames_sampled": self.total_frames_sampled,
            "total_valid_frames": self.total_valid_frames,
            "total_no_hand_frames": self.total_no_hand_frames,
            "total_error_frames": self.total_error_frames,
            "total_sequences": self.total_sequences,
            "total_samples": self.total_samples,
            "samples_per_class": self.samples_per_class,
            "videos_per_class": self.videos_per_class,
            "missing_classes": self.missing_classes,
            "feature_dim": self.feature_dim,
            "sequence_length": self.sequence_length,
            "skipped_videos": self.skipped_videos,
            "per_video": self.per_video,
        }


# ---------------------------------------------------------------------------
# Normalization (shared between training and inference)
# ---------------------------------------------------------------------------

def normalize_class_name(class_name: str) -> str:
    return class_name.strip().lower().replace(" ", "_")


def normalize_hand_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Normalize a single hand landmark array to a consistent scale and center."""
    if landmarks.shape != (21, 3):
        return np.zeros((21, 3), dtype=np.float32)

    wrist = landmarks[0]
    index_mcp = landmarks[5]
    pinky_mcp = landmarks[17]
    center = np.mean(np.vstack([wrist, index_mcp, pinky_mcp]), axis=0)
    centered = landmarks - center
    scale = float(np.linalg.norm(centered, axis=1).max())
    if scale < 1e-6:
        scale = 1.0
    normalized = centered / scale
    return normalized.astype(np.float32)


def make_zero_hand() -> np.ndarray:
    return np.zeros((21, 3), dtype=np.float32)


# ---------------------------------------------------------------------------
# HandLandmarker lifecycle helpers
# ---------------------------------------------------------------------------

def create_hand_landmarker(
    running_mode: RunningMode = RunningMode.IMAGE,
    num_hands: int = 2,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    result_callback=None,
) -> HandLandmarker:
    """Create a ``HandLandmarker`` using the Tasks API.

    The caller is responsible for closing the returned object.
    """
    if not HAND_LANDMARKER_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"HandLandmarker model not found at {HAND_LANDMARKER_MODEL_PATH}. "
            "Download it from https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        )

    opts = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_LANDMARKER_MODEL_PATH)),
        running_mode=running_mode,
        num_hands=num_hands,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        result_callback=result_callback,
    )
    return HandLandmarker.create_from_options(opts)


# ---------------------------------------------------------------------------
# Frame-level landmark extraction (shared between training and inference)
# ---------------------------------------------------------------------------

def extract_landmarks_from_result(result) -> np.ndarray:
    """Convert a ``HandLandmarkerResult`` to a 126-dim feature vector.

    Layout: left_hand(63) | right_hand(63)
    21 landmarks * 3 coords for left hand + 21 landmarks * 3 coords for right hand.
    """
    left_hand = make_zero_hand()
    right_hand = make_zero_hand()
    left_present = 0.0
    right_present = 0.0

    if result.hand_landmarks and result.handedness:
        for hand_lms, hand_cls in zip(result.hand_landmarks, result.handedness):
            landmarks = np.array(
                [(lm.x, lm.y, lm.z) for lm in hand_lms],
                dtype=np.float32,
            )
            label = hand_cls[0].category_name  # "Left" or "Right"
            if label == "Left" and left_present == 0.0:
                left_hand = normalize_hand_landmarks(landmarks)
                left_present = 1.0
            elif label == "Right" and right_present == 0.0:
                right_hand = normalize_hand_landmarks(landmarks)
                right_present = 1.0

    features = np.concatenate(
        [
            left_hand.reshape(-1),
            right_hand.reshape(-1),
        ]
    ).astype(np.float32)

    if features.shape[0] != FRAME_FEATURE_DIM:
        raise ValueError(f"Expected {FRAME_FEATURE_DIM} features, got {features.shape[0]}")

    return features


def extract_landmarks_from_frame(frame_bgr: np.ndarray, landmarker: HandLandmarker) -> np.ndarray:
    """Return a 126-dim feature vector for a single frame (IMAGE mode)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    return extract_landmarks_from_result(result)


def extract_landmarks_from_video_frame(
    frame_bgr: np.ndarray,
    landmarker: HandLandmarker,
    timestamp_ms: int,
) -> np.ndarray:
    """Return a 126-dim feature vector for a video frame (VIDEO mode)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    return extract_landmarks_from_result(result)


# ---------------------------------------------------------------------------
# Video discovery
# ---------------------------------------------------------------------------

def get_video_files() -> OrderedDict[str, List[Path]]:
    """Return ordered mapping of class name to sorted video files for the prototype labels."""
    class_map: OrderedDict[str, List[Path]] = OrderedDict((class_name, []) for class_name in CLASS_NAMES)
    if not TRAINING_DATA_DIR.exists():
        return class_map

    for class_dir in sorted(TRAINING_DATA_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        normalized_folder = normalize_class_name(class_dir.name)
        # Check mapping from folder name (e.g. 'hello') to prototype label (e.g. 'HELLO')
        label_name = CLASS_DIR_MAP.get(normalized_folder)
        if label_name is None:
            # Check direct match with CLASS_NAMES
            for c in CLASS_NAMES:
                if normalize_class_name(c) == normalized_folder or c == class_dir.name.upper():
                    label_name = c
                    break
        if label_name not in class_map:
            continue
        videos = [
            p for p in sorted(class_dir.iterdir())
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        ]
        class_map[label_name].extend(videos)
    return class_map


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------

def compute_sample_stride(total_frames: int, fps: float, target_fps: float = 10.0) -> int:
    """Compute frame sampling stride to reduce near-duplicate frames.

    Aims to sample at approximately target_fps (default 10) from the original
    video FPS.  Returns a stride >= 1.  For videos already at or below the
    target FPS the stride is 1 (keep every frame).
    """
    if fps <= 0 or fps <= target_fps:
        return 1
    stride = max(1, int(round(fps / target_fps)))
    return stride


def sample_frame_indices(total_frames: int, stride: int) -> List[int]:
    """Return the list of frame indices to actually read, given a stride."""
    return list(range(0, total_frames, stride))


# ---------------------------------------------------------------------------
# Video → frame features (single-frame 126-D vectors with sampling and stats)
# ---------------------------------------------------------------------------

def video_to_frame_features(
    video_path: Path,
    landmarker: Optional[HandLandmarker] = None,
    target_sample_fps: float = 10.0,
) -> Tuple[List[np.ndarray], VideoStats]:
    """Extract 126-D landmark vectors from a video with stride-based sampling.

    Args:
        video_path: Path to the video file.
        landmarker: An already-initialised ``HandLandmarker`` instance (IMAGE mode).
            If *None* a temporary one in IMAGE mode is created and closed after.
        target_sample_fps: Desired sample rate in frames per second.

    Returns:
        A tuple of (list of 126-D frame feature arrays, VideoStats).
    """
    stats = VideoStats(video_path=str(video_path))

    if not video_path.exists():
        stats.skipped = True
        stats.skip_reason = "File not found"
        return [], stats

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        stats.skipped = True
        stats.skip_reason = "Failed to open video"
        return [], stats

    owns_model = landmarker is None
    if owns_model:
        landmarker = create_hand_landmarker(running_mode=RunningMode.IMAGE, num_hands=2)

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        stats.total_frames = total_frames
        stats.fps = fps
        stats.duration_sec = round(total_frames / fps, 2) if fps > 0 else 0.0

        stride = compute_sample_stride(total_frames, fps, target_sample_fps)
        sample_indices = set(sample_frame_indices(total_frames, stride))
        stats.sampled_frames = len(sample_indices)

        frame_features: List[np.ndarray] = []
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx not in sample_indices:
                frame_idx += 1
                continue

            try:
                feature = extract_landmarks_from_frame(frame, landmarker)
                frame_features.append(feature)
                stats.valid_frames += 1
                if np.allclose(feature, 0.0):
                    stats.no_hand_frames += 1
            except Exception:
                stats.error_frames += 1

            frame_idx += 1

        cap.release()

        if not frame_features:
            stats.skipped = True
            stats.skip_reason = "No frames sampled"
            return [], stats

        stats.sequences_created = len(frame_features)
        return frame_features, stats

    finally:
        if owns_model:
            landmarker.close()


def video_to_feature_sequences(
    video_path: Path,
    landmarker: Optional[HandLandmarker] = None,
    target_sample_fps: float = 10.0,
) -> Tuple[List[np.ndarray], VideoStats]:
    """Alias for video_to_frame_features (single frame feature extraction)."""
    return video_to_frame_features(video_path, landmarker, target_sample_fps)



# ---------------------------------------------------------------------------
# Label utilities
# ---------------------------------------------------------------------------

def build_label_map() -> Dict[str, int]:
    return {label: idx for idx, label in enumerate(CLASS_NAMES)}


def class_names_to_indices(class_names: Sequence[str]) -> List[int]:
    label_map = build_label_map()
    return [label_map[name] for name in class_names]


def validate_labels(class_names: Iterable[str]) -> None:
    unknown = [name for name in class_names if name not in CLASS_NAMES]
    if unknown:
        raise ValueError(f"Unknown class names found: {unknown}")
