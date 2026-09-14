from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DATA_DIR = PROJECT_ROOT / "training_data"
BACKEND_DIR = PROJECT_ROOT / "backend"
MODELS_DIR = BACKEND_DIR / "models"
DATASET_DIR = BACKEND_DIR / "ml" / "data"
ARTIFACTS_DIR = BACKEND_DIR / "ml" / "artifacts"
LABEL_MAP_PATH = DATASET_DIR / "label_map.json"
DATASET_PATH = DATASET_DIR / "dataset.npz"
ROOT_DATASET_PATH = PROJECT_ROOT / "dataset.npz"
LABELS_PATH = DATASET_DIR / "labels.json"
ROOT_LABELS_PATH = PROJECT_ROOT / "labels.json"
VALIDATION_REPORT_PATH = ARTIFACTS_DIR / "dataset_validation.json"

# Hackathon prototype 5 supported labels:
CLASS_NAMES = [
    "HELLO",
    "THANK YOU",
    "YES",
    "NO",
    "HELP",
]

# Mapping from training_data folder names to prototype label names
CLASS_DIR_MAP = {
    "hello": "HELLO",
    "thank_you": "THANK YOU",
    "yes": "YES",
    "no": "NO",
    "help": "HELP",
}

# Exact feature representation:
# 21 landmarks x 3 coordinates for left hand (63) + 21 landmarks x 3 coordinates for right hand (63)
# = 126 landmark features per sampled frame
LANDMARKS_PER_HAND = 21
COORDS_PER_LANDMARK = 3
LEFT_RIGHT_HAND_FEATURES = 2 * LANDMARKS_PER_HAND * COORDS_PER_LANDMARK
FRAME_FEATURE_DIM = LEFT_RIGHT_HAND_FEATURES  # 126
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

MODEL_OUTPUT_PATH = MODELS_DIR / "isl_classifier.tflite"
KERAS_MODEL_PATH = ARTIFACTS_DIR / "model.keras"
ROOT_MODEL_PATH = PROJECT_ROOT / "model.keras"
METRICS_PATH = ARTIFACTS_DIR / "test_metrics.json"
CONFUSION_MATRIX_PATH = ARTIFACTS_DIR / "confusion_matrix.npy"
PREPROCESSING_CONFIG_PATH = ARTIFACTS_DIR / "preprocessing_config.json"
PREPROCESSING_JSON_PATH = ARTIFACTS_DIR / "preprocessing.json"
ROOT_PREPROCESSING_JSON_PATH = PROJECT_ROOT / "preprocessing.json"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


