# ISL ML pipeline

This directory contains the minimal machine learning pipeline for the 20-sign SanketSetu ISL MVP.

## Feature design

The shared input representation is intentionally fixed and used in training, validation, and webcam testing:

- left hand: 21 landmarks × 3 coordinates = 63 features
- right hand: 21 landmarks × 3 coordinates = 63 features
- total landmark features = 126
- presence flags: left_present, right_present = 2 features
- per-frame input = 128 features
- temporal sequence length = 8
- model input shape = (8, 128)

This ensures a consistent representation even when one hand is missing, by filling absent landmarks with zeros and setting the appropriate presence flag to 0.

## Folder structure

```text
backend/
  ml/
    README.md
    config.py
    preprocessing.py
    build_dataset.py
    validate_dataset.py
    train_model.py
    export_tflite.py
    webcam_test.py
    data/
      label_map.json
      isl_sequences.npz
    artifacts/
      dataset_validation.json
      preprocessing_config.json
      test_metrics.json
      confusion_matrix.npy
      isl_classifier.keras
models/
  isl_classifier.tflite
```

## Dataset preparation

Place your manually recorded sign videos under:

```text
training_data/
  hello/
  thank_you/
  please/
  yes/
  no/
  good_morning/
  good_night/
  i/
  name/
  water/
  food/
  bathroom/
  hospital/
  doctor/
  emergency/
  stop/
  help/
  need/
  want/
  sorry/
```

Each class folder should contain video files recorded for that sign.

## Commands

### Build dataset

```bash
cd /path/to/sanket-setu
C:/Users/User/AppData/Local/Microsoft/WindowsApps/python3.13.exe -m backend.ml.build_dataset
```

### Validate dataset

```bash
cd /path/to/sanket-setu
C:/Users/User/AppData/Local/Microsoft/WindowsApps/python3.13.exe -m backend.ml.validate_dataset
```

### Train model

```bash
cd /path/to/sanket-setu
C:/Users/User/AppData/Local/Microsoft/WindowsApps/python3.13.exe -m backend.ml.train_model
```

### Export TFLite

```bash
cd /path/to/sanket-setu
C:/Users/User/AppData/Local/Microsoft/WindowsApps/python3.13.exe -m backend.ml.export_tflite
```

### Webcam evaluation test

```bash
cd /path/to/sanket-setu
C:/Users/User/AppData/Local/Microsoft/WindowsApps/python3.13.exe -m backend.ml.webcam_test
```

## Assumptions and limitations

- This is a lightweight MVP for 20 signs, not a general-purpose ISL recognizer.
- Dynamic signs are harder than static ones and are naturally more sensitive to sequence quality and capture conditions.
- The model is trained on the user-provided dataset and should be treated as signer- and environment-dependent until more diverse data is collected.
- This stage intentionally does not integrate the model into the frontend or FastAPI route yet.
