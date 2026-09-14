from __future__ import annotations

import time
import cv2

from backend.ml.inference import SignClassifier


def main() -> None:
    print("Initializing SignClassifier with model.keras and MediaPipe...")
    classifier = SignClassifier(confidence_threshold=0.7)
    print(f"Supported labels: {classifier.labels}")
    print("Press 'q' or 'ESC' to exit webcam test.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Warning: Could not open default webcam (index 0).")
        classifier.close()
        return

    fps = 0.0
    last_time = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            # Run inference on the current frame
            res = classifier.predict_frame(frame)
            label = res["label"]
            confidence = res["confidence"]
            hand_detected = res["hand_detected"]

            # Visual styling based on result
            if not hand_detected:
                status_text = "No hand detected"
                color = (128, 128, 128)  # Gray
            elif label == "UNKNOWN":
                status_text = f"Sign unclear (conf: {confidence:.2f} < 0.7)"
                color = (0, 165, 255)  # Orange
            else:
                status_text = f"{label} ({confidence * 100:.1f}%)"
                color = (0, 255, 0)  # Green

            # Overlay on camera feed
            cv2.putText(frame, f"Prediction: {label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
            cv2.putText(frame, f"Status: {status_text}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Confidence: {confidence:.4f}", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

            cv2.imshow("SanketSetu ISL Real-time Inference Test", frame)

            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

            current_time = time.time()
            delta = max(current_time - last_time, 1e-6)
            fps = 1.0 / delta
            last_time = current_time

    finally:
        cap.release()
        cv2.destroyAllWindows()
        classifier.close()


if __name__ == "__main__":
    main()
