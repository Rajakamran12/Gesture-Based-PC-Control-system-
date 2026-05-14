# Gesture-Based PC Control (Modular Desktop App)

This module provides a modular OpenCV + MediaPipe + Tkinter desktop application for real-time hand tracking.

## Modules

- `camera_module.py` - Webcam acquisition with stable backend/index fallback
- `hand_detector.py` - MediaPipe Hands detection + landmark drawing
- `landmark_extractor.py` - Extracts 21 landmarks into normalized and pixel coordinates
- `feature_engineering.py` - Builds normalized geometric feature vectors from landmarks
- `gesture_classifier.py` - Hybrid classifier (ML model if present, rules as fallback)
- `gesture_smoothing.py` - Temporal smoothing, confidence filtering, and transition stability
- `pc_controller.py` - Optional PC controls (mouse/click/navigation) behind runtime toggle
- `dashboard_ui.py` - Tkinter real-time dashboard (feed, gesture placeholder, FPS)
- `main.py` - Integration runner
- `train_gesture_model.py` - Trains an ML model on `model/keypoint_classifier/keypoint.csv`

## Install (if needed)

```bash
.\venv_mediapipe\Scripts\python.exe -m pip install mediapipe opencv-python pillow pyautogui
```

## Run

From the project root:

```bash
.\venv_mediapipe\Scripts\python.exe .\gesture_pc_control\main.py
```

For this repository's current setup, `.venv` is typically used:

```bash
.\.venv\Scripts\python.exe .\gesture_pc_control\main.py
```

## Train Landmark Model

Train from collected keypoint data and produce evaluation metrics:

```bash
.\.venv\Scripts\python.exe .\gesture_pc_control\train_gesture_model.py
```

Or use:

```bash
.\run_train_gesture_model.bat
```

Outputs:
- `model/keypoint_classifier/gesture_rf_model.pkl`
- `model/keypoint_classifier/gesture_model_metrics.json`

If `gesture_rf_model.pkl` exists, `main.py` loads it automatically.

## Smoothing Tuning

Tune temporal behavior via environment variables:
- `GESTURE_SMOOTH_PROFILE` (`aggressive`, `fast`, `balanced`, `stable`, `ultra_stable`; default `fast`)
- `GESTURE_SMOOTH_WINDOW` (default `7`)
- `GESTURE_MIN_CONFIDENCE` (default `0.55`)
- `GESTURE_MIN_CONSENSUS` (default `0.60`)
- `GESTURE_HOLD_FRAMES` (default `2`)
- `GESTURE_SWITCH_COOLDOWN` (default `1`)

Suggested profile usage:
- `aggressive`: fastest response, lowest stabilization (good for high FPS + steady lighting)
- `fast`: low-latency default
- `balanced`: moderate smoothing
- `stable`: strong smoothing for noisy scenes
- `ultra_stable`: maximum stability for low-light / shaky input

Optional per-gesture overrides (JSON string):
- `GESTURE_GESTURE_THRESHOLDS`

Example:

```bash
set GESTURE_SMOOTH_PROFILE=fast
set GESTURE_GESTURE_THRESHOLDS={"Point":{"min_confidence":0.6,"min_consensus":0.65,"hold_frames":2},"Two Finger":{"min_confidence":0.68,"min_consensus":0.7,"hold_frames":2}}
```

## Notes

- Press `C` in the Tkinter window to toggle PC controls ON/OFF.
- Press `Q` in the Tkinter window to exit.
- Controls are OFF by default for safety.
- If `pyautogui` is not installed, app runs in simulation mode and shows that in Action status.
- If no hand is detected, the app continues streaming and reports `Hands: 0`.
