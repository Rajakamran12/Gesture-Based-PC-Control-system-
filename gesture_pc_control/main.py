import os
import math
import time
import json
from typing import Any, Dict, Optional

import cv2

from camera_module import CameraModule
from dashboard_ui import DashboardUI
from feature_engineering import FeatureEngineer
from gesture_classifier import GestureClassifier
from gesture_smoothing import GestureSmoother
from hand_detector import HandDetector
from landmark_extractor import LandmarkExtractor
from pc_controller import PCController


class GesturePCControlApp:
    """Main application that integrates camera, detection, extraction, and dashboard."""

    _ASSUMED_PALM_WIDTH_FT = 0.27
    _ASSUMED_CAMERA_HFOV_DEG = 60.0
    _DEFAULT_MODEL_PATH = "model/keypoint_classifier/gesture_rf_model.pkl"
    _SMOOTH_PRESETS: Dict[str, Dict[str, float | int]] = {
        "aggressive": {
            "window_size": 4,
            "min_confidence": 0.48,
            "min_consensus": 0.52,
            "hold_frames": 1,
            "switch_cooldown_frames": 0,
        },
        "fast": {
            "window_size": 5,
            "min_confidence": 0.50,
            "min_consensus": 0.55,
            "hold_frames": 1,
            "switch_cooldown_frames": 0,
        },
        "balanced": {
            "window_size": 7,
            "min_confidence": 0.55,
            "min_consensus": 0.60,
            "hold_frames": 2,
            "switch_cooldown_frames": 1,
        },
        "stable": {
            "window_size": 9,
            "min_confidence": 0.62,
            "min_consensus": 0.68,
            "hold_frames": 3,
            "switch_cooldown_frames": 2,
        },
        "ultra_stable": {
            "window_size": 11,
            "min_confidence": 0.68,
            "min_consensus": 0.74,
            "hold_frames": 4,
            "switch_cooldown_frames": 3,
        },
    }

    _DEFAULT_GESTURE_THRESHOLDS: Dict[str, Dict[str, float | int]] = {
        "Point": {"min_confidence": 0.58, "min_consensus": 0.62, "hold_frames": 2},
        "Point Left": {"min_confidence": 0.62, "min_consensus": 0.65, "hold_frames": 2},
        "Point Right": {"min_confidence": 0.62, "min_consensus": 0.65, "hold_frames": 2},
        "Two Finger": {"min_confidence": 0.65, "min_consensus": 0.68, "hold_frames": 2},
        "Open Palm": {"min_confidence": 0.52, "min_consensus": 0.58, "hold_frames": 1},
        "Fist": {"min_confidence": 0.54, "min_consensus": 0.60, "hold_frames": 1},
    }

    def __init__(self) -> None:
        auto_start_permissions = None
        if os.getenv("DRIVEFLOW_PERMISSION_PRESET", "0") == "1":
            camera_allowed = os.getenv("DRIVEFLOW_CAMERA_ALLOWED", "0") == "1"
            pc_allowed = os.getenv("DRIVEFLOW_PC_ALLOWED", "0") == "1"
            auto_start_permissions = (camera_allowed, pc_allowed)

        self.camera = CameraModule(device_index=0, width=960, height=540)
        self.detector = HandDetector(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.extractor = LandmarkExtractor()
        self.feature_engineer = FeatureEngineer()

        model_path = os.getenv("GESTURE_MODEL_PATH", self._DEFAULT_MODEL_PATH)
        self.classifier = GestureClassifier(model_path=model_path)
        self._model_path = model_path
        self._model_loaded = self.classifier.model is not None

        smooth_profile = os.getenv("GESTURE_SMOOTH_PROFILE", "fast").lower()
        preset = self._SMOOTH_PRESETS.get(smooth_profile, self._SMOOTH_PRESETS["fast"])
        self._smooth_profile = smooth_profile if smooth_profile in self._SMOOTH_PRESETS else "fast"

        smooth_window = int(os.getenv("GESTURE_SMOOTH_WINDOW", str(preset["window_size"])))
        smooth_min_conf = float(os.getenv("GESTURE_MIN_CONFIDENCE", str(preset["min_confidence"])))
        smooth_min_consensus = float(os.getenv("GESTURE_MIN_CONSENSUS", str(preset["min_consensus"])))
        smooth_hold_frames = int(os.getenv("GESTURE_HOLD_FRAMES", str(preset["hold_frames"])))
        smooth_cooldown = int(os.getenv("GESTURE_SWITCH_COOLDOWN", str(preset["switch_cooldown_frames"])))

        gesture_thresholds = dict(self._DEFAULT_GESTURE_THRESHOLDS)
        thresholds_env = os.getenv("GESTURE_GESTURE_THRESHOLDS", "").strip()
        if thresholds_env:
            try:
                parsed_thresholds = json.loads(thresholds_env)
                if isinstance(parsed_thresholds, dict):
                    for gesture_name, overrides in parsed_thresholds.items():
                        if isinstance(overrides, dict):
                            gesture_thresholds[str(gesture_name)] = overrides
            except Exception:
                pass

        self.smoother = GestureSmoother(
            window_size=smooth_window,
            min_confidence=smooth_min_conf,
            min_consensus=smooth_min_consensus,
            hold_frames=smooth_hold_frames,
            switch_cooldown_frames=smooth_cooldown,
            gesture_thresholds=gesture_thresholds,
        )
        self.controller = PCController(min_action_interval_sec=0.45)
        self.ui = DashboardUI(
            width=960,
            height=540,
            auto_start_permissions=auto_start_permissions,
        )
        self._last_action = "No action"
        self._camera_started = False

    def _set_controls_enabled(self, enabled: bool) -> None:
        self.controller.set_enabled(enabled)

    def _start_after_consent(self, camera_allowed: bool, pc_allowed: bool) -> bool:
        if not camera_allowed:
            return False

        if not self._camera_started:
            if not self.camera.start_camera():
                return False
            self._camera_started = True

        self.controller.set_enabled(pc_allowed)
        self.ui.set_frame_provider(self._process_next_frame)
        return True

    def _build_landmark_debug(
        self,
        landmarks_payload: list[dict[str, Any]],
        features_payload: Optional[Dict[str, Any]],
        classify_payload: Dict[str, Any],
        smooth_payload: Dict[str, Any],
    ) -> str:
        if not landmarks_payload:
            return "No landmarks detected"

        hand = landmarks_payload[0]
        label = hand.get("handedness", "Unknown")
        score = hand.get("score", 0.0)
        distance_ft = hand.get("distance_ft")
        points = hand.get("landmarks", [])[:5]
        coords = [f"({p['x']:.3f}, {p['y']:.3f}, {p['z']:.3f})" for p in points]
        distance_text = f", approx {distance_ft:.2f} ft" if isinstance(distance_ft, (int, float)) else ""
        feature_dim = features_payload.get("feature_dim") if features_payload else 0
        raw_gesture = classify_payload.get("gesture", "None")
        raw_conf = float(classify_payload.get("confidence", 0.0))
        source = classify_payload.get("source", "rule")
        stable_gesture = smooth_payload.get("stable_gesture", "None")
        consensus = float(smooth_payload.get("consensus_ratio", 0.0))

        base = f"{label} ({score:.2f}{distance_text}) first-5: " + ", ".join(coords)
        module_line = (
            f"\nFeatures: {feature_dim} dims | Raw: {raw_gesture} ({raw_conf:.2f}, {source}) "
            f"| Stable: {stable_gesture} | Consensus: {consensus:.2f}"
        )
        return base + module_line

    def _estimate_distance_ft(self, landmarks: list[dict[str, Any]], frame_width: int) -> Optional[float]:
        if len(landmarks) < 18 or frame_width <= 0:
            return None

        index_mcp = landmarks[5]
        pinky_mcp = landmarks[17]
        palm_width_px = math.hypot(
            index_mcp["px"] - pinky_mcp["px"],
            index_mcp["py"] - pinky_mcp["py"],
        )
        if palm_width_px <= 1:
            return None

        focal_length_px = frame_width / (2 * math.tan(math.radians(self._ASSUMED_CAMERA_HFOV_DEG / 2)))
        distance_ft = (self._ASSUMED_PALM_WIDTH_FT * focal_length_px) / palm_width_px
        return max(distance_ft, 0.1)

    def _annotate_hand_distances(self, frame: Any, landmarks_payload: list[dict[str, Any]]) -> Optional[float]:
        if frame is None or not landmarks_payload:
            return None

        frame_width = frame.shape[1]
        primary_distance_ft: Optional[float] = None

        for hand in landmarks_payload:
            landmarks = hand.get("landmarks", [])
            distance_ft = self._estimate_distance_ft(landmarks, frame_width)
            hand["distance_ft"] = distance_ft
            if distance_ft is None or not landmarks:
                continue

            if primary_distance_ft is None or distance_ft < primary_distance_ft:
                primary_distance_ft = distance_ft

            min_x = min(point["px"] for point in landmarks)
            min_y = min(point["py"] for point in landmarks)
            label = f"Distance: {distance_ft:.2f} ft"

            text_origin = (max(10, min_x), max(24, min_y - 12))
            cv2.putText(frame, label, text_origin,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, label, text_origin,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1, cv2.LINE_AA)

        return primary_distance_ft

    def _process_next_frame(self) -> Optional[Dict[str, Any]]:
        start = time.perf_counter()
        frame = self.camera.get_frame()
        if frame is None:
            return {
                "frame": None,
                "gesture": "None",
                "raw_gesture": "None",
                "gesture_confidence": 0.0,
                "landmarks": [],
                "distance_ft": None,
                "feature_dim": 0,
            }

        processed_frame, results = self.detector.process_frame(frame)
        h, w = processed_frame.shape[:2]
        landmarks_payload = self.extractor.extract(results, frame_width=w, frame_height=h)
        primary_distance_ft = self._annotate_hand_distances(processed_frame, landmarks_payload)
        features_payload = self.feature_engineer.extract(landmarks_payload)
        classify_payload = self.classifier.classify(landmarks_payload, features_payload)
        smooth_payload = self.smoother.update(
            str(classify_payload.get("gesture", "None")),
            float(classify_payload.get("confidence", 0.0)),
        )

        stable_gesture = str(smooth_payload.get("stable_gesture", "None"))
        self._last_action = self.controller.handle_gesture(stable_gesture, landmarks_payload)
        landmark_debug = self._build_landmark_debug(
            landmarks_payload,
            features_payload,
            classify_payload,
            smooth_payload,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        return {
            "frame": processed_frame,
            "gesture": stable_gesture,
            "raw_gesture": str(classify_payload.get("gesture", "None")),
            "gesture_confidence": float(classify_payload.get("confidence", 0.0)),
            "classification_source": str(classify_payload.get("source", "rule")),
            "consensus_ratio": float(smooth_payload.get("consensus_ratio", 0.0)),
            "jitter_index": float(smooth_payload.get("jitter_index", 0.0)),
            "smooth_profile": self._smooth_profile,
            "modules_version": "v2",
            "model_loaded": self._model_loaded,
            "model_path": self._model_path,
            "landmarks": landmarks_payload,
            "distance_ft": primary_distance_ft,
            "controls_enabled": self.controller.enabled,
            "action": self._last_action,
            "landmark_debug": landmark_debug,
            "feature_dim": int(features_payload.get("feature_dim", 0)) if features_payload else 0,
            "pipeline_latency_ms": latency_ms,
        }

    def run(self) -> None:
        self.ui.set_start_callback(self._start_after_consent)
        self.ui.set_control_toggle_callback(self._set_controls_enabled)
        try:
            self.ui.start()
        finally:
            self.detector.close()
            self.camera.release_camera()


def main() -> None:
    app = GesturePCControlApp()
    app.run()


if __name__ == "__main__":
    main()
