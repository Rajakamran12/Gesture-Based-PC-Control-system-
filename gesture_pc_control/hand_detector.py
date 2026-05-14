from pathlib import Path
from typing import Optional, Tuple
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np


class HandDetector:
    """MediaPipe hand detector and renderer."""

    _HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17),
    ]

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
    ):
        self._mode = "solutions" if hasattr(mp, "solutions") else "tasks"
        self.hands = None

        if self._mode == "solutions":
            self._mp_hands = mp.solutions.hands
            self._mp_drawing = mp.solutions.drawing_utils
            self._mp_drawing_styles = mp.solutions.drawing_styles
            self.hands = self._mp_hands.Hands(
                static_image_mode=static_image_mode,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        else:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            model_path = self._ensure_task_model()
            base_options = python.BaseOptions(model_asset_path=str(model_path))
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=max_num_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_tracking_confidence,
                min_tracking_confidence=min_tracking_confidence,
                running_mode=vision.RunningMode.IMAGE,
            )
            self.hands = vision.HandLandmarker.create_from_options(options)

    def _ensure_task_model(self) -> Path:
        model_dir = Path(__file__).resolve().parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "hand_landmarker.task"
        if model_path.exists():
            return model_path

        model_url = (
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task"
        )
        urlretrieve(model_url, model_path)
        return model_path

    def _draw_tasks_landmarks(self, image: np.ndarray, result: object) -> np.ndarray:
        output = image.copy()
        if not result or not getattr(result, "hand_landmarks", None):
            return output

        h, w = output.shape[:2]
        for hand_landmarks in result.hand_landmarks:
            points = []
            for lm in hand_landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)
                points.append((x, y))
                cv2.circle(output, (x, y), 3, (0, 255, 0), -1)

            for start_idx, end_idx in self._HAND_CONNECTIONS:
                if start_idx < len(points) and end_idx < len(points):
                    cv2.line(output, points[start_idx], points[end_idx], (0, 200, 255), 2)

        return output

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[object]]:
        if self._mode == "solutions":
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = self.hands.process(rgb_frame)
            rgb_frame.flags.writeable = True

            output = frame.copy()
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self._mp_drawing.draw_landmarks(
                        output,
                        hand_landmarks,
                        self._mp_hands.HAND_CONNECTIONS,
                        self._mp_drawing_styles.get_default_hand_landmarks_style(),
                        self._mp_drawing_styles.get_default_hand_connections_style(),
                    )
            return output, results

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.hands.detect(mp_image)
        output = self._draw_tasks_landmarks(frame, results)
        return output, results

    def close(self) -> None:
        if self.hands is not None:
            self.hands.close()
