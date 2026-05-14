import platform
from typing import Optional

import cv2
import numpy as np


class CameraModule:
    """Handles webcam acquisition with backend/index fallbacks for stability."""

    def __init__(self, device_index: int = 0, width: int = 960, height: int = 540):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None

    def start_camera(self) -> bool:
        indexes = [self.device_index] + [idx for idx in [0, 1, 2] if idx != self.device_index]
        backend_options = [None]

        if platform.system().lower() == "windows":
            backend_options = [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]

        for cam_index in indexes:
            for backend in backend_options:
                cap = cv2.VideoCapture(cam_index) if backend is None else cv2.VideoCapture(cam_index, backend)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    self.cap = cap
                    return True
                cap.release()

        self.cap = None
        return False

    def get_frame(self) -> Optional[np.ndarray]:
        if self.cap is None or not self.cap.isOpened():
            return None

        ok, frame = self.cap.read()
        if not ok:
            return None

        return frame

    def release_camera(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
