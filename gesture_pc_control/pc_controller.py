import time
from typing import Any, Dict, List


class PCController:
    """Optional PC control actions gated by an explicit runtime toggle."""

    def __init__(self, min_action_interval_sec: float = 0.45) -> None:
        self.enabled = False
        self.min_action_interval_sec = min_action_interval_sec
        self._last_action_time = 0.0

        try:
            import pyautogui  # type: ignore

            self.pyautogui = pyautogui
            self.pyautogui.FAILSAFE = True
            self.available = True
        except Exception:
            self.pyautogui = None
            self.available = False

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def handle_gesture(self, gesture: str, hands_payload: List[Dict[str, Any]]) -> str:
        if not self.enabled:
            return "Controls disabled"

        if not self.available:
            return "pyautogui missing (simulation only)"

        now = time.perf_counter()
        if now - self._last_action_time < self.min_action_interval_sec:
            return "Cooldown"

        action_label = self._execute(gesture, hands_payload)
        if action_label != "No action":
            self._last_action_time = now

        return action_label

    def _execute(self, gesture: str, hands_payload: List[Dict[str, Any]]) -> str:
        if gesture == "Open Palm":
            self.pyautogui.press("space")
            return "Play/Pause"

        if gesture == "Fist":
            self.pyautogui.click()
            return "Left Click"

        if gesture == "Two Finger":
            self.pyautogui.click(button="right")
            return "Right Click"

        if gesture == "Point Left":
            self.pyautogui.hotkey("alt", "left")
            return "Back"

        if gesture == "Point Right":
            self.pyautogui.hotkey("alt", "right")
            return "Forward"

        if gesture == "Point":
            return self._move_mouse_with_index(hands_payload)

        return "No action"

    def _move_mouse_with_index(self, hands_payload: List[Dict[str, Any]]) -> str:
        if not hands_payload:
            return "No action"

        landmarks = hands_payload[0].get("landmarks", [])
        if len(landmarks) != 21:
            return "No action"

        screen_w, screen_h = self.pyautogui.size()
        idx = landmarks[8]
        target_x = int(idx["x"] * screen_w)
        target_y = int(idx["y"] * screen_h)

        self.pyautogui.moveTo(target_x, target_y, duration=0.0)
        return "Mouse Move"
