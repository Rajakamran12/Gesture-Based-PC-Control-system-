import time
from collections import deque
from typing import Deque, Dict, Optional, Tuple


class GestureSmoother:
    """Stabilize per-frame predictions into low-jitter gesture states."""

    def __init__(
        self,
        window_size: int = 7,
        min_confidence: float = 0.55,
        min_consensus: float = 0.60,
        hold_frames: int = 2,
        switch_cooldown_frames: int = 1,
        gesture_thresholds: Optional[Dict[str, Dict[str, float | int]]] = None,
    ) -> None:
        self.window_size = max(3, window_size)
        self.min_confidence = min_confidence
        self.min_consensus = min_consensus
        self.hold_frames = max(1, hold_frames)
        self.switch_cooldown_frames = max(0, switch_cooldown_frames)
        self.gesture_thresholds = gesture_thresholds or {}

        self._history: Deque[Tuple[str, float, float]] = deque(maxlen=self.window_size)
        self._stable_gesture = "None"
        self._pending_gesture = "None"
        self._pending_count = 0
        self._cooldown_frames = 0
        self._last_update_time = time.perf_counter()

    def update(self, raw_gesture: str, raw_confidence: float) -> Dict[str, float | str | bool]:
        now = time.perf_counter()
        frame_delta_ms = max(0.0, (now - self._last_update_time) * 1000.0)
        self._last_update_time = now

        thresholds = self._thresholds_for(raw_gesture)
        min_conf = float(thresholds["min_confidence"])
        min_consensus = float(thresholds["min_consensus"])
        hold_frames = int(thresholds["hold_frames"])

        filtered_gesture = raw_gesture if raw_confidence >= min_conf else "None"
        filtered_confidence = raw_confidence if raw_confidence >= min_conf else 0.0
        self._history.append((filtered_gesture, filtered_confidence, now))

        candidate, consensus_ratio = self._best_consensus_label()
        candidate_thresholds = self._thresholds_for(candidate)
        candidate_min_consensus = float(candidate_thresholds["min_consensus"])
        candidate_hold_frames = int(candidate_thresholds["hold_frames"])

        if consensus_ratio < max(min_consensus, candidate_min_consensus):
            candidate = self._stable_gesture

        changed = self._stabilize_transition(candidate, hold_frames=max(hold_frames, candidate_hold_frames))
        jitter_index = self._jitter_index()

        return {
            "raw_gesture": raw_gesture,
            "filtered_gesture": filtered_gesture,
            "stable_gesture": self._stable_gesture,
            "raw_confidence": float(raw_confidence),
            "filtered_confidence": float(filtered_confidence),
            "consensus_ratio": float(consensus_ratio),
            "jitter_index": float(jitter_index),
            "changed": changed,
            "frame_delta_ms": float(frame_delta_ms),
            "min_confidence_used": float(min_conf),
            "min_consensus_used": float(max(min_consensus, candidate_min_consensus)),
            "hold_frames_used": int(max(hold_frames, candidate_hold_frames)),
        }

    def _thresholds_for(self, gesture: str) -> Dict[str, float | int]:
        defaults: Dict[str, float | int] = {
            "min_confidence": self.min_confidence,
            "min_consensus": self.min_consensus,
            "hold_frames": self.hold_frames,
        }
        gesture_overrides = self.gesture_thresholds.get(gesture, {})
        thresholds = {
            "min_confidence": float(gesture_overrides.get("min_confidence", defaults["min_confidence"])),
            "min_consensus": float(gesture_overrides.get("min_consensus", defaults["min_consensus"])),
            "hold_frames": int(gesture_overrides.get("hold_frames", defaults["hold_frames"])),
        }
        thresholds["hold_frames"] = max(1, int(thresholds["hold_frames"]))
        return thresholds

    def _best_consensus_label(self) -> Tuple[str, float]:
        if not self._history:
            return "None", 0.0

        weighted_votes: Dict[str, float] = {}
        total = 0.0
        hist_list = list(self._history)
        size = len(hist_list)

        for idx, (label, confidence, _) in enumerate(hist_list):
            recency_weight = 1.0 + (idx / max(1, size - 1)) * 0.5
            vote = max(confidence, 0.01) * recency_weight
            weighted_votes[label] = weighted_votes.get(label, 0.0) + vote
            total += vote

        if total <= 0.0:
            return "None", 0.0

        best_label = max(weighted_votes.items(), key=lambda item: item[1])[0]
        best_ratio = weighted_votes[best_label] / total
        return best_label, best_ratio

    def _stabilize_transition(self, candidate: str, hold_frames: int) -> bool:
        if candidate == self._stable_gesture:
            self._pending_gesture = candidate
            self._pending_count = 0
            if self._cooldown_frames > 0:
                self._cooldown_frames -= 1
            return False

        if self._cooldown_frames > 0:
            self._cooldown_frames -= 1
            return False

        if candidate != self._pending_gesture:
            self._pending_gesture = candidate
            self._pending_count = 1
            return False

        self._pending_count += 1
        if self._pending_count < hold_frames:
            return False

        self._stable_gesture = candidate
        self._pending_count = 0
        self._cooldown_frames = self.switch_cooldown_frames
        return True

    def _jitter_index(self) -> float:
        if not self._history:
            return 0.0

        labels = [label for label, _, _ in self._history]
        unique_count = len(set(labels))
        return unique_count / len(labels)
