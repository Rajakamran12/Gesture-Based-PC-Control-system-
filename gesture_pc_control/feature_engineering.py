import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class FeatureEngineer:
    """Build compact, normalized geometric features from 21 hand landmarks."""

    # Distances that capture finger spread, palm openness, and pointing geometry.
    _DISTANCE_PAIRS: Tuple[Tuple[int, int], ...] = (
        (0, 4), (0, 8), (0, 12), (0, 16), (0, 20),
        (4, 8), (8, 12), (12, 16), (16, 20), (5, 17),
    )

    # Joint-angle triplets: angle ABC where B is the vertex.
    _ANGLE_TRIPLETS: Tuple[Tuple[int, int, int], ...] = (
        (1, 2, 3), (2, 3, 4),
        (5, 6, 7), (6, 7, 8),
        (9, 10, 11), (10, 11, 12),
        (13, 14, 15), (14, 15, 16),
        (17, 18, 19), (18, 19, 20),
    )

    _DEPTH_IDS: Tuple[int, ...] = (0, 4, 8, 12, 16, 20)

    def extract(self, hands_payload: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not hands_payload:
            return None

        hand = hands_payload[0]
        landmarks = hand.get("landmarks", [])
        if len(landmarks) != 21:
            return None

        coords = np.array(
            [[float(p["x"]), float(p["y"]), float(p.get("z", 0.0))] for p in landmarks],
            dtype=np.float32,
        )

        normalized, scale = self._normalize(coords)
        xy_features = normalized[:, :2].reshape(-1)
        depth_features = normalized[list(self._DEPTH_IDS), 2]
        distance_features = self._distance_features(normalized)
        angle_features = self._angle_features(normalized)

        handedness = hand.get("handedness", "Unknown").lower()
        handedness_feature = 1.0 if handedness == "right" else -1.0 if handedness == "left" else 0.0
        tracking_score = float(hand.get("score", 0.0))

        feature_vector = np.concatenate(
            [
                xy_features,
                depth_features,
                distance_features,
                angle_features,
                np.array([handedness_feature, tracking_score], dtype=np.float32),
            ]
        )

        # Keep a dedicated ML vector compatible with keypoint.csv (42 values: x/y for 21 points).
        ml_feature_vector = xy_features.astype(np.float32)

        return {
            "feature_vector": feature_vector.astype(np.float32).tolist(),
            "ml_feature_vector": ml_feature_vector.tolist(),
            "feature_dim": int(feature_vector.shape[0]),
            "ml_feature_dim": int(ml_feature_vector.shape[0]),
            "normalization_scale": float(scale),
            "distance_features": distance_features.astype(np.float32).tolist(),
            "angle_features": angle_features.astype(np.float32).tolist(),
        }

    def _normalize(self, coords: np.ndarray) -> Tuple[np.ndarray, float]:
        wrist = coords[0]
        centered = coords - wrist

        palm_width = float(np.linalg.norm(centered[5, :2] - centered[17, :2]))
        fingertip_span = float(np.max(np.linalg.norm(centered[[4, 8, 12, 16, 20], :2], axis=1)))
        scale = max(palm_width, fingertip_span, 1e-6)
        normalized = centered / scale

        return normalized, scale

    def _distance_features(self, normalized_coords: np.ndarray) -> np.ndarray:
        values = []
        for a, b in self._DISTANCE_PAIRS:
            values.append(float(np.linalg.norm(normalized_coords[a] - normalized_coords[b])))
        return np.array(values, dtype=np.float32)

    def _angle_features(self, normalized_coords: np.ndarray) -> np.ndarray:
        values = []
        for a, b, c in self._ANGLE_TRIPLETS:
            values.append(self._joint_angle(normalized_coords[a], normalized_coords[b], normalized_coords[c]))
        return np.array(values, dtype=np.float32)

    def _joint_angle(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        ab = a - b
        cb = c - b

        denom = float(np.linalg.norm(ab) * np.linalg.norm(cb))
        if denom < 1e-8:
            return 0.0

        cos_value = float(np.dot(ab, cb) / denom)
        cos_value = max(-1.0, min(1.0, cos_value))
        return float(math.degrees(math.acos(cos_value)))
