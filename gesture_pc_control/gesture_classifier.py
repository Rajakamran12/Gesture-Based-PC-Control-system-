import os
import pickle
from typing import Any, Dict, List, Optional


class GestureClassifier:
    """Gesture classifier with optional ML backend and rule-based fallback."""

    TIP_IDS = [4, 8, 12, 16, 20]
    PIP_IDS = [3, 6, 10, 14, 18]

    LABELS = ["None", "Fist", "Open Palm", "Two Finger", "Point Left", "Point Right", "Point", "Thumb"]

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path = model_path
        self.model = self._load_model(model_path)

    def classify(self, hands_payload: List[Dict[str, Any]], features_payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not hands_payload:
            return {
                "gesture": "None",
                "confidence": 0.0,
                "source": "rule",
                "scores": {},
            }

        if self.model is not None and features_payload is not None:
            model_result = self._classify_with_model(features_payload)
            if model_result is not None:
                return model_result

        rule_result = self._classify_with_rules(hands_payload)
        return {
            "gesture": rule_result["gesture"],
            "confidence": float(rule_result["confidence"]),
            "source": "rule",
            "scores": rule_result.get("scores", {}),
        }

    def _load_model(self, model_path: Optional[str]) -> Optional[Any]:
        if not model_path:
            return None

        if not os.path.exists(model_path):
            return None

        try:
            with open(model_path, "rb") as model_file:
                return pickle.load(model_file)
        except Exception:
            return None

    def _classify_with_model(self, features_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if self.model is None:
            return None

        vector = features_payload.get("ml_feature_vector") or features_payload.get("feature_vector")
        if not vector:
            return None

        try:
            probabilities: Dict[str, float] = {}

            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba([vector])[0]
                classes = list(getattr(self.model, "classes_", range(len(probs))))
                for cls_idx, prob in zip(classes, probs):
                    label = str(cls_idx)
                    if isinstance(cls_idx, str):
                        label = cls_idx
                    elif isinstance(cls_idx, int) and 0 <= cls_idx < len(self.LABELS):
                        label = self.LABELS[cls_idx]
                    probabilities[label] = float(prob)

                gesture = max(probabilities.items(), key=lambda item: item[1])[0]
                confidence = probabilities[gesture]
                return {
                    "gesture": gesture,
                    "confidence": confidence,
                    "source": "ml",
                    "scores": probabilities,
                }

            prediction = self.model.predict([vector])[0]
            if isinstance(prediction, int) and 0 <= prediction < len(self.LABELS):
                label = self.LABELS[prediction]
            else:
                label = str(prediction)

            return {
                "gesture": label,
                "confidence": 0.75,
                "source": "ml",
                "scores": {label: 0.75},
            }
        except Exception:
            return None

    def _classify_with_rules(self, hands_payload: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not hands_payload:
            return {"gesture": "None", "confidence": 0.0, "scores": {}}

        hand = hands_payload[0]
        landmarks = hand.get("landmarks", [])
        handedness = hand.get("handedness", "Right")

        if len(landmarks) != 21:
            return {"gesture": "None", "confidence": 0.0, "scores": {}}

        fingers_up = self._fingers_up(landmarks, handedness)
        count_up = sum(1 for is_up in fingers_up if is_up)

        thumb_up, index_up, middle_up, ring_up, little_up = fingers_up
        wrist_x = landmarks[0]["x"]
        index_x = landmarks[8]["x"]
        confidence = min(0.99, 0.45 + (count_up / 5.0) * 0.35)

        if count_up == 0:
            return {"gesture": "Fist", "confidence": 0.90, "scores": {"Fist": 0.90}}

        if count_up >= 4:
            return {"gesture": "Open Palm", "confidence": 0.90, "scores": {"Open Palm": 0.90}}

        if index_up and middle_up and not ring_up and not little_up:
            return {"gesture": "Two Finger", "confidence": 0.85, "scores": {"Two Finger": 0.85}}

        if index_up and not middle_up and not ring_up and not little_up:
            if index_x < wrist_x - 0.05:
                return {"gesture": "Point Left", "confidence": 0.80, "scores": {"Point Left": 0.80}}
            if index_x > wrist_x + 0.05:
                return {"gesture": "Point Right", "confidence": 0.80, "scores": {"Point Right": 0.80}}
            return {"gesture": "Point", "confidence": 0.82, "scores": {"Point": 0.82}}

        if thumb_up and not index_up and not middle_up and not ring_up and not little_up:
            return {"gesture": "Thumb", "confidence": 0.78, "scores": {"Thumb": 0.78}}

        return {"gesture": "None", "confidence": confidence * 0.4, "scores": {"None": confidence * 0.4}}

    def _fingers_up(self, landmarks: List[Dict[str, float]], handedness: str) -> List[bool]:
        thumb_tip_x = landmarks[self.TIP_IDS[0]]["x"]
        thumb_ip_x = landmarks[self.PIP_IDS[0]]["x"]

        if handedness.lower() == "right":
            thumb_up = thumb_tip_x > thumb_ip_x
        else:
            thumb_up = thumb_tip_x < thumb_ip_x

        other_fingers = []
        for tip_id, pip_id in zip(self.TIP_IDS[1:], self.PIP_IDS[1:]):
            tip_y = landmarks[tip_id]["y"]
            pip_y = landmarks[pip_id]["y"]
            other_fingers.append(tip_y < pip_y)

        return [thumb_up, *other_fingers]
