from typing import Any, Dict, List


class LandmarkExtractor:
    """Extracts and normalizes 21 hand landmarks."""

    def extract(self, results: Any, frame_width: int, frame_height: int) -> List[Dict[str, Any]]:
        extracted: List[Dict[str, Any]] = []
        if not results:
            return extracted

        # MediaPipe Solutions API format
        if hasattr(results, "multi_hand_landmarks"):
            if not results.multi_hand_landmarks:
                return extracted

            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness_label = "Unknown"
                handedness_score = 0.0

                if results.multi_handedness and idx < len(results.multi_handedness):
                    classification = results.multi_handedness[idx].classification[0]
                    handedness_label = classification.label
                    handedness_score = float(classification.score)

                landmarks = []
                for point in hand_landmarks.landmark:
                    landmarks.append(
                        {
                            "x": point.x,
                            "y": point.y,
                            "z": point.z,
                            "px": int(point.x * frame_width),
                            "py": int(point.y * frame_height),
                            "pz": point.z * frame_width,
                        }
                    )

                extracted.append(
                    {
                        "hand_index": idx,
                        "handedness": handedness_label,
                        "score": handedness_score,
                        "landmarks": landmarks,
                    }
                )

            return extracted

        # MediaPipe Tasks API format
        hand_landmarks_list = getattr(results, "hand_landmarks", None)
        if not hand_landmarks_list:
            return extracted

        handedness_list = getattr(results, "handedness", [])

        for idx, hand_landmarks in enumerate(hand_landmarks_list):
            handedness_label = "Unknown"
            handedness_score = 0.0

            if idx < len(handedness_list) and handedness_list[idx]:
                category = handedness_list[idx][0]
                handedness_label = getattr(category, "category_name", "Unknown")
                handedness_score = float(getattr(category, "score", 0.0))

            landmarks = []
            for point in hand_landmarks:
                landmarks.append(
                    {
                        "x": point.x,
                        "y": point.y,
                        "z": point.z,
                        "px": int(point.x * frame_width),
                        "py": int(point.y * frame_height),
                        "pz": point.z * frame_width,
                    }
                )

            extracted.append(
                {
                    "hand_index": idx,
                    "handedness": handedness_label,
                    "score": handedness_score,
                    "landmarks": landmarks,
                }
            )

        return extracted
