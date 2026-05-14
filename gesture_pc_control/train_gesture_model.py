import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_DATASET = Path("model/keypoint_classifier/keypoint.csv")
DEFAULT_LABELS = Path("model/keypoint_classifier/keypoint_classifier_label.csv")
DEFAULT_MODEL_OUT = Path("model/keypoint_classifier/gesture_rf_model.pkl")
DEFAULT_REPORT_OUT = Path("model/keypoint_classifier/gesture_model_metrics.json")


def _load_labels(path: Path) -> List[str]:
    labels: List[str] = []
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            if not row:
                continue
            labels.append(row[0].strip())
    return labels


def _load_keypoint_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    x_values: List[List[float]] = []
    y_values: List[int] = []

    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            if not row:
                continue
            y_values.append(int(float(row[0])))
            x_values.append([float(value) for value in row[1:]])

    if not x_values:
        raise ValueError("Dataset is empty. Collect gesture landmarks before training.")

    x = np.asarray(x_values, dtype=np.float32)
    y = np.asarray(y_values, dtype=np.int64)
    return x, y


def train(
    dataset_path: Path,
    labels_path: Path,
    model_out: Path,
    report_out: Path,
    test_size: float,
    random_state: int,
) -> None:
    print("=" * 68)
    print("GESTURE LANDMARK CLASSIFIER TRAINING")
    print("=" * 68)
    print(f"Dataset: {dataset_path}")

    labels = _load_labels(labels_path)
    x, y = _load_keypoint_dataset(dataset_path)

    print(f"Samples: {x.shape[0]}")
    print(f"Features: {x.shape[1]}")
    print(f"Class IDs: {sorted(np.unique(y).tolist())}")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=350,
                    max_depth=24,
                    min_samples_split=3,
                    min_samples_leaf=1,
                    random_state=random_state,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )

    print("\nTraining model...")
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    acc = float(accuracy_score(y_test, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred).tolist()

    class_names = []
    for class_id in sorted(np.unique(y).tolist()):
        if 0 <= class_id < len(labels):
            class_names.append(labels[class_id])
        else:
            class_names.append(f"class_{class_id}")

    report_text = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )

    model_out.parent.mkdir(parents=True, exist_ok=True)
    with model_out.open("wb") as model_file:
        pickle.dump(model, model_file)

    report_payload = {
        "dataset": str(dataset_path),
        "samples": int(x.shape[0]),
        "features": int(x.shape[1]),
        "test_size": test_size,
        "random_state": random_state,
        "labels": class_names,
        "metrics": {
            "accuracy": acc,
            "precision_weighted": float(precision),
            "recall_weighted": float(recall),
            "f1_weighted": float(f1),
        },
        "confusion_matrix": cm,
        "classification_report": report_text,
    }

    with report_out.open("w", encoding="utf-8") as report_file:
        json.dump(report_payload, report_file, indent=2)

    print("\nSaved model:", model_out)
    print("Saved metrics:", report_out)
    print("\nAccuracy:", f"{acc:.4f}")
    print("Weighted Precision:", f"{float(precision):.4f}")
    print("Weighted Recall:", f"{float(recall):.4f}")
    print("\nClassification report:\n")
    print(report_text)
    print("\nConfusion matrix:")
    for row in cm:
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a gesture classifier from keypoint.csv")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_OUT)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    train(
        dataset_path=args.dataset,
        labels_path=args.labels,
        model_out=args.model_out,
        report_out=args.report_out,
        test_size=args.test_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
