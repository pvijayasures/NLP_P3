"""
SVM baseline for Audio2Emotion (closer to the MELD paper's approach).

CalibratedClassifierCV wraps LinearSVC to provide predict_proba for fusion.

Usage:
  python -m src.audio.svm.train                 # egemaps (default)
  python -m src.audio.svm.train is10            # IS10 features
  python -m src.audio.svm.train emobase         # emobase features

Outputs (per feature set):
  outputs/checkpoints/audio_svm_{feature_set}_scaler.pkl
  outputs/checkpoints/audio_svm_{feature_set}_model.pkl
  outputs/metrics/audio_svm_{feature_set}.json
  outputs/predictions/audio_svm_{feature_set}_softmax_test.npy
"""

import sys
import json
import pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report

from config import (
    CHECKPOINTS_DIR, METRICS_DIR, PREDICTIONS_DIR,
    FEATURE_SETS, DEFAULT_FEATURE_SET,
    EMOTION_LABELS, NUM_CLASSES,
)
from audio.dataset import load_feature_arrays
from evaluate import evaluate_and_save

for d in (CHECKPOINTS_DIR, METRICS_DIR, PREDICTIONS_DIR):
    d.mkdir(parents=True, exist_ok=True)

def load_split(split: str, feature_set: str) -> tuple[np.ndarray, np.ndarray]:
    return load_feature_arrays(split=split, feature_set=feature_set)


def main(feature_set: str = DEFAULT_FEATURE_SET) -> None:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set '{feature_set}'. Valid: {list(FEATURE_SETS)}")

    model_tag = f"audio_svm_{feature_set}"
    print(f"Feature set : {feature_set} ({FEATURE_SETS[feature_set][0]})")

    print("Loading features...")
    X_train, y_train = load_split("train", feature_set)
    X_dev,   y_dev   = load_split("dev",   feature_set)
    X_test,  y_test  = load_split("test",  feature_set)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_dev   = scaler.transform(X_dev)
    X_test  = scaler.transform(X_test)

    with open(CHECKPOINTS_DIR / f"{model_tag}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("Scaler saved.")

    print("Training SVM (LinearSVC + Platt calibration)...")
    svm = CalibratedClassifierCV(
        LinearSVC(max_iter=2000, class_weight="balanced", C=0.1),
        cv=3,
    )
    svm.fit(X_train, y_train)

    with open(CHECKPOINTS_DIR / f"{model_tag}_model.pkl", "wb") as f:
        pickle.dump(svm, f)
    print("Model saved.")

    print("\n--- Dev set ---")
    y_dev_pred = svm.predict(X_dev)
    print(classification_report(y_dev, y_dev_pred, target_names=EMOTION_LABELS, zero_division=0))

    print("--- Test set ---")
    y_test_pred = svm.predict(X_test)
    results = evaluate_and_save(
        y_true=y_test,
        y_pred=y_test_pred,
        model_name=model_tag,
    )

    proba = svm.predict_proba(X_test)
    softmax = np.zeros((len(y_test), NUM_CLASSES), dtype=np.float32)
    classes = svm.calibrated_classifiers_[0].classes
    for col_idx, class_idx in enumerate(classes):
        softmax[:, class_idx] = proba[:, col_idx]

    np.save(PREDICTIONS_DIR / f"{model_tag}_softmax_test.npy", softmax)
    print(f"Softmax probabilities saved → shape {softmax.shape}")

    # compare all available SVM variants
    print("\nComparison across feature sets (SVM):")
    for key in FEATURE_SETS:
        p = METRICS_DIR / f"audio_svm_{key}.json"
        if p.exists():
            with open(p) as f:
                m = json.load(f)
            marker = " ←" if key == feature_set else ""
            print(f"  {key:10s}  mF1 {m['macro_f1']:.4f}{marker}")


if __name__ == "__main__":
    fs = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FEATURE_SET
    main(fs)
