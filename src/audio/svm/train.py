"""
Optimised SVM baseline for Audio2Emotion (closest to MELD paper approach).

CalibratedClassifierCV(cv='prefit') wraps LinearSVC for predict_proba.
Calibration is done on dev set after training on full train split.

CHANGES FROM NAIVE BASELINE:
- cv='prefit' + method='sigmoid': train once on full train, calibrate on dev
  (faster than cv=3, no train data spent on calibration folds)
- Bayesian hyperparameter search via optuna (20 trials) on dev weighted F1
- NaN rows dropped before scaling, logged per split
- Assert NUM_CLASSES classes present after calibration

SEARCH SPACE (optuna):
  C        : float log [1e-3, 10.0]
  tol      : float log [1e-5, 1e-2]
  max_iter : int [1000, 5000, step=500]

Usage:
  python -m src.audio.svm.train                 # egemaps (default)
  python -m src.audio.svm.train is10
  python -m src.audio.svm.train emobase

Outputs (per feature set):
  outputs/checkpoints/audio_svm_{feature_set}_scaler.pkl
  outputs/checkpoints/audio_svm_{feature_set}_model.pkl
  outputs/checkpoints/audio_svm_{feature_set}_hparams.json
  outputs/metrics/audio_svm_{feature_set}.json
  outputs/predictions/audio_svm_{feature_set}_softmax_test.npy
"""

import sys
import json
import pickle
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, f1_score

from config import (
    CHECKPOINTS_DIR, METRICS_DIR, PREDICTIONS_DIR,
    FEATURE_SETS, DEFAULT_FEATURE_SET,
    EMOTION_LABELS, NUM_CLASSES,
    SVM_BEST_PARAMS,
    get_model_tag, get_model_path, get_scaler_path,
    get_hparams_path, get_metrics_path, get_prediction_path,
)
from audio.dataset import load_feature_arrays
from evaluate import evaluate_and_save

optuna.logging.set_verbosity(optuna.logging.WARNING)

for d in (CHECKPOINTS_DIR, METRICS_DIR, PREDICTIONS_DIR):
    d.mkdir(parents=True, exist_ok=True)

N_TRIALS = 20


def objective(trial, X_train, y_train, X_dev, y_dev) -> float:
    C        = trial.suggest_float("C", 1e-3, 10.0, log=True)
    tol      = trial.suggest_float("tol", 1e-5, 1e-2, log=True)
    max_iter = trial.suggest_int("max_iter", 1000, 5000, step=500)

    svc = LinearSVC(C=C, tol=tol, max_iter=max_iter, class_weight="balanced")
    svc.fit(X_train, y_train)
    svm = CalibratedClassifierCV(svc, cv="prefit", method="sigmoid")
    svm.fit(X_dev, y_dev)
    return f1_score(y_dev, svm.predict(X_dev), average="macro", zero_division=0)


def main(feature_set: str = DEFAULT_FEATURE_SET, optimize: bool = True) -> None:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set '{feature_set}'. Valid: {list(FEATURE_SETS)}")

    model_key = "svm"
    model_tag = get_model_tag(model_key, feature_set)
    scaler_path = get_scaler_path(model_key, feature_set)
    hparams_path = get_hparams_path(model_key, feature_set)
    model_path = get_model_path(model_key, feature_set)
    pred_path = get_prediction_path(model_key, feature_set, split="test")
    print(f"Feature set : {feature_set} ({FEATURE_SETS[feature_set][0]})")

    print("Loading features...")
    X_train, y_train = load_feature_arrays("train", feature_set)
    X_dev,   y_dev   = load_feature_arrays("dev",   feature_set)
    X_test,  y_test  = load_feature_arrays("test",  feature_set)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_dev   = scaler.transform(X_dev)
    X_test  = scaler.transform(X_test)

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print("Scaler saved.")

    if optimize:
        print(f"Bayesian search ({N_TRIALS} trials, dev macro-F1)...")
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective(trial, X_train, y_train, X_dev, y_dev),
            n_trials=N_TRIALS,
            n_jobs=4,
            show_progress_bar=True,
        )
        best = study.best_params
        print(f"Best dev mF1 : {study.best_value:.4f}")
        print(f"Best params  : {best}")
        with open(hparams_path, "w") as f:
            json.dump({**best, "dev_mf1": round(study.best_value, 4)}, f, indent=2)
    else:
        if feature_set not in SVM_BEST_PARAMS:
            raise ValueError(
                f"No saved params for '{feature_set}' in config.SVM_BEST_PARAMS. "
                "Run without --no-optimize first."
            )
        best = {k: v for k, v in SVM_BEST_PARAMS[feature_set].items() if k != "dev_mf1"}
        if not hparams_path.exists():
            with open(hparams_path, "w") as f:
                json.dump({**best, "dev_mf1": SVM_BEST_PARAMS[feature_set].get("dev_mf1")}, f, indent=2)
            print(f"Hparams missing -> created from config: {hparams_path.name}")
        print(f"Using saved params from config: {best}")
        print(f"(dev mF1 when found: {SVM_BEST_PARAMS[feature_set]['dev_mf1']})")

    print("Training final model with best hyperparameters...")
    svc = LinearSVC(
        C=best["C"], tol=best["tol"], max_iter=best["max_iter"],
        class_weight="balanced",
    )
    svc.fit(X_train, y_train)
    svm = CalibratedClassifierCV(svc, cv="prefit", method="sigmoid")
    svm.fit(X_dev, y_dev)

    assert len(svm.classes_) == NUM_CLASSES, (
        f"Expected {NUM_CLASSES} classes after calibration, got {len(svm.classes_)}"
    )

    with open(model_path, "wb") as f:
        pickle.dump(svm, f)
    print("Model saved.")

    print("\n--- Dev set ---")
    y_dev_pred = svm.predict(X_dev)
    print(classification_report(y_dev, y_dev_pred, target_names=EMOTION_LABELS, zero_division=0))

    print("--- Test set ---")
    y_test_pred = svm.predict(X_test)
    evaluate_and_save(y_true=y_test, y_pred=y_test_pred, model_name=model_tag)

    proba = svm.predict_proba(X_test)
    softmax = np.zeros((len(y_test), NUM_CLASSES), dtype=np.float32)
    for col_idx, class_idx in enumerate(svm.classes_):
        softmax[:, class_idx] = proba[:, col_idx]
    np.save(pred_path, softmax)
    print(f"Softmax saved → shape {softmax.shape}")

    print("\nComparison across feature sets (SVM, mF1):")
    for key in FEATURE_SETS:
        p = get_metrics_path(model_key, key)
        if p.exists():
            m = json.loads(p.read_text())
            marker = " ←" if key == feature_set else ""
            print(f"  {key:10s}  wF1 {m['macro_f1']:.4f}{marker}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("feature_set", nargs="?", default=DEFAULT_FEATURE_SET)
    parser.add_argument("--no-optimize", action="store_true",
                        help="Skip Optuna search and use saved best params from config")
    args = parser.parse_args()
    main(args.feature_set, optimize=not args.no_optimize)
