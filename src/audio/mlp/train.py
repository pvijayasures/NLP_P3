"""
Optimised Audio2Emotion MLP — supports egemaps, is10, emobase feature sets.

CHANGES FROM BASELINE:
- StandardScaler fit on train only (no leakage)
- Configurable architecture searched by optuna [128, 64] default
- Dropout searched by optuna (0.3–0.6)
- Early stopping with patience=10 on dev weighted F1
- ReduceLROnPlateau scheduler (factor=0.5, patience=5)
- Bayesian hyperparameter search via optuna (20 trials)
- FocalLoss with class weights moved to correct device

SEARCH SPACE (optuna):
  hidden_1   : categorical [64, 128, 256]
  hidden_2   : categorical [32, 64, 128]
  dropout    : float [0.3, 0.6]
  lr         : float log [1e-4, 1e-2]
  focal_gamma: float [1.0, 3.0]

Usage:
  python -m src.audio.mlp.train              # egemaps (default)
  python -m src.audio.mlp.train is10
  python -m src.audio.mlp.train emobase

Outputs (per feature set):
  outputs/checkpoints/audio_mlp_{feature_set}_scaler.pkl
  outputs/checkpoints/audio_mlp_{feature_set}_best.pt
  outputs/checkpoints/audio_mlp_{feature_set}_hparams.json
  outputs/metrics/audio_mlp_{feature_set}.json
  outputs/predictions/audio_mlp_{feature_set}_softmax_test.npy
"""

import json
import pickle
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

from config import (
    CHECKPOINTS_DIR, METRICS_DIR, PREDICTIONS_DIR,
    RANDOM_SEED,
    NUM_CLASSES, LABEL2IDX,
    FEATURE_SETS, DEFAULT_FEATURE_SET,
)
from audio.dataset import load_feature_arrays
from audio.mlp.model import AudioMLP, FocalLoss
from evaluate import evaluate_and_save

optuna.logging.set_verbosity(optuna.logging.WARNING)

for d in (CHECKPOINTS_DIR, METRICS_DIR, PREDICTIONS_DIR):
    d.mkdir(parents=True, exist_ok=True)

BATCH_SIZE   = 64
N_TRIALS     = 20
SEARCH_EPOCHS = 30
SEARCH_PATIENCE = 5
FINAL_EPOCHS  = 100
FINAL_PATIENCE = 10

torch.manual_seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, drop_last=shuffle)


def make_class_weights(y_train: np.ndarray) -> torch.Tensor:
    w = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=y_train)
    return torch.tensor(w, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    all_preds, all_labels = [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(y.cpu().tolist())
    wf1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return wf1


def train_model(
    X_train, y_train, X_dev, y_dev,
    hidden_1, hidden_2, dropout, lr, focal_gamma,
    max_epochs, patience,
    checkpoint: Path | None = None,
) -> float:
    class_weights = make_class_weights(y_train).to(DEVICE)
    model = AudioMLP(
        input_dim=X_train.shape[1],
        hidden=[hidden_1, hidden_2],
        dropout=dropout,
    ).to(DEVICE)
    criterion = FocalLoss(weight=class_weights, gamma=focal_gamma)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    train_loader = make_loader(X_train, y_train, shuffle=True)
    dev_loader   = make_loader(X_dev,   y_dev,   shuffle=False)

    best_dev_mf1 = 0.0
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        run_epoch(model, train_loader, criterion, optimizer)
        dev_mf1 = run_epoch(model, dev_loader, criterion)
        scheduler.step(dev_mf1)

        if dev_mf1 > best_dev_mf1:
            best_dev_mf1 = dev_mf1
            epochs_no_improve = 0
            if checkpoint is not None:
                torch.save(model.state_dict(), checkpoint)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    return best_dev_mf1


def objective(trial, X_train, y_train, X_dev, y_dev) -> float:
    hidden_1    = trial.suggest_categorical("hidden_1", [64, 128, 256])
    hidden_2    = trial.suggest_categorical("hidden_2", [32, 64, 128])
    dropout     = trial.suggest_float("dropout", 0.3, 0.6)
    lr          = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    focal_gamma = trial.suggest_float("focal_gamma", 1.0, 3.0)
    return train_model(
        X_train, y_train, X_dev, y_dev,
        hidden_1, hidden_2, dropout, lr, focal_gamma,
        max_epochs=SEARCH_EPOCHS, patience=SEARCH_PATIENCE,
        checkpoint=None,
    )


def main(feature_set: str = DEFAULT_FEATURE_SET):
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set '{feature_set}'. Valid: {list(FEATURE_SETS)}")

    model_tag = f"audio_mlp_{feature_set}"
    fs_name   = FEATURE_SETS[feature_set][0]

    print(f"Device      : {DEVICE}")
    print(f"Feature set : {feature_set} ({fs_name})")

    print("Loading features...")
    X_train, y_train = load_feature_arrays("train", feature_set)
    X_dev,   y_dev   = load_feature_arrays("dev",   feature_set)
    X_test,  y_test  = load_feature_arrays("test",  feature_set)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_dev   = scaler.transform(X_dev)
    X_test  = scaler.transform(X_test)

    scaler_file = CHECKPOINTS_DIR / f"{model_tag}_scaler.pkl"
    with open(scaler_file, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved → {scaler_file.name}")

    print(f"\nBayesian search ({N_TRIALS} trials, dev macro-F1)...")
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

    hparams_file = CHECKPOINTS_DIR / f"{model_tag}_hparams.json"
    with open(hparams_file, "w") as f:
        json.dump({**best, "dev_mf1": round(study.best_value, 4)}, f, indent=2)
    print(f"Hparams saved → {hparams_file.name}")

    checkpoint = CHECKPOINTS_DIR / f"{model_tag}_best.pt"
    print(f"\nFinal training (max {FINAL_EPOCHS} epochs, patience={FINAL_PATIENCE})...")
    best_dev_mf1 = train_model(
        X_train, y_train, X_dev, y_dev,
        best["hidden_1"], best["hidden_2"],
        best["dropout"], best["lr"], best["focal_gamma"],
        max_epochs=FINAL_EPOCHS, patience=FINAL_PATIENCE,
        checkpoint=checkpoint,
    )
    print(f"Best dev mF1 : {best_dev_mf1:.4f}")
    print(f"Checkpoint   → {checkpoint.name}")

    model = AudioMLP(
        input_dim=X_train.shape[1],
        hidden=[best["hidden_1"], best["hidden_2"]],
        dropout=best["dropout"],
    ).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.eval()

    test_loader = make_loader(X_test, y_test, shuffle=False)
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x.to(DEVICE))
            all_probs.append(F.softmax(logits, dim=1).cpu())
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(y.tolist())

    evaluate_and_save(
        y_true=np.array(all_labels),
        y_pred=np.array(all_preds),
        model_name=model_tag,
    )

    softmax = torch.cat(all_probs, dim=0).numpy().astype(np.float32)
    pred_file = PREDICTIONS_DIR / f"{model_tag}_softmax_test.npy"
    np.save(pred_file, softmax)
    print(f"Softmax saved → {pred_file.name}  shape {softmax.shape}")

    print(f"\nComparison across feature sets (MLP, mF1):")
    for fs in FEATURE_SETS:
        p = METRICS_DIR / f"audio_mlp_{fs}.json"
        if p.exists():
            m = json.loads(p.read_text())
            marker = " ←" if fs == feature_set else ""
            print(f"  {fs:10s}  mF1 {m['macro_f1']:.4f}{marker}")


if __name__ == "__main__":
    fs = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FEATURE_SET
    main(fs)
