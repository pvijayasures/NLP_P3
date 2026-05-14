"""
Training loop for Audio2Emotion (openSMILE features → MLP).

Usage:
  python -m src.audio.mlp.train
  python -m src.audio.mlp.train is10
  python -m src.audio.mlp.train emobase

Outputs (per feature set):
  models/audio_{feature_set}_best.pt        best checkpoint (by dev weighted-F1)
  outputs/metrics/audio_{feature_set}.json  test metrics
  outputs/plots/cm_audio_{feature_set}.png  confusion matrix
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from config import MODELS_DIR, METRICS_DIR, RANDOM_SEED, NUM_CLASSES, FEATURE_SETS, DEFAULT_FEATURE_SET
from audio.dataset import AudioFeaturesDataset
from audio.mlp.model import AudioMLP, AUDIO_LR, AUDIO_EPOCHS, AUDIO_BATCH
from evaluate import evaluate_and_save

MODELS_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_class_weights(dataset: AudioFeaturesDataset) -> torch.Tensor:
    counts = torch.bincount(dataset.labels, minlength=NUM_CLASSES).float()
    weights = counts.sum() / (NUM_CLASSES * counts)
    return weights.to(DEVICE)


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, all_preds, all_labels = 0.0, [], []

    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(y)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    mf1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, mf1


def checkpoint_path(feature_set: str) -> Path:
    return MODELS_DIR / f"audio_mlp_{feature_set}_best.pt"


def train(feature_set: str = DEFAULT_FEATURE_SET):
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set '{feature_set}'. Valid: {list(FEATURE_SETS)}")

    checkpoint = checkpoint_path(feature_set)
    print(f"Device: {DEVICE}")
    print(f"Feature set: {feature_set} ({FEATURE_SETS[feature_set][0]})")

    train_ds = AudioFeaturesDataset("train", feature_set=feature_set)
    dev_ds   = AudioFeaturesDataset("dev",  feature_set=feature_set, scaler=train_ds.scaler)
    test_ds  = AudioFeaturesDataset("test", feature_set=feature_set, scaler=train_ds.scaler)

    train_loader = DataLoader(train_ds, batch_size=AUDIO_BATCH, shuffle=True, drop_last=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=AUDIO_BATCH, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=AUDIO_BATCH, shuffle=False)

    model = AudioMLP(input_dim=train_ds.feature_dim).to(DEVICE)
    class_weights = compute_class_weights(train_ds)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=AUDIO_LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_dev_f1 = 0.0

    for epoch in range(1, AUDIO_EPOCHS + 1):
        train_loss, train_f1 = run_epoch(model, train_loader, criterion, optimizer)
        dev_loss,   dev_f1   = run_epoch(model, dev_loader,   criterion)
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{AUDIO_EPOCHS} | "
            f"train loss {train_loss:.4f} mF1 {train_f1:.4f} | "
            f"dev loss {dev_loss:.4f} mF1 {dev_f1:.4f}"
            + (" ← best" if dev_f1 > best_dev_f1 else "")
        )

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            torch.save(model.state_dict(), checkpoint)

    print(f"\nBest dev macro-F1: {best_dev_f1:.4f}")

    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x.to(DEVICE))
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(y.tolist())

    evaluate_and_save(
        y_true=np.array(all_labels),
        y_pred=np.array(all_preds),
        model_name=f"audio_mlp_{feature_set}",
    )

    print("\nComparison across feature sets (MLP):")
    for key in FEATURE_SETS:
        p = METRICS_DIR / f"audio_mlp_{key}.json"
        if p.exists():
            with open(p) as f:
                m = json.load(f)
            marker = " ←" if key == feature_set else ""
            print(f"  {key:10s}  mF1 {m['macro_f1']:.4f}{marker}")


if __name__ == "__main__":
    fs = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FEATURE_SET
    train(fs)
