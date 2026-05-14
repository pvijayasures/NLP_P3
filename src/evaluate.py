"""
Einheitliche Evaluation für alle Modelle.
Alle Modelle müssen diese Funktion verwenden → faire Vergleiche garantiert.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report,
    confusion_matrix
)
from config import EMOTION_LABELS, OUTPUTS_DIR


def evaluate_and_save(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_plots: bool = True,
) -> dict:
    """
    Berechnet alle Metriken, speichert JSON + Confusion-Matrix-Plot.

    Returns: dict mit allen Metriken
    """
    metrics_dir = OUTPUTS_DIR / "metrics"
    plots_dir   = OUTPUTS_DIR / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    macro_f1    = f1_score(y_true, y_pred, average="macro")
    accuracy    = accuracy_score(y_true, y_pred)
    report      = classification_report(
        y_true, y_pred,
        target_names=EMOTION_LABELS,
        output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred)

    results = {
        "model":       model_name,
        "weighted_f1": round(weighted_f1, 4),
        "macro_f1":    round(macro_f1, 4),
        "accuracy":    round(accuracy, 4),
        "per_class":   {
            label: {
                "precision": round(report[label]["precision"], 4),
                "recall":    round(report[label]["recall"], 4),
                "f1":        round(report[label]["f1-score"], 4),
                "support":   int(report[label]["support"]),
            }
            for label in EMOTION_LABELS
        },
    }

    # JSON speichern
    json_path = metrics_dir / f"{model_name}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"  Weighted F1 : {weighted_f1:.4f}")
    print(f"  Macro F1    : {macro_f1:.4f}")
    print(f"  Accuracy    : {accuracy:.4f}")
    print(classification_report(y_true, y_pred, target_names=EMOTION_LABELS))

    if save_plots:
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=EMOTION_LABELS,
            yticklabels=EMOTION_LABELS,
            ax=ax
        )
        ax.set_title(f"Confusion Matrix — {model_name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(plots_dir / f"cm_{model_name}.png", dpi=150)
        plt.close()

    return results