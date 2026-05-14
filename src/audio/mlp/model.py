"""
MLP classifier on top of openSMILE feature vectors.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NUM_CLASSES

# Hyperparameters (imported by train.py)
AUDIO_INPUT_DIM = 88  # default for eGeMAPSv02; training passes the active feature dimension explicitly
AUDIO_HIDDEN  = [256, 128]
AUDIO_DROPOUT = 0.4
AUDIO_LR      = 1e-3
AUDIO_EPOCHS  = 50
AUDIO_BATCH   = 64
FOCAL_GAMMA   = 2.0  # tunable: try 1.0 if minority classes still underperform


class FocalLoss(nn.Module):
    """Cross-entropy with focal down-weighting of easy examples."""
    def __init__(self, weight: torch.Tensor | None = None, gamma: float = FOCAL_GAMMA):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=-1)
        weight = self.weight.to(logits.device) if self.weight is not None else None
        ce = F.nll_loss(log_p, targets, weight=weight, reduction="none")
        p_t = torch.exp(-ce)
        return ((1 - p_t) ** self.gamma * ce).mean()


class AudioMLP(nn.Module):
    def __init__(self, input_dim: int = AUDIO_INPUT_DIM, hidden: list[int] | None = None, dropout: float = AUDIO_DROPOUT):
        super().__init__()
        if hidden is None:
            hidden = AUDIO_HIDDEN

        layers = []
        in_dim = input_dim
        for out_dim in hidden:
            layers += [
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = out_dim

        layers.append(nn.Linear(in_dim, NUM_CLASSES))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
