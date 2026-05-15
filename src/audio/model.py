"""
MLP classifier on top of eGeMAPS feature vectors.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from config import NUM_CLASSES

# Hyperparameters (imported by train.py)
AUDIO_HIDDEN  = [256, 128]
AUDIO_DROPOUT = 0.4
AUDIO_LR      = 1e-3
AUDIO_EPOCHS  = 50
AUDIO_BATCH   = 64


class AudioMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: list[int] = AUDIO_HIDDEN, dropout: float = AUDIO_DROPOUT):
        super().__init__()

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
