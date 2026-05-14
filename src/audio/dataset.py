"""
PyTorch Dataset for openSMILE feature vectors + StandardScaler.

AudioFeaturesDataset loads a parquet produced by extract_audio_features.py
and returns (feature_tensor, label) pairs.

Scaler workflow (no leakage):
  - Fit on train split → saved to models/audio_{feature_set}_scaler.pkl
  - Pass the fitted scaler for dev/test splits
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from config import FEATURES_DIR, MODELS_DIR, DEFAULT_FEATURE_SET

MODELS_DIR.mkdir(parents=True, exist_ok=True)

ID_COLS = {"dialogue_id", "utterance_id", "label_idx"}


def scaler_path(feature_set: str) -> Path:
    return MODELS_DIR / f"audio_{feature_set}_scaler.pkl"


class AudioFeaturesDataset(Dataset):
    def __init__(
        self,
        split: str,
        feature_set: str = DEFAULT_FEATURE_SET,
        scaler: StandardScaler | None = None,
    ):
        """
        split:       "train" | "dev" | "test"
        feature_set: "egemaps" | "is10" | "emobase"
        scaler:      pre-fit scaler for dev/test; None triggers fit on train.
        """
        parquet = FEATURES_DIR / f"{split}_{feature_set}.parquet"
        df = pd.read_parquet(parquet)

        self.labels = torch.tensor(df["label_idx"].values, dtype=torch.long)

        feat_cols = [c for c in df.columns if c not in ID_COLS]
        X = df[feat_cols].values.astype(np.float32)

        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(X)
            with open(scaler_path(feature_set), "wb") as f:
                pickle.dump(scaler, f)

        self.features = torch.tensor(scaler.transform(X), dtype=torch.float32)
        self.scaler = scaler
        self.feature_dim = self.features.shape[1]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def load_scaler(feature_set: str = DEFAULT_FEATURE_SET) -> StandardScaler:
    with open(scaler_path(feature_set), "rb") as f:
        return pickle.load(f)
