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


def _add_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepend the previous utterance's audio features within the same dialogue.
    First utterance in each dialogue gets zero-padded previous features.
    Result: [id_cols | prev_<feat> | <feat>] — input dimension doubles.
    """
    id_cols   = [c for c in df.columns if c in ID_COLS]
    feat_cols = [c for c in df.columns if c not in ID_COLS]

    df = df.sort_values(["dialogue_id", "utterance_id"]).reset_index(drop=True)

    prev_df = (df.groupby("dialogue_id")[feat_cols]
                 .shift(1)
                 .fillna(0.0))
    prev_df.columns = [f"prev_{c}" for c in feat_cols]

    return pd.concat([df[id_cols + feat_cols], prev_df], axis=1)


def load_feature_dataframe(
    split: str,
    feature_set: str = DEFAULT_FEATURE_SET,
    use_context: bool = False,
) -> pd.DataFrame:
    parquet = FEATURES_DIR / f"{split}_{feature_set}.parquet"
    df = pd.read_parquet(parquet)

    missing_cols = ID_COLS.difference(df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required columns in {parquet.name}: {sorted(missing_cols)}"
        )

    nan_mask = df.isna().any(axis=1)
    dropped_rows = int(nan_mask.sum())
    if dropped_rows:
        print(
            f"[WARN] Dropped {dropped_rows} row(s) with NaN values from "
            f"{parquet.name} ({split}/{feature_set})."
        )
        df = df.loc[~nan_mask].reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No valid rows left in {parquet.name} after NaN filtering.")

    if use_context:
        df = _add_context(df)

    return df


def load_feature_arrays(
    split: str,
    feature_set: str = DEFAULT_FEATURE_SET,
    use_context: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    df = load_feature_dataframe(split=split, feature_set=feature_set, use_context=use_context)
    feat_cols = [c for c in df.columns if c not in ID_COLS]
    X = df[feat_cols].values.astype(np.float32)
    y = df["label_idx"].values.astype(int)
    return X, y


class AudioFeaturesDataset(Dataset):
    def __init__(
        self,
        split: str,
        feature_set: str = DEFAULT_FEATURE_SET,
        scaler: StandardScaler | None = None,
        use_context: bool = False,
    ):
        """
        split:       "train" | "dev" | "test"
        feature_set: "egemaps" | "is10" | "emobase"
        scaler:      pre-fit scaler for dev/test; None triggers fit on train.
        use_context: if True, prepend previous utterance features (doubles input dim).
        """
        df = load_feature_dataframe(split=split, feature_set=feature_set, use_context=use_context)

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
