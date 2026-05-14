"""
Phase 2: 16kHz WAV → openSMILE feature vectors.

Reads pre-converted WAVs from data/wav_16k/{split}/ (produced by convert_to_wav.py)
and saves features to data/features/{split}_{feature_set}.parquet.

Keys (dialogue_id, utterance_id, label_idx) are preserved for joining.
Failed utterances are logged to {split}_{feature_set}_failed.csv.

Run convert_to_wav.py once before this script.

Usage:
  python -m src.preprocessing.extract_audio_features               # egemaps (default)
  python -m src.preprocessing.extract_audio_features is10          # IS10 (1582 features)
  python -m src.preprocessing.extract_audio_features emobase       # emobase (988 features)
  python -m src.preprocessing.extract_audio_features egemaps train # single split
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import opensmile
from tqdm import tqdm

from config import (
    ROOT, PROCESSED_DIR, FEATURES_DIR, WAV_DIR,
    FEATURE_SETS, DEFAULT_FEATURE_SET,
)

FEATURES_DIR.mkdir(parents=True, exist_ok=True)


def build_smile(feature_set_key: str) -> opensmile.Smile:
    if feature_set_key not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature set '{feature_set_key}'. "
            f"Valid options: {list(FEATURE_SETS)}"
        )
    fs_name, fl_name = FEATURE_SETS[feature_set_key]

    try:
        fs = getattr(opensmile.FeatureSet, fs_name)
    except AttributeError:
        raise ValueError(
            f"openSMILE has no FeatureSet '{fs_name}'. "
            f"Available: {[e.name for e in opensmile.FeatureSet]}"
        )
    try:
        fl = getattr(opensmile.FeatureLevel, fl_name)
    except AttributeError:
        raise ValueError(
            f"openSMILE has no FeatureLevel '{fl_name}'. "
            f"Available: {[e.name for e in opensmile.FeatureLevel]}"
        )

    return opensmile.Smile(feature_set=fs, feature_level=fl)


def extract_split(split: str, smile: opensmile.Smile, feature_set_key: str) -> pd.DataFrame:
    manifest_path = PROCESSED_DIR / f"manifest_{split}.csv"
    manifest = pd.read_csv(manifest_path)

    rows = []
    failed_ids = []

    for _, row in tqdm(
        manifest.iterrows(),
        total=len(manifest),
        desc=f"[{split}/{feature_set_key}] extracting",
        ncols=90,
    ):
        dia_id   = int(row["dialogue_id"])
        utt_id   = int(row["utterance_id"])
        wav_path = ROOT / row["audio_path"]

        if not wav_path.exists():
            tqdm.write(f"  [WARN] missing: {wav_path.name}")
            failed_ids.append({
                "dialogue_id":  dia_id,
                "utterance_id": utt_id,
                "reason": "file_not_found",
            })
            continue

        try:
            feats_df = smile.process_file(str(wav_path)).reset_index(drop=True)

            assert len(feats_df) == 1, (
                f"Expected 1 feature row, got {len(feats_df)} for {wav_path.name}"
            )

            feat_row = feats_df.iloc[0].to_dict()

        except Exception as e:
            tqdm.write(f"  [WARN] failed: {wav_path.name} — {e}")
            failed_ids.append({
                "dialogue_id":  dia_id,
                "utterance_id": utt_id,
                "reason": str(e),
            })
            continue

        feat_row["dialogue_id"]  = dia_id
        feat_row["utterance_id"] = utt_id
        feat_row["label_idx"]    = int(row["label_idx"])
        rows.append(feat_row)

    id_cols = ["dialogue_id", "utterance_id", "label_idx"]
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=id_cols)
        feat_cols = []
    else:
        feat_cols = [c for c in df.columns if c not in id_cols]
        df = df[id_cols + feat_cols]

    out_path = FEATURES_DIR / f"{split}_{feature_set_key}.parquet"
    df.to_parquet(out_path, index=False)

    if failed_ids:
        failed_df = pd.DataFrame(failed_ids)
        failed_path = FEATURES_DIR / f"{split}_{feature_set_key}_failed.csv"
        failed_df.to_csv(failed_path, index=False)
        tqdm.write(f"  [INFO] failed IDs saved → {failed_path}")

    tqdm.write(
        f"[{split}/{feature_set_key}] done — {len(df)} utterances | "
        f"{len(failed_ids)} failed | {len(feat_cols)} features"
    )
    return df


if __name__ == "__main__":
    args = sys.argv[1:]

    # parse: optional feature_set key, then optional split names
    feature_set_key = DEFAULT_FEATURE_SET
    splits = ["train", "dev", "test"]

    if args and args[0] in FEATURE_SETS:
        feature_set_key = args.pop(0)
    if args:
        splits = args

    print(f"Feature set : {feature_set_key} ({FEATURE_SETS[feature_set_key][0]})")
    print(f"Splits      : {splits}")

    smile = build_smile(feature_set_key)
    for split in splits:
        extract_split(split, smile, feature_set_key)
