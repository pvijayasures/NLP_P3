"""
Phase 1: MELD CSVs → aligned manifests mit 8 Labels.
Surprise wird anhand der vorhandenen MELD Sentiment-Spalte aufgeteilt:
  - Sentiment == "positive" → surprise_positive
  - Sentiment == "negative" / "neutral" → surprise_negative
Kein externes Sentiment-Tool nötig — kein Leakage-Risiko.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import ftfy
from config import (
    ROOT, CSV_FILES, AUDIO_DIRS, PROCESSED_DIR, LABEL2IDX
)

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def normalize_text(value: str) -> str:
    return (
        ftfy.fix_text(value)
        .replace("\u2019", "'")
        .replace("\u2013", "-")
    )


def resolve_emotion(emotion: str, sentiment: str) -> str:
    if emotion == "surprise":
        return "surprise_positive" if sentiment == "positive" else "surprise_negative"
    return emotion


def build_manifest(split: str) -> pd.DataFrame:
    df = pd.read_csv(CSV_FILES[split])
    audio_dir = AUDIO_DIRS[split]

    rows = []
    missing_audio = 0

    for _, row in df.iterrows():
        print(f"Processing Dialogue {row['Dialogue_ID']} Utterance {row['Utterance_ID']}...", end="\r")
        dia_id    = int(row["Dialogue_ID"])
        utt_id    = int(row["Utterance_ID"])
        emotion   = row["Emotion"].lower().strip()
        sentiment = row["Sentiment"].lower().strip()
        speaker   = normalize_text(str(row["Speaker"]))
        text      = normalize_text(str(row["Utterance"]))

        audio_path = audio_dir / f"dia{dia_id}_utt{utt_id}.mp4"
        if not audio_path.exists():
            print(f"\n  [WARN] missing: {audio_path}")
            missing_audio += 1
            continue

        label = resolve_emotion(emotion, sentiment)

        rows.append({
            "dialogue_id":  dia_id,
            "utterance_id": utt_id,
            "speaker":      speaker,
            "text":         text,
            "audio_path":   str(audio_path.relative_to(ROOT)),
            "emotion":      label,
            "label_idx":    LABEL2IDX[label],
        })

    manifest = pd.DataFrame(rows)
    out_path = PROCESSED_DIR / f"manifest_{split}.csv"
    manifest.to_csv(out_path, index=False)

    total = len(manifest) + missing_audio
    print(f"\n[{split}] {len(manifest)}/{total} Utterances OK | {missing_audio} missing audio files")
    if missing_audio > 0:
        print(f"  → Check data/raw/audio/{split}/ — expected naming: dia{{id}}_utt{{id}}.mp4")
        if split == "test":
            print(f"  → Did you run the 'final_videos_test' rename fix?")
    print(manifest["emotion"].value_counts().to_string())
    print()
    return manifest


if __name__ == "__main__":
    for split in ("train", "dev", "test"):
        build_manifest(split)