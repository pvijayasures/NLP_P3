"""
Phase 1: MELD CSVs → aligned manifests with 8 labels.

Requires convert_to_wav.py to have run first — audio_path in the manifest
points to data/wav_16k/{split}/{stem}.wav, not the original mp4.

Surprise is split using the MELD Sentiment column:
  - Sentiment == "positive" → surprise_positive
  - Sentiment == "negative" / "neutral" → surprise_negative
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import ftfy
from config import ROOT, CSV_FILES, WAV_DIR, PROCESSED_DIR, LABEL2IDX

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def normalize_text(value: str) -> str:
    return (
        ftfy.fix_text(value)
        .replace("’", "'")
        .replace("–", "-")
    )


def resolve_emotion(emotion: str, sentiment: str) -> str:
    if emotion == "surprise":
        return "surprise_positive" if sentiment == "positive" else "surprise_negative"
    return emotion


def build_manifest(split: str) -> pd.DataFrame:
    wav_dir = WAV_DIR / split
    if not wav_dir.exists():
        raise FileNotFoundError(
            f"{wav_dir} not found — run convert_to_wav.py first"
        )

    df = pd.read_csv(CSV_FILES[split])

    rows = []
    missing_wav = 0

    for _, row in df.iterrows():
        print(f"Processing Dialogue {row['Dialogue_ID']} Utterance {row['Utterance_ID']}...", end="\r")
        dia_id    = int(row["Dialogue_ID"])
        utt_id    = int(row["Utterance_ID"])
        emotion   = row["Emotion"].lower().strip()
        sentiment = row["Sentiment"].lower().strip()
        speaker   = normalize_text(str(row["Speaker"]))
        text      = normalize_text(str(row["Utterance"]))

        wav_path = wav_dir / f"dia{dia_id}_utt{utt_id}.wav"
        if not wav_path.exists():
            print(f"\n  [WARN] missing WAV: {wav_path.name}")
            missing_wav += 1
            continue

        label = resolve_emotion(emotion, sentiment)

        rows.append({
            "dialogue_id":  dia_id,
            "utterance_id": utt_id,
            "speaker":      speaker,
            "text":         text,
            "audio_path":   str(wav_path.relative_to(ROOT)),
            "emotion":      label,
            "label_idx":    LABEL2IDX[label],
        })

    manifest = pd.DataFrame(rows)
    out_path = PROCESSED_DIR / f"manifest_{split}.csv"
    manifest.to_csv(out_path, index=False)

    total = len(manifest) + missing_wav
    print(f"\n[{split}] {len(manifest)}/{total} utterances OK | {missing_wav} missing WAVs")
    if missing_wav > 0:
        print(f"  → Check data/wav_16k/{split}/ — did convert_to_wav.py complete?")
    print(manifest["emotion"].value_counts().to_string())
    print()
    return manifest


if __name__ == "__main__":
    for split in ("train", "dev", "test"):
        build_manifest(split)
