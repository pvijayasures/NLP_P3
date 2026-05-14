"""
Phase 0: mp4 → 16kHz mono WAV (run once, before build_manifest.py).

Scans data/raw/audio/{split}/ for all mp4 files and converts them to
data/wav_16k/{split}/{stem}.wav. Already-converted files are skipped (idempotent).

Must run before build_manifest.py — the manifest links to these WAV files.

Usage:
  python -m src.preprocessing.convert_to_wav              # all splits
  python -m src.preprocessing.convert_to_wav train        # single split
  python -m src.preprocessing.convert_to_wav train dev    # multiple splits
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torchaudio
from tqdm import tqdm

from config import AUDIO_DIRS, WAV_DIR


def convert_split(split: str) -> None:
    audio_dir = AUDIO_DIRS[split]
    if not audio_dir.exists():
        print(f"[{split}] audio dir not found: {audio_dir}")
        return

    mp4_files = sorted(audio_dir.glob("*.mp4"))
    if not mp4_files:
        print(f"[{split}] no mp4 files found in {audio_dir}")
        return

    out_dir = WAV_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)

    skipped = converted = failed = 0

    for mp4_path in tqdm(mp4_files, desc=f"[{split}] converting", ncols=90):
        wav_path = out_dir / (mp4_path.stem + ".wav")

        if wav_path.exists():
            skipped += 1
            continue

        try:
            waveform, sr = torchaudio.load(str(mp4_path))
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
            torchaudio.save(str(wav_path), waveform, 16000)
            converted += 1
        except Exception as e:
            tqdm.write(f"  [WARN] failed: {mp4_path.name} — {e}")
            failed += 1

    print(f"[{split}] converted {converted} | skipped {skipped} | failed {failed}")


if __name__ == "__main__":
    splits = sys.argv[1:] if len(sys.argv) > 1 else ["train", "dev", "test"]
    for split in splits:
        convert_split(split)
