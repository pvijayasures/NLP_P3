# NLP_P3
## NLP-Projekt 3

### Idee
Wir untersuchen, ob Stimmmerkmale oder Sprachinhalt besser geeignet sind, um Emotionen zu erkennen. Dafür trainieren wir drei Modelle (`Audio2Emotion`, `Text2Emotion`, Kombination aus beiden) und vergleichen den `weighted F1`-Score. Mit dem kombinierten Modell prüfen wir, ob Audio- und Textmodell komplementäre Informationen lernen und zusammen besser performen als einzeln.

### Umsetzung
- **Modell 1 (Audio → Emotion):** openSMILE eGeMAPSv02 Funktionals (88 Features) als Audiorepräsentation, normalisiert mit `StandardScaler`, klassifiziert mit `Logistic Regression`.
- **Modell 2 (Text → Emotion):** `roberta-base` direkt auf den MELD-Transkripten feinabgestimmt (End-to-End), linearer Klassifikations-Head.
- **Modell 3 (Multimodal):** eGeMAPS-Features und RoBERTa-CLS-Embedding (aus dem feinabgestimmten Modell) konkateniert → 2-Layer MLP.

### Datenquelle (MELD)
Wir verwenden den MELD-Datensatz (Friends-Dialoge, ca. 13k Utterances). Die `surprise`-Klasse wird anhand der MELD-Sentiment-Spalte in zwei Klassen aufgeteilt, sodass **8 Emotionsklassen** entstehen: `anger`, `disgust`, `sadness`, `joy`, `neutral`, `fear`, `surprise_positive`, `surprise_negative`.

- Download (in diesem Projekt genutzt): https://huggingface.co/datasets/declare-lab/MELD
- Projektseite: https://affective-meld.github.io

## Run the notebook (minimal setup)

### 1) Prerequisites
- Python `3.10` (recommended)
- `ffmpeg` installed (needed for audio processing of `.mp4` files)
- `openSMILE` installed via `pip install opensmile`

### 2) Install dependencies
From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch torchaudio
pip install -r requirements.txt
```

### 3) Download MELD data
From the project root:

```bash
mkdir -p data/raw/audio
cd data
curl -L -o MELD.Raw.tar.gz https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz
tar -xvzf MELD.Raw.tar.gz
cd MELD.Raw
tar -xvzf train.tar.gz
tar -xvzf dev.tar.gz
tar -xvzf test.tar.gz
mv train_sent_emo.csv ../raw/train_sent_emo.csv
mv dev_sent_emo.csv ../raw/dev_sent_emo.csv
mv test_sent_emo.csv ../raw/test_sent_emo.csv
mv train_splits ../raw/audio/train
mv dev_splits_complete ../raw/audio/dev
mv output_repeated_splits_test ../raw/audio/test
cd ..
cd ..
```

> **Note:** The test split contains a known filename bug — all files are prefixed with `final_videos_test` (e.g. `final_videos_testdia1_utt0.mp4` instead of `dia1_utt0.mp4`). Fix with:

```bash
for f in data/raw/audio/test/final_videos_test*.mp4; do
    mv "$f" "data/raw/audio/test/$(basename $f | sed 's/final_videos_test//')"
done
```

Windows (PowerShell):

```powershell
New-Item -ItemType Directory -Force -Path data/raw/audio | Out-Null
Set-Location data
Invoke-WebRequest -Uri "https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz" -OutFile "MELD.Raw.tar.gz"
tar -xvzf MELD.Raw.tar.gz
Set-Location MELD.Raw
tar -xvzf train.tar.gz
tar -xvzf dev.tar.gz
tar -xvzf test.tar.gz
Move-Item train_sent_emo.csv ../raw/train_sent_emo.csv
Move-Item dev_sent_emo.csv ../raw/dev_sent_emo.csv
Move-Item test_sent_emo.csv ../raw/test_sent_emo.csv
Move-Item train_splits ../raw/audio/train
Move-Item dev_splits_complete ../raw/audio/dev
Move-Item output_repeated_splits_test ../raw/audio/test
Set-Location ..
Set-Location ..
```

> **Note (Windows):** The test split contains a known filename bug — fix with:

```powershell
Get-ChildItem "data/raw/audio/test/final_videos_test*.mp4" | ForEach-Object {
    Rename-Item $_.FullName ($_.Name -replace "^final_videos_test", "")
}
```

### 4) Quick path check (optional)

```bash
ls data/raw/train_sent_emo.csv data/raw/dev_sent_emo.csv data/raw/test_sent_emo.csv
ls data/raw/audio/train data/raw/audio/dev data/raw/audio/test
```

---

## Preprocessing pipeline

Run the three steps below **in order** before opening the experiment notebook.  
Each step is idempotent — safe to re-run if interrupted.

### Step 0 — Convert mp4 → 16 kHz mono WAV

Scans `data/raw/audio/{train,dev,test}/` and writes `data/wav_16k/{split}/{stem}.wav`.  
Already-converted files are skipped automatically.

```bash
python -m src.preprocessing.convert_to_wav          # all three splits
# or a single split:
python -m src.preprocessing.convert_to_wav train
```

### Step 1 — Build manifests (CSV → parquet)

Aligns the MELD CSVs with the WAV files and applies the 8-class label mapping  
(`surprise` is split into `surprise_positive` / `surprise_negative` via the Sentiment column).  
Writes `data/processed/{split}_manifest.parquet`.

> **Requires Step 0 to have run first** — the manifest links to `data/wav_16k/`.

```bash
python -m src.preprocessing.build_manifest
```

### Step 2 — Extract openSMILE audio features

Reads the 16 kHz WAVs and saves feature vectors to `data/features/{split}_{feature_set}.parquet`.  
Failed utterances are logged to `data/features/{split}_{feature_set}_failed.csv`.

```bash
# eGeMAPS v02 — 88 features (default, fastest)
python -m src.preprocessing.extract_audio_features

# IS10 — 1582 features (slowest, most expressive)
python -m src.preprocessing.extract_audio_features is10

# emobase — 988 features
python -m src.preprocessing.extract_audio_features emobase
```

Run all three feature sets to unlock every model variant in the notebook:

```bash
for fs in egemaps is10 emobase; do
    python -m src.preprocessing.extract_audio_features $fs
done
```

---

## Run the experiment notebook

After preprocessing is complete, register the venv as a Jupyter kernel and open the notebook:

```bash
python -m ipykernel install --user --name nlp_p3 --display-name "NLP_P3 (venv)"
jupyter notebook notebook/audio_experiments.ipynb
```

Select the **NLP_P3 (venv)** kernel, then run cells top-to-bottom.  
Each training cell is independent — re-run any single cell to retrain that model.

To execute the full notebook non-interactively:

```bash
python -m jupyter nbconvert \
  --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=3600 \
  --ExecutePreprocessor.kernel_name=nlp_p3 \
  notebook/audio_experiments.ipynb
```

## Training the audio models

All three audio classifiers share the same CLI pattern:

```
python -m src.audio.<model>.train [feature_set] [--no-optimize]
```

- `feature_set` — one of `egemaps` (88 features), `is10` (1582), `emobase` (988). Defaults to `is10`.
- `--no-optimize` — skip Optuna and use the best hyperparameters already stored in `src/config.py`.

The hyperparameter search was originally conducted in `notebook/audio_experiments.ipynb`.
Best parameters per model and feature set are stored in `LR_BEST_PARAMS`, `SVM_BEST_PARAMS`,
and `MLP_BEST_PARAMS` in `src/config.py`.

### Best model — LR + IS10 *(recommended)*

The overall best model by macro F1 (0.1732 on test, 0.1809 on dev):

```bash
python -m src.audio.lr.train is10 --no-optimize
```

### Logistic Regression

```bash
# use saved best params (fast, reproducible)
python -m src.audio.lr.train is10     --no-optimize
python -m src.audio.lr.train egemaps  --no-optimize
python -m src.audio.lr.train emobase  --no-optimize

# re-run Optuna hyperparameter search (30 trials)
python -m src.audio.lr.train is10
python -m src.audio.lr.train egemaps
python -m src.audio.lr.train emobase
```

### SVM

```bash
# use saved best params (fast, reproducible)
python -m src.audio.svm.train is10     --no-optimize
python -m src.audio.svm.train egemaps  --no-optimize
python -m src.audio.svm.train emobase  --no-optimize

# re-run Optuna hyperparameter search (20 trials)
python -m src.audio.svm.train is10
python -m src.audio.svm.train egemaps
python -m src.audio.svm.train emobase
```

### MLP

```bash
# use saved best params (fast, reproducible)
python -m src.audio.mlp.train is10     --no-optimize
python -m src.audio.mlp.train egemaps  --no-optimize
python -m src.audio.mlp.train emobase  --no-optimize

# re-run Optuna hyperparameter search (20 trials)
python -m src.audio.mlp.train is10
python -m src.audio.mlp.train egemaps
python -m src.audio.mlp.train emobase
```

---

## Citation

Please cite the following papers if you find this dataset useful in your research:

S. Poria, D. Hazarika, N. Majumder, G. Naik, E. Cambria, R. Mihalcea.
*Multimodal EmotionLines: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversation.* (2018)

Chen, S.Y., Hsu, C.C., Kuo, C.C. and Ku, L.W.
*EmotionLines: An Emotion Corpus of Multi-Party Conversations.* arXiv preprint arXiv:1802.08379 (2018).