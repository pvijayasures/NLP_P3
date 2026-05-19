# NLP_P3
## NLP-Projekt 3

### Idee
Wir untersuchen, ob Stimmmerkmale oder Sprachinhalt besser geeignet sind, um Emotionen zu erkennen. Dafür trainieren wir drei Modelle (`Audio2Emotion`, `Text2Emotion`, Kombination aus beiden) und vergleichen den `weighted F1`-Score. Mit dem kombinierten Modell prüfen wir, ob Audio- und Textmodell komplementäre Informationen lernen und zusammen besser performen als einzeln.

### Umsetzung
- **Modell 1 (Audio → Emotion):** openSMILE eGeMAPSv02 Funktionals (88 Features) als Audiorepräsentation, normalisiert mit `StandardScaler`, klassifiziert mit `Logistic Regression`.
- **Modell 2 (Text → Emotion):** `roberta-base` direkt auf den MELD-Transkripten feinabgestimmt (End-to-End), linearer Klassifikations-Head.
- **Modell 3 (Multimodal):** Audio-Softmax (8 dims, bestes Modell per Komplementaritätsanalyse) und skaliertes RoBERTa-CLS-Embedding (768 dims) konkateniert → 2-Layer MLP (776 dims). Zusätzlich wird ein Late-Fusion-Ceiling-Check durchgeführt (gewichteter Durchschnitt der Softmax-Ausgaben beider Modalitäten).

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
python -m src.audio.<model>.train [feature_set] [--no-optimize] [--context]
```

- `feature_set` — one of `egemaps` (88 features), `is10` (1582), `emobase` (988). Defaults to `is10`.
- `--no-optimize` — skip Optuna and use the best hyperparameters already stored in `src/config.py`.
- `--context` — prepend the **previous utterance's** audio features to each utterance (doubles input dim, e.g. IS10: 1582 → 3164). Adds single-step conversational context, analogous to the `</s>` context used in the text model. Outputs are saved under a separate `_ctx` tag (e.g. `audio_lr_is10_ctx`) so non-context models are never overwritten.

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


## Training the text model

The text classification pipeline utilizes a fine-tuned `roberta-base` model. We experimented with two approaches: analyzing isolated utterances ("no context") versus appending the previous conversational utterance ("with context"). 

All text experiments, including data preprocessing, baseline training, and Optuna hyperparameter optimization, are conducted in:
`notebook/text_experiments.ipynb`

### Best model — RoBERTa + Context *(recommended)*

[cite_start]Including the previous conversational utterance significantly improved the model's ability to classify emotions[cite: 1, 3]. The overall best text model was found during Optuna hyperparameter tuning (Trial 5) using the context-appended dataset:

* [cite_start]**Macro F1:** 0.434 [cite: 6]
* [cite_start]**Weighted F1:** 0.562 [cite: 6]
* [cite_start]**Validation Accuracy:** 0.549 [cite: 5]

[cite_start]*Note on performance:* The original authors of the MELD dataset achieved a baseline weighted F1 score of 0.570 using text-only models[cite: 8]. [cite_start]Our tuned model's score of 0.562 is highly competitive [cite: 9][cite_start], especially considering we increased the difficulty of the classification task by splitting the original "surprise" emotion into `surprise_positive` and `surprise_negative` (creating an 8-class problem instead of 7)[cite: 10].

### Hyperparameters

[cite_start]The optimal hyperparameters discovered via Optuna and used for the best context model are[cite: 7]:
* **Learning Rate:** 8.2e-06
* **Train Batch Size:** 8
* **Train Epochs:** 4
* **Weight Decay:** 0.08

### Running the pipeline and extracting features

Because the text model operates inside a Jupyter Notebook, running the training and extraction pipeline is done sequentially via `notebook/text_experiments.ipynb`. 
* [cite_start]`plots/confusion_matrix_text.png`: Visual evaluation of the text model's classification accuracy and common misclassifications[cite: 11].

---

## Multimodal Experiment Notebook

All multimodal experiments — complementarity analysis, feature-fusion MLP, and late-fusion ceiling check — are in:
`notebook/multimodal_experiments.ipynb`

> **Requires** the preprocessing pipeline and both the audio and text experiments to have run first.

### Audio model selection (Section 1c)

The audio model used in the fusion is chosen automatically via a **complementarity analysis**:
each of the 9 trained audio models (LR / SVM / MLP × eGeMAPS / IS10 / emobase) is scored by
how well it covers the emotion classes where the text model is weakest.  
The score is a weighted average of the audio model's per-class F1 over the 7 **minority classes**
(neutral excluded), weighted by `1 − text_F1` per class.

The winner (`audio_mlp_egemaps`) is selected automatically and used for the rest of the pipeline.

### Feature-fusion MLP (Modell 3)

```
audio_mlp_egemaps softmax (8d) ‖ RoBERTa CLS (768d, StandardScaler)  →  776-dim  →  MLP
```

- Audio: `predict_proba` from the saved `audio_mlp_egemaps` model
- Text: `[CLS]` token of the fine-tuned RoBERTa (`last_hidden_state[:, 0, :]`), normalized
- MLP: 2-layer with BatchNorm, Dropout, FocalLoss; hyperparameters found via Optuna (20 trials)
- Artifacts: `outputs/checkpoints/multimodal_feature_fusion_best.pt`, `outputs/metrics/multimodal_feature_fusion.json`

### Late-fusion ceiling check

A performance-proportional weighted average of audio and text softmax outputs:

```
w_audio = audio_dev_mF1 / (text_dev_mF1 + audio_dev_mF1)
fused   = w_text × text_softmax + w_audio × audio_softmax
```

No training needed. If the fused result does not beat text alone, it confirms that audio and
text carry no complementary signal in MELD.  
Artifacts: `outputs/metrics/multimodal_late_fusion.json`, `outputs/predictions/multimodal_late_fusion_softmax_test.npy`

### Running the notebook

```bash
jupyter notebook notebook/multimodal_experiments.ipynb
```

Select the **NLP_P3 (venv)** kernel and run cells top-to-bottom.

**Cell order and dependencies:**

| Section | What it does | Requires |
|---|---|---|
| 1 · Load & Align | Reads manifests + IS10 parquet | preprocessing done |
| 1b · Audio Baseline | Trains `audio_lr_is10` if missing | Step 2 features |
| 1c · Complementarity | Selects best audio model | `text_roberta.json` (section 2d) |
| 2 · RoBERTa Checkpoint | Finds / trains RoBERTa | `data/models/optuna_trial_5/` |
| 2d · Evaluate RoBERTa | Saves `text_roberta.json` | Section 2 checkpoint |
| 2c · CLS Extraction | Extracts + caches CLS embeddings | Section 2 checkpoint |
| 3 · Fused Vectors | Builds 776-dim feature matrix | Sections 1 + 2c |
| 5 · Optuna | Searches MLP hyperparameters | Section 3 |
| 6 · Final Training | Trains best MLP | Section 5 |
| 7 · Evaluate | Saves fusion MLP metrics | Section 6 |
| 8b · Late Fusion | Ceiling check | Sections 3 + 2d |
| 9 · Comparison | Bar charts + full F1 table | All above |

> **Note on first run:** Section 1c requires `text_roberta.json` which is created in section 2d.
> On the very first run the analysis falls back to `audio_mlp_egemaps` (set in the config block).
> Re-run section 1c after section 2d has completed to confirm the selection.

### Outputs

| File | Description |
|---|---|
| `outputs/metrics/multimodal_feature_fusion.json` | Feature-fusion MLP test metrics |
| `outputs/metrics/multimodal_late_fusion.json` | Late-fusion ceiling check test metrics |
| `outputs/predictions/multimodal_feature_fusion_softmax_test.npy` | MLP softmax (2610 × 8) |
| `outputs/predictions/multimodal_late_fusion_softmax_test.npy` | Late-fusion softmax (2610 × 8) |
| `outputs/plots/audio_text_overlap.png` | Complementarity heatmap (absolute + delta F1) |
| `outputs/plots/comparison_fusion.png` | Macro / Weighted F1 bar charts (all models) |
| `outputs/plots/all_models_f1_table.png` | Full per-class F1 table (all 12 models) |

---

## Results & Comparison with MELD Baseline

### Our results (test set, 8 classes)

| Model | Macro F1 | Weighted F1 |
|---|---|---|
| Audio LR IS10 *(best audio)* | 0.1732 | 0.2479 |
| Audio MLP eGeMAPS | 0.1405 | 0.1067 |
| Audio SVM eGeMAPS | 0.1201 | 0.3548 |
| Text RoBERTa | 0.4338 | 0.5879 |
| **Late Fusion** *(best overall)* | **0.4371** | **0.5884** |
| Feature Fusion MLP | 0.4239 | 0.5570 |

### Comparison with MELD paper (Poria et al., 2019)

The original MELD paper reports results on **7 emotion classes** (surprise not split).
Our setup uses **8 classes** (surprise split into `surprise_positive` / `surprise_negative`), making the task harder.

| Modality | MELD paper (weighted F1) | Ours (weighted F1) | Δ |
|---|---|---|---|
| Audio only | 0.4179 | 0.2479 | −0.170 |
| Text only | 0.5703 | **0.5879** | **+0.018** ✓ |
| Multimodal | **0.6025** | 0.5884 | −0.014 |

### Why the audio gap exists

The paper's audio model outperforms ours by a large margin due to three compounding factors:

| Factor | MELD paper | Our implementation |
|---|---|---|
| **Feature set** | openSMILE **ComParE** — 6373 dims | openSMILE IS10 — 1582 dims |
| **Feature selection** | L2-based SVM selection on 6373 dims | None |
| **Model** | **DialogueRNN** — tracks speaker state across the full dialogue via 3 GRUs | LR / SVM / MLP — single utterance, no context |

DialogueRNN models the entire conversation with three stacked GRUs:
- *Global GRU* — encodes all preceding utterances from all speakers
- *Party GRU* — tracks each individual speaker's emotional state
- *Emotion GRU* — decodes the final emotion label

Our audio classifiers treat every utterance in isolation — no knowledge of who is speaking or what was said before.

### Why our text model is stronger

Despite the harder 8-class setup, our fine-tuned `roberta-base` (0.5879) outperforms the paper's text-only DialogueRNN (0.5703). Pre-trained transformer representations compensate for the lack of full conversational context; appending the previous utterance via `</s>` provides sufficient local context for emotion recognition in scripted dialogue.

### Multimodal fusion finding

Our fusion result (late fusion weighted F1: **0.5884**) nearly matches text alone (0.5879) — a gain of only +0.0005. The complementarity analysis confirms this: no audio model achieves positive delta F1 against the text model on any minority emotion class. The audio branch is too weak (weighted F1 0.25 vs paper's 0.42) to contribute meaningful complementary signal. The MELD paper's +0.032 fusion gain comes directly from having a much stronger audio model as the second branch.

---

## Citation

Please cite the following papers if you find this dataset useful in your research:

S. Poria, D. Hazarika, N. Majumder, G. Naik, E. Cambria, R. Mihalcea.
*Multimodal EmotionLines: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversation.* (2018)

Chen, S.Y., Hsu, C.C., Kuo, C.C. and Ku, L.W.
*EmotionLines: An Emotion Corpus of Multi-Party Conversations.* arXiv preprint arXiv:1802.08379 (2018).