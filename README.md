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

All multimodal experiments are in `notebook/multimodal_experiments.ipynb`. The notebook runs a **comprehensive fusion sweep**: every trained audio model × RoBERTa, with **both** fusion strategies (late + feature-fusion MLP) and a final XAI section that explains the winners.

> **Requires** the preprocessing pipeline and both the audio and text experiments to have run first.

### Structure

| Section | What it does |
|---|---|
| 1 · Load & Align | Reads manifests + IS10 parquet + context-linked text |
| 2 · Audio Baseline | Trains `audio_lr_is10` if missing |
| 3 · Text Model (RoBERTa) | Checkpoint load / train, test eval, CLS extraction |
| 4 · Complementarity Analysis | Per-class F1 heatmap; ranks audio models by minority-class coverage of text weaknesses |
| 5 · Late Fusion — All Combinations | `(audio + text) / 2` for every audio model; per-class delta heatmap |
| 6 · Feature Fusion MLP — All Combinations | `concat(audio_softmax, scaled CLS) → MLP` with mini-Optuna per audio model; per-class delta + late-vs-feature comparison |
| 7 · Final Comparison | Mega-heatmap (all models × all classes) + best-of-group summary |
| 8 · XAI & Insights | LR coefficient heatmap · per-modality confusion matrices · modality ablation · audio help/hurt analysis |

### Fusion variants and naming

| Tag pattern | Strategy |
|---|---|
| `mm_late_<audio_short>` | Parameter-free 50/50 softmax average |
| `mm_ff_<audio_short>` | 2-layer MLP on `audio_softmax (8d) ‖ scaled CLS (768d) = 776-d` input, mini-Optuna (10 trials, 20 epochs) |

So `mm_late_svm_egemaps` is RoBERTa late-fused with `audio_svm_egemaps`, and `mm_ff_mlp_is10_ctx` is the feature-fusion MLP trained on top of `audio_mlp_is10_ctx`'s softmax.

### Running the notebook

```bash
jupyter notebook notebook/multimodal_experiments.ipynb
```

Select the **NLP_P3 (venv)** kernel and run cells top-to-bottom. The feature-fusion sweep (§6.2) is the only heavy cell (~30–60 min on CPU); it skips combinations whose metrics already exist, so it's incremental.

### Outputs

| File pattern | Description |
|---|---|
| `outputs/metrics/mm_late_<audio>.json` | One late-fusion result per audio model |
| `outputs/metrics/mm_ff_<audio>.json` | One feature-fusion MLP result per audio model |
| `outputs/predictions/mm_*_softmax_test.npy` | Fused softmax per variant (2610 × 8) |
| `outputs/plots/late_fusion_all_models.png` | Bar chart: fused mF1 + Δ vs text for every late-fusion combination |
| `outputs/plots/late_fusion_perclass_delta.png` | Per-class F1 delta heatmap, all late-fusion variants |
| `outputs/plots/feature_fusion_all_models.png` | Same for feature-fusion MLP |
| `outputs/plots/feature_fusion_perclass_delta.png` | Per-class delta heatmap, all FF variants |
| `outputs/plots/late_vs_feature_fusion.png` | Grouped bar: same audio model under both strategies |
| `outputs/plots/all_models_full_table.png` | Mega F1 table (audio / text / late / feature) |
| `outputs/plots/xai_lr_coefficients.png` | Top-40 acoustic features × emotion (LR coefficients) |
| `outputs/plots/xai_confusion_matrices.png` | Side-by-side row-normalised CMs (audio / text / late / FF) |
| `outputs/plots/xai_modality_ablation.png` | Per-class drop in P(true) when audio or text is replaced with its mean |
| `outputs/plots/xai_audio_help_hurt.png` | Per-class counts of utterances where audio rescued vs broke text |

---

## Audio Sentiment Results (3-class)

`notebook/audio_sentiment.ipynb` probes whether audio prosody carries **valence signal** by predicting a coarser 3-class target (negative / neutral / positive) derived from the MELD Sentiment column.

### Weighted F1 heatmap — all models × feature sets

|       | eGeMAPSv02 | IS10   | emobase |
|-------|:----------:|:------:|:-------:|
| **LR**  | 0.4678 | 0.4596 | 0.4637 |
| **SVM** | 0.4687 | 0.4632 | 0.4490 |
| **MLP** | 0.4714 | 0.4536 | **0.4752** ← best |

### Macro F1 heatmap — all models × feature sets

|       | eGeMAPSv02 | IS10   | emobase |
|-------|:----------:|:------:|:-------:|
| **LR**  | 0.4350 | 0.4333 | 0.4378 |
| **SVM** | 0.3838 | 0.3823 | 0.3654 |
| **MLP** | 0.4344 | 0.4246 | **0.4438** ← best |

> Plot: `outputs/plots/audio_sentiment_comparison.png`  
> Comparison with emotion baseline: `outputs/plots/audio_sentiment_vs_emotion.png`

### Key findings

| Aspect | Emotion (8-class) | Sentiment (3-class) |
|--------|:-----------------:|:-------------------:|
| Best wF1 | 0.3548 (SVM / egemaps) | **0.4752** (MLP / emobase) |
| Best mF1 | 0.1743 (LR / is10) | **0.4438** (MLP / emobase) |
| Random baseline | 0.125 | 0.333 |
| Hardest class | fear / disgust (F1 < 0.10) | positive (F1 = 0.34) |

All sentiment models clear the 0.33 random baseline by +0.10–0.14 mF1, confirming that **audio prosody carries real valence signal** even when it cannot reliably separate fine-grained emotions.  
The SVM macro/weighted gap (0.38 vs 0.47) reveals a neutral-class bias identical to what was seen in the 8-class task.  
LR and MLP balance predictions across classes more evenly, keeping macro F1 close to weighted F1.

---

## Results & Comparison with MELD Baseline

### Our results (test set, 8 classes)

The full fusion sweep (9 audio models × 2 strategies = 18 fused variants, plus baselines) is in the notebook. Headline numbers:

| Model | Macro F1 | Weighted F1 |
|---|---|---|
| Audio LR IS10 *(best audio standalone, mF1)* | 0.1732 | 0.2479 |
| Audio SVM eGeMAPS *(best audio standalone, wF1)* | 0.1201 | 0.3548 |
| Text RoBERTa | 0.4338 | 0.5879 |
| **Late fusion: RoBERTa + SVM eGeMAPS** *(best overall)* | **0.4535** | **0.6178** |
| Late fusion: RoBERTa + SVM emobase | 0.4525 | 0.6184 |
| Late fusion: RoBERTa + LR IS10 | 0.4432 | 0.5911 |
| Feature fusion MLP (best: `mm_ff_mlp_is10_ctx`) | 0.4205 | 0.5408 |

### Key observation: the trained MLP loses to the parameter-free average

Across every audio model, the parameter-free 50/50 late fusion **beats** the feature-fusion MLP by roughly +0.03 mF1. The best feature-fusion variant (0.4205) is actually **below text-only** (0.4338) — the MLP overfits its 776-d input on a dataset this small. See `outputs/plots/late_vs_feature_fusion.png`.

### Comparison with MELD paper (Poria et al., 2019)

The original MELD paper reports results on **7 emotion classes** (surprise not split).
Our setup uses **8 classes** (surprise split into `surprise_positive` / `surprise_negative`), making the task harder.

| Modality | MELD paper (weighted F1) | Ours (weighted F1) | Δ |
|---|---|---|---|
| Audio only | 0.4179 | 0.2479 *(LR + IS10)* | −0.170 |
| Text only | 0.5703 | **0.5879** *(RoBERTa)* | **+0.018** ✓ |
| Multimodal | **0.6025** | **0.6178** *(late fusion + SVM eGeMAPS)* | **+0.015** ✓ |

> **Note:** the prior README reported a fusion wF1 of 0.5884 (single complementarity-selected model). The comprehensive sweep surfaced that `audio_svm_egemaps` late-fused with RoBERTa actually exceeds the MELD paper's multimodal wF1 — though see the "fusion finding" note below: the gain is mostly a neutral-class calibration effect, not genuine minority-class signal.

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

The comprehensive sweep (§5–6 in the multimodal notebook) confirms — with much more nuance than the original single-pair analysis — that audio adds a real but **shallow** signal to fusion on MELD.

**1 · SVM models dominate late fusion despite the worst standalone macro F1.** `audio_svm_egemaps` scores 0.12 mF1 on its own (vs LR IS10 at 0.17), yet `mm_late_svm_egemaps` is the best fused result (0.4535 mF1 / 0.6178 wF1). Reason: LinearSVC + CalibratedClassifierCV produces a well-spread probability distribution that nudges RoBERTa's argmax in the right direction; the SVM's poor argmax accuracy doesn't matter because we're averaging softmaxes.

**2 · The SVM fusion gain is mostly a neutral-recall artefact.** XAI section §8.4 shows that audio rescues many neutral utterances that text underpredicts (neutral recall jumps 0.65 → 0.82), but the minority emotion classes (anger, sadness, disgust, fear, surprise) barely move (±0.02 F1). So the macro F1 improvement is essentially the SVM correcting a calibration bias, not adding genuine minority-class signal.

**3 · The trained MLP underperforms the average.** Every feature-fusion variant loses to its late-fusion counterpart by ~0.03 mF1, and the best FF result (0.4205) is *below* text-only (0.4338). The 776-d MLP overfits with only ~10k training utterances; the parameter-free average is the right choice for this dataset size.

**4 · The DialogueRNN gap remains the real ceiling.** The MELD paper's +0.032 wF1 fusion gain over text comes from a 2× stronger audio branch (0.42 wF1 vs our best at 0.25), not from a smarter fusion mechanism. To close that gap we'd need dialogue-level context modelling (DialogueRNN, COSMIC) or pre-trained acoustic representations (wav2vec2, HuBERT), not better classifiers on utterance-level openSMILE functionals.

### Acoustic interpretability (XAI §8.1)

LR coefficient inspection on the best LR audio model (`audio_lr_is10`) shows the top-40 most influential features per emotion (`outputs/plots/xai_lr_coefficients.png`). Patterns that emerged:
- **F0 (pitch) statistics** are the dominant signal across all high-arousal classes (anger, joy, surprise)
- **Spectral roll-off / slope** features distinguish neutral and sadness from high-arousal classes
- **Voice quality** features (jitter, shimmer) carry weight specifically for fear and sadness
- IS10's MFCC functionals contribute broadly but no single coefficient dominates — the model relies on many small weighted contributions rather than a few discriminative features.

---

## Citation

Please cite the following papers if you find this dataset useful in your research:

S. Poria, D. Hazarika, N. Majumder, G. Naik, E. Cambria, R. Mihalcea.
*Multimodal EmotionLines: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversation.* (2018)

Chen, S.Y., Hsu, C.C., Kuo, C.C. and Ku, L.W.
*EmotionLines: An Emotion Corpus of Multi-Party Conversations.* arXiv preprint arXiv:1802.08379 (2018).