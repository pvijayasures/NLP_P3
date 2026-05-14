# NLP_P3
## NLP-Projekt 3

### Idee
Wir untersuchen, ob Stimmmerkmale oder Sprachinhalt besser geeignet sind, um Emotionen zu erkennen. Dafür trainieren wir drei Modelle (`Audio2Emotion`, `Text2Emotion`, Kombination aus beiden) und vergleichen den `weighted F1`-Score. Mit dem kombinierten Modell prüfen wir, ob Audio- und Textmodell komplementäre Informationen lernen und zusammen besser performen als einzeln.

### Umsetzung
- **Modell 1 (Audio -> Emotion):** Pretrained Audio-Modelle (z. B. `facebook/wav2vec2-base` oder `facebook/hubert-large`) als Feature-Extraktor mit Klassifikations-Head.
- **Modell 2 (Text -> Emotion):** Nutzung der MELD-Transkripte zur rein semantischen Klassifikation (z. B. `roberta-base` oder `j-hartmann/emotion-english-distilroberta-base`).
- **Modell 3 (Multimodal):** Kombination der Softmax-Outputs von Audio- und Textmodell.

### Datenquelle (MELD)
Wir verwenden den MELD-Datensatz (Friends-Dialoge, ca. 13k Utterances, 7 Emotionsklassen: `anger`, `disgust`, `sadness`, `joy`, `neutral`, `surprise`, `fear`).

- Download (in diesem Projekt genutzt): https://huggingface.co/datasets/declare-lab/MELD
- Projektseite: https://affective-meld.github.io

## Run the notebook (minimal setup)

### 1) Prerequisites
- Python `3.10` (recommended)
- `ffmpeg` installed (needed for audio cells that read `.mp4` files)

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
mv train_splits ../raw/audio/train_splits
mv dev_splits_complete ../raw/audio/dev_splits
mv output_repeated_splits_test ../raw/audio/output_repeated_splits
cd ..
cd ..
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
Move-Item train_splits ../raw/audio/train_splits
Move-Item dev_splits_complete ../raw/audio/dev_splits
Move-Item output_repeated_splits_test ../raw/audio/output_repeated_splits
Set-Location ..
Set-Location ..
```

### 4) Quick path check (optional)

```bash
ls data/raw/train_sent_emo.csv data/raw/dev_sent_emo.csv data/raw/test_sent_emo.csv
ls data/raw/audio/train_splits data/raw/audio/dev_splits data/raw/audio/output_repeated_splits
```

## Citation

Please cite the following papers if you find this dataset useful in your research:

S. Poria, D. Hazarika, N. Majumder, G. Naik, E. Cambria, R. Mihalcea.
*Multimodal EmotionLines: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversation.* (2018)

Chen, S.Y., Hsu, C.C., Kuo, C.C. and Ku, L.W.
*EmotionLines: An Emotion Corpus of Multi-Party Conversations.* arXiv preprint arXiv:1802.08379 (2018).
