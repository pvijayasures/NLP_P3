from pathlib import Path

ROOT = Path(__file__).parent.parent

# Rohdaten
RAW_DIR       = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_DIR  = ROOT / "data" / "features"
MODELS_DIR    = ROOT / "models"
OUTPUTS_DIR   = ROOT / "outputs"

# Audio-Verzeichnisse (MELD-Struktur)
AUDIO_DIRS = {
    "train": RAW_DIR / "audio" / "train_splits",
    "dev":   RAW_DIR / "audio" / "dev_splits",
    "test":  RAW_DIR / "audio" / "output_repeated_splits",
}

# CSV-Dateien
CSV_FILES = {
    "train": RAW_DIR / "train_sent_emo.csv",
    "dev":   RAW_DIR / "dev_sent_emo.csv",
    "test":  RAW_DIR / "test_sent_emo.csv",
}

# 8 Klassen (surprise aufgeteilt)
EMOTION_LABELS = [
    "neutral", "joy", "anger",
    "sadness", "disgust", "fear",
    "surprise_positive", "surprise_negative",
]
LABEL2IDX = {label: i for i, label in enumerate(EMOTION_LABELS)}
IDX2LABEL = {i: label for label, i in LABEL2IDX.items()}
NUM_CLASSES = len(EMOTION_LABELS)

# Sentiment-Schwellenwert für Surprise-Split (VADER compound score)
SURPRISE_SENTIMENT_THRESHOLD = 0.05  # >= 0.05 → positiv, sonst negativ

# RoBERTa
ROBERTA_MODEL = "roberta-base"
ROBERTA_MAX_LEN = 128
ROBERTA_BATCH_SIZE = 64

# eGeMAPS
OPENSMILE_FEATURE_SET = "eGeMAPSv02"
OPENSMILE_FEATURE_LEVEL = "Functionals"

# Training
RANDOM_SEED = 42