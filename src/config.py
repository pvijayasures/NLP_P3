from pathlib import Path

ROOT = Path(__file__).parent.parent

# Rohdaten
RAW_DIR       = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_DIR  = ROOT / "data" / "features"
MODELS_DIR       = ROOT / "models"
OUTPUTS_DIR      = ROOT / "outputs"
CHECKPOINTS_DIR  = OUTPUTS_DIR / "checkpoints"
METRICS_DIR      = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR  = OUTPUTS_DIR / "predictions"

# Audio-Verzeichnisse (MELD-Struktur)
AUDIO_DIRS = {
    "train": RAW_DIR / "audio" / "train",
    "dev":   RAW_DIR / "audio" / "dev",
    "test":  RAW_DIR / "audio" / "test",
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

# RoBERTa
ROBERTA_MODEL = "roberta-base"
ROBERTA_MAX_LEN = 128
ROBERTA_BATCH_SIZE = 64

# openSMILE feature sets available for extraction and training
# key → (FeatureSet name, FeatureLevel name, parquet suffix)
FEATURE_SETS: dict[str, tuple[str, str]] = {
    "egemaps": ("eGeMAPSv02",    "Functionals"),  # 88 features
    "is10":    ("IS10_paraling", "Functionals"),  # 1582 features — closest to MELD baseline
    "emobase": ("emobase",       "Functionals"),  # 988 features — emotion-specific
}
DEFAULT_FEATURE_SET = "egemaps"

# Legacy aliases kept for existing scripts
OPENSMILE_FEATURE_SET   = FEATURE_SETS[DEFAULT_FEATURE_SET][0]
OPENSMILE_FEATURE_LEVEL = FEATURE_SETS[DEFAULT_FEATURE_SET][1]

# Training
RANDOM_SEED = 42