from pathlib import Path

ROOT = Path(__file__).parent.parent

# Rohdaten
RAW_DIR       = ROOT / "data" / "raw"
WAV_DIR       = ROOT / "data" / "wav_16k"   # pre-converted 16kHz mono WAVs (one-time, reusable)
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
    "is10":    ("IS10",          "Functionals"),  # 1582 features — closest to MELD baseline
    "emobase": ("emobase",       "Functionals"),  # 988 features — emotion-specific
}
DEFAULT_FEATURE_SET = "is10"
DEFAULT_CLASSIFIER  = "lr"

# Best hyperparameters per feature set (found by Optuna, metric = dev macro F1)
LR_BEST_PARAMS: dict[str, dict] = {
    "is10": {
        "C": 0.0012355217597810556, "solver": "saga", "max_iter": 3000,
        "dev_mf1": 0.1809,
    },
    "egemaps": {
        "C": 0.001959526011514871,  "solver": "saga", "max_iter": 2500,
        "dev_mf1": 0.1882,
    },
    "emobase": {
        "C": 0.0014272758626179937, "solver": "saga", "max_iter": 2000,
        "dev_mf1": 0.1889,
    },
}

SVM_BEST_PARAMS: dict[str, dict] = {
    "egemaps": {
        "C": 0.7746080211942609, "tol": 7.109576294541398e-05, "max_iter": 2500,
        "dev_mf1": 0.1434,
    },
    "emobase": {
        "C": 0.0011906993082973114, "tol": 0.0033713490173047905, "max_iter": 2000,
        "dev_mf1": 0.1066,
    },
    "is10": {
        "C": 0.0012350877142584726, "tol": 0.00010064717994056096, "max_iter": 1000,
        "dev_mf1": 0.1161,
    },
}

MLP_BEST_PARAMS: dict[str, dict] = {
    "egemaps": {
        "hidden_1": 256, "hidden_2": 128,
        "dropout": 0.3292914326871103, "lr": 0.0012194982531971237,
        "focal_gamma": 1.1221794786771562,
        "dev_mf1": 0.1527,
    },
    "emobase": {
        "hidden_1": 128, "hidden_2": 32,
        "dropout": 0.3374280975221521, "lr": 0.00011831179860048076,
        "focal_gamma": 1.1422797796707243,
        "dev_mf1": 0.1639,
    },
    "is10": {
        "hidden_1": 256, "hidden_2": 64,
        "dropout": 0.3343154038810235, "lr": 0.00014066422434020742,
        "focal_gamma": 1.5057610510013644,
        "dev_mf1": 0.1555,
    },
}

# Training
RANDOM_SEED = 42


# -----------------------------------------------------------------------------
# Central artifact path helpers
# -----------------------------------------------------------------------------
MODEL_KEYS = ("lr", "svm", "mlp")


def get_model_tag(model_key: str, feature_set: str) -> str:
    """Return canonical model tag, e.g. 'audio_lr_is10'."""
    return f"audio_{model_key}_{feature_set}"


def get_model_path(model_key: str, feature_set: str):
    """Return checkpoint path for trained model weights/object."""
    model_tag = get_model_tag(model_key, feature_set)
    if model_key == "mlp":
        return CHECKPOINTS_DIR / f"{model_tag}_best.pt"
    return CHECKPOINTS_DIR / f"{model_tag}_model.pkl"


def get_scaler_path(model_key: str, feature_set: str):
    """Return checkpoint path for fitted scaler."""
    model_tag = get_model_tag(model_key, feature_set)
    return CHECKPOINTS_DIR / f"{model_tag}_scaler.pkl"


def get_hparams_path(model_key: str, feature_set: str):
    """Return checkpoint path for best hyperparameters JSON."""
    model_tag = get_model_tag(model_key, feature_set)
    return CHECKPOINTS_DIR / f"{model_tag}_hparams.json"


def get_metrics_path(model_key: str, feature_set: str):
    """Return metrics JSON path for one model/feature-set run."""
    model_tag = get_model_tag(model_key, feature_set)
    return METRICS_DIR / f"{model_tag}.json"


def get_prediction_path(model_key: str, feature_set: str, split: str = "test"):
    """Return prediction softmax npy path for a split (default: test)."""
    model_tag = get_model_tag(model_key, feature_set)
    return PREDICTIONS_DIR / f"{model_tag}_softmax_{split}.npy"
