from pathlib import Path

ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
RAW_DIR       = ROOT / "data" / "raw"
WAV_DIR       = ROOT / "data" / "wav_16k"
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_DIR  = ROOT / "data" / "features"
MODELS_DIR       = ROOT / "models"
OUTPUTS_DIR      = ROOT / "outputs"
CHECKPOINTS_DIR  = OUTPUTS_DIR / "checkpoints"
METRICS_DIR      = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR  = OUTPUTS_DIR / "predictions"

AUDIO_DIRS = {
    "train": RAW_DIR / "audio" / "train",
    "dev":   RAW_DIR / "audio" / "dev",
    "test":  RAW_DIR / "audio" / "test",
}
CSV_FILES = {
    "train": RAW_DIR / "train_sent_emo.csv",
    "dev":   RAW_DIR / "dev_sent_emo.csv",
    "test":  RAW_DIR / "test_sent_emo.csv",
}

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
EMOTION_LABELS = [
    "neutral", "joy", "anger",
    "sadness", "disgust", "fear",
    "surprise_positive", "surprise_negative",
]
LABEL2IDX = {label: i for i, label in enumerate(EMOTION_LABELS)}
IDX2LABEL  = {i: label for label, i in LABEL2IDX.items()}
NUM_CLASSES = len(EMOTION_LABELS)

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Audio — feature sets & model keys
# ---------------------------------------------------------------------------
FEATURE_SETS: dict[str, tuple[str, str]] = {
    "egemaps": ("eGeMAPSv02", "Functionals"),  # 88 dims
    "is10":    ("IS10",       "Functionals"),  # 1582 dims
    "emobase": ("emobase",    "Functionals"),  # 988 dims
}
DEFAULT_FEATURE_SET = "is10"
AUDIO_MODEL_KEYS    = ("lr", "svm", "mlp")

# ---------------------------------------------------------------------------
# Text — model keys & config
# ---------------------------------------------------------------------------
TEXT_MODEL_KEYS = ("roberta",)

ROBERTA_MODEL      = "roberta-base"
ROBERTA_MAX_LEN    = 128
ROBERTA_BATCH_SIZE = 64

# ---------------------------------------------------------------------------
# Multimodal — fusion variants
# ---------------------------------------------------------------------------
MULTIMODAL_MODEL_KEYS = ("late_fusion",)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_CLASSIFIER = "lr"

# ---------------------------------------------------------------------------
# Best hyperparameters per feature set (found by Optuna, metric: dev macro-F1)
# ---------------------------------------------------------------------------
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

# Populated after first text training run
ROBERTA_BEST_PARAMS: dict[str, dict] = {}

# Populated after first multimodal training run
LATE_FUSION_BEST_PARAMS: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Tag factories  →  canonical string used in all file names
#   audio:      audio_{model_key}_{feature_set}   e.g. audio_lr_is10
#   text:       text_{model_key}                  e.g. text_roberta
#   multimodal: multimodal_{variant}              e.g. multimodal_late_fusion
# ---------------------------------------------------------------------------

def audio_tag(model_key: str, feature_set: str, context: bool = False) -> str:
    tag = f"audio_{model_key}_{feature_set}"
    return f"{tag}_ctx" if context else tag

def text_tag(model_key: str) -> str:
    return f"text_{model_key}"

def multimodal_tag(variant: str) -> str:
    return f"multimodal_{variant}"

# ---------------------------------------------------------------------------
# Generic artifact path builders  (accept any tag from the factories above)
# ---------------------------------------------------------------------------

def get_scaler_path(tag: str) -> Path:
    return CHECKPOINTS_DIR / f"{tag}_scaler.pkl"

def get_hparams_path(tag: str) -> Path:
    return CHECKPOINTS_DIR / f"{tag}_hparams.json"

def get_model_path(tag: str) -> Path:
    """Returns .pt for PyTorch models (mlp, roberta, multimodal), .pkl for sklearn."""
    if any(k in tag for k in ("mlp", "roberta", "multimodal")):
        return CHECKPOINTS_DIR / f"{tag}_best.pt"
    return CHECKPOINTS_DIR / f"{tag}_model.pkl"

def get_metrics_path(tag: str) -> Path:
    return METRICS_DIR / f"{tag}.json"

def get_prediction_path(tag: str, split: str = "test") -> Path:
    return PREDICTIONS_DIR / f"{tag}_softmax_{split}.npy"
