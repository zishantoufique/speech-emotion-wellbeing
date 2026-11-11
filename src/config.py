from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAVDESS_DIR = RAW_DIR / "ravdess"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = PROCESSED_DIR / "features"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

RAVDESS_ZIP_NAME = "Audio_Speech_Actors_01-24.zip"
RAVDESS_ZIP_PATH = RAW_DIR / RAVDESS_ZIP_NAME
RAVDESS_DOWNLOAD_URL = (
    "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
)

SAMPLE_RATE = 22_050
N_MFCC = 13
HOP_LENGTH = 512

EMOTION_NAMES = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

INTENSITY_NAMES = {"01": "normal", "02": "strong"}
STATEMENT_NAMES = {"01": "kids_are_talking_by_the_door", "02": "dogs_are_sitting_by_the_door"}
