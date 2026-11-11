"""Reusable, training-matched inference functions for the Streamlit app and tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .audio_features import extract_features
from .classical_ml import predict_feature_row, summarise_feature_arrays
from .cnn_model import predict_audio
from .config import MODELS_DIR

CLASSICAL_MODELS = {"svm", "random_forest", "knn"}


def classical_prediction(model_name: str, audio_path: str | Path) -> tuple[str, dict[str, float] | None, dict[str, np.ndarray]]:
    """Predict with a saved Phase 2 pipeline and exact Phase 1 preprocessing."""
    if model_name not in CLASSICAL_MODELS:
        raise ValueError(f"Unknown classical model: {model_name}")
    model_dir = MODELS_DIR / "classical"
    schema = json.loads((model_dir / "feature_schema.json").read_text(encoding="utf-8"))
    model = joblib.load(model_dir / f"{model_name}_pipeline.joblib")
    extracted = extract_features(Path(audio_path))
    row = summarise_feature_arrays(extracted)
    label = predict_feature_row(model, row, schema["feature_columns"])
    matrix = np.array([[row[column] for column in schema["feature_columns"]]], dtype=float)
    probabilities = None
    if hasattr(model, "predict_proba"):
        values = model.predict_proba(matrix)[0]
        probabilities = {str(name): float(value) for name, value in zip(model.classes_, values, strict=True)}
    return label, probabilities, extracted


def cnn_prediction(audio_path: str | Path) -> tuple[str, dict[str, float]]:
    """Predict with the saved Phase 3 Keras model and exact log-Mel preprocessing."""
    model_dir = MODELS_DIR / "cnn"
    config = json.loads((model_dir / "preprocessing_config.json").read_text(encoding="utf-8"))
    label, probabilities = predict_audio(model_dir / "cnn_ravdess.keras", config["labels"], audio_path)
    return label, {name: float(value) for name, value in zip(config["labels"], probabilities, strict=True)}


def model_availability() -> dict[str, bool]:
    """Return whether each trained model artifact is present without loading it."""
    return {
        "svm": (MODELS_DIR / "classical" / "svm_pipeline.joblib").exists(),
        "random_forest": (MODELS_DIR / "classical" / "random_forest_pipeline.joblib").exists(),
        "knn": (MODELS_DIR / "classical" / "knn_pipeline.joblib").exists(),
        "cnn": (MODELS_DIR / "cnn" / "cnn_ravdess.keras").exists(),
    }
