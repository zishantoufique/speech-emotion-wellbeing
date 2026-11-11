"""Reusable feature summaries, actor splits, and model helpers for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SEQUENCE_FEATURES = (
    "rms_energy",
    "zero_crossing_rate",
    "spectral_centroid",
    "spectral_rolloff",
    "spectral_bandwidth",
)


@dataclass(frozen=True)
class ModelSpec:
    """A model and a deliberately small validation tuning grid."""

    estimator: Pipeline
    parameter_grid: tuple[dict[str, Any], ...]


def _summary_statistics(values: np.ndarray) -> tuple[float, float, float, float]:
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Feature array is empty or contains non-finite values")
    return (float(values.mean()), float(values.std()), float(values.min()), float(values.max()))


def summarise_feature_arrays(features: dict[str, np.ndarray]) -> dict[str, float]:
    """Summarise Phase 1 arrays into the 77-feature schema used for training."""
    required = {"mfcc", "pitch_hz", *SEQUENCE_FEATURES}
    missing = required.difference(features)
    if missing:
        raise ValueError(f"Feature mapping is missing features: {sorted(missing)}")
    mfcc = np.asarray(features["mfcc"], dtype=np.float64)
    if mfcc.ndim != 2 or mfcc.shape[0] == 0:
        raise ValueError("Feature mapping has an invalid MFCC matrix")

    row: dict[str, float] = {}
    statistic_names = ("mean", "std", "min", "max")
    for coefficient, values in enumerate(mfcc, start=1):
        for name, value in zip(statistic_names, _summary_statistics(values), strict=True):
            row[f"mfcc_{coefficient:02d}_{name}"] = value

    pitch = np.asarray(features["pitch_hz"], dtype=np.float64).ravel()
    voiced_pitch = pitch[pitch > 0]
    if voiced_pitch.size:
        for name, value in zip(statistic_names, _summary_statistics(voiced_pitch), strict=True):
            row[f"pitch_hz_{name}"] = value
    else:
        for name in statistic_names:
            row[f"pitch_hz_{name}"] = 0.0
    row["pitch_voiced_fraction"] = float(voiced_pitch.size / pitch.size) if pitch.size else 0.0

    for feature_name in SEQUENCE_FEATURES:
        for name, value in zip(statistic_names, _summary_statistics(features[feature_name]), strict=True):
            row[f"{feature_name}_{name}"] = value
    return row


def summarise_feature_archive(feature_path: str | Path) -> dict[str, float]:
    """Convert a Phase 1 .npz archive into one fixed-size classical-ML row."""
    path = Path(feature_path)
    with np.load(path) as archive:
        return summarise_feature_arrays({name: archive[name] for name in archive.files})


def load_classical_dataset(metadata_path: str | Path) -> tuple[pd.DataFrame, list[str]]:
    """Load Phase 1 metadata and its saved features without touching raw WAVs."""
    metadata = pd.read_csv(metadata_path)
    required_columns = {"feature_path", "emotion", "actor", "path"}
    missing = required_columns.difference(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")
    if metadata.empty:
        raise ValueError("Metadata contains no recordings")

    rows: list[dict[str, object]] = []
    failures: list[str] = []
    for item in metadata.itertuples(index=False):
        try:
            features = summarise_feature_archive(item.feature_path)
            rows.append({"path": item.path, "feature_path": item.feature_path, "emotion": item.emotion, "actor": int(item.actor), **features})
        except (OSError, ValueError, KeyError) as exc:
            failures.append(f"{item.feature_path}: {exc}")
    if failures:
        preview = "\n".join(failures[:5])
        raise ValueError(f"Could not load {len(failures)} feature archives. Examples:\n{preview}")

    dataset = pd.DataFrame(rows)
    feature_columns = [column for column in dataset.columns if column not in {"path", "feature_path", "emotion", "actor"}]
    if not feature_columns or not np.isfinite(dataset[feature_columns].to_numpy(dtype=float)).all():
        raise ValueError("Classical feature table is empty or has non-finite values")
    return dataset, feature_columns


def make_actor_splits(actors: np.ndarray, seed: int = 42) -> dict[str, np.ndarray]:
    """Create disjoint 16/4/4 actor train/validation/test index splits."""
    actors = np.asarray(actors)
    unique_actors = np.unique(actors)
    if unique_actors.size < 6:
        raise ValueError("At least six distinct actors are needed for actor-based splits")
    test_actor_count = max(1, round(unique_actors.size / 6))
    outer = GroupShuffleSplit(n_splits=1, test_size=test_actor_count, random_state=seed)
    remaining_index, test_index = next(outer.split(actors, groups=actors))
    remaining_actors = actors[remaining_index]
    validation_actor_count = max(1, round(unique_actors.size / 6))
    inner = GroupShuffleSplit(n_splits=1, test_size=validation_actor_count, random_state=seed + 1)
    train_relative, validation_relative = next(inner.split(remaining_actors, groups=remaining_actors))
    splits = {
        "train": remaining_index[train_relative],
        "validation": remaining_index[validation_relative],
        "test": test_index,
    }
    actor_sets = {name: set(actors[index]) for name, index in splits.items()}
    if actor_sets["train"] & actor_sets["validation"] or actor_sets["train"] & actor_sets["test"] or actor_sets["validation"] & actor_sets["test"]:
        raise RuntimeError("Actor leakage detected while creating splits")
    return splits


def build_model_specs(seed: int = 42) -> dict[str, ModelSpec]:
    """Build reproducible models and compact tuning grids suitable for this dataset."""
    return {
        "svm": ModelSpec(
            Pipeline([("scaler", StandardScaler()), ("model", SVC(class_weight="balanced", random_state=seed))]),
            (
                {"model__kernel": "rbf", "model__C": 1.0, "model__gamma": "scale"},
                {"model__kernel": "rbf", "model__C": 3.0, "model__gamma": "scale"},
                {"model__kernel": "rbf", "model__C": 10.0, "model__gamma": "scale"},
                {"model__kernel": "linear", "model__C": 1.0},
            ),
        ),
        "random_forest": ModelSpec(
            Pipeline([("model", RandomForestClassifier(random_state=seed, n_jobs=-1, class_weight="balanced"))]),
            (
                {"model__n_estimators": 200, "model__max_depth": None, "model__min_samples_leaf": 1},
                {"model__n_estimators": 300, "model__max_depth": 20, "model__min_samples_leaf": 1},
                {"model__n_estimators": 300, "model__max_depth": None, "model__min_samples_leaf": 2},
            ),
        ),
        "knn": ModelSpec(
            Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier())]),
            (
                {"model__n_neighbors": 3, "model__weights": "distance"},
                {"model__n_neighbors": 5, "model__weights": "distance"},
                {"model__n_neighbors": 7, "model__weights": "distance"},
                {"model__n_neighbors": 5, "model__weights": "uniform"},
            ),
        ),
    }


def predict_feature_row(model: BaseEstimator, feature_row: dict[str, float], feature_columns: list[str]) -> str:
    """Run a saved pipeline on one feature dictionary with schema validation."""
    missing = set(feature_columns).difference(feature_row)
    if missing:
        raise ValueError(f"Inference row is missing features: {sorted(missing)}")
    values = np.array([[feature_row[name] for name in feature_columns]], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Inference row contains non-finite values")
    return str(model.predict(values)[0])
