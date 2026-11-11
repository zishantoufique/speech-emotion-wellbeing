"""Supplementary GroupKFold speaker-independent evaluation for classical models.

This is deliberately separate from the primary Phase 2 actor-held-out comparison.
The fixed settings were selected using the original validation actors; this script
uses those settings only for a stable grouped-CV generalisation estimate.
"""

from __future__ import annotations

import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold

from .classical_ml import load_classical_dataset
from .config import LOGS_DIR, MODELS_DIR, PROCESSED_DIR, RESULTS_DIR

RESULT_DIR = RESULTS_DIR / "analysis"
CLASSICAL_MODEL_DIR = MODELS_DIR / "classical"


def metrics(y_true: np.ndarray, y_predicted: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_predicted, average="weighted", zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_predicted)), "precision_weighted": float(precision), "recall_weighted": float(recall), "f1_weighted": float(f1)}


def main() -> int:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", handlers=[logging.FileHandler(LOGS_DIR / "grouped_cv.log", encoding="utf-8"), logging.StreamHandler()], force=True)
    logger = logging.getLogger(__name__)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    dataset, feature_columns = load_classical_dataset(PROCESSED_DIR / "metadata.csv")
    X = dataset[feature_columns].to_numpy(dtype=float)
    y = dataset["emotion"].to_numpy()
    groups = dataset["actor"].to_numpy()
    splitter = GroupKFold(n_splits=6)
    model_paths = {"svm": CLASSICAL_MODEL_DIR / "svm_pipeline.joblib", "random_forest": CLASSICAL_MODEL_DIR / "random_forest_pipeline.joblib", "knn": CLASSICAL_MODEL_DIR / "knn_pipeline.joblib"}
    rows: list[dict[str, object]] = []
    assignments: list[dict[str, object]] = []
    for model_name, model_path in model_paths.items():
        if not model_path.exists():
            raise FileNotFoundError(f"Missing trained model: {model_path}")
        template = joblib.load(model_path)
        for fold, (train_index, test_index) in enumerate(splitter.split(X, y, groups), start=1):
            train_actors, test_actors = set(groups[train_index]), set(groups[test_index])
            if train_actors & test_actors:
                raise RuntimeError(f"Actor leakage in {model_name} fold {fold}")
            model = clone(template).fit(X[train_index], y[train_index])
            predicted = model.predict(X[test_index])
            rows.append({"model": model_name, "fold": fold, "train_actors": json.dumps(sorted(int(actor) for actor in train_actors)), "test_actors": json.dumps(sorted(int(actor) for actor in test_actors)), **metrics(y[test_index], predicted)})
            assignments.extend({"model": model_name, "fold": fold, "actor": int(actor), "role": "test"} for actor in test_actors)
            logger.info("%s fold %d: train actors=%s test actors=%s accuracy=%.4f", model_name, fold, sorted(train_actors), sorted(test_actors), rows[-1]["accuracy"])

    folds = pd.DataFrame(rows)
    folds.to_csv(RESULT_DIR / "classical_grouped_cv_folds.csv", index=False)
    summary = folds.groupby("model")[["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]].agg(["mean", "std"])
    summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]
    summary.reset_index().to_csv(RESULT_DIR / "classical_grouped_cv_summary.csv", index=False)
    pd.DataFrame(assignments).to_csv(RESULT_DIR / "classical_grouped_cv_assignments.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
