"""Train and evaluate a compact actor-held-out CNN on cached RAVDESS log-Mels."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .cnn_model import build_cnn, predict_audio
from .config import LOGS_DIR, MODELS_DIR, PROCESSED_DIR, RESULTS_DIR
from .spectrograms import SPECTROGRAM_CONFIG, cache_spectrograms, load_spectrogram_tensor

SEED = 42
MODEL_DIR = MODELS_DIR / "cnn"
RESULT_DIR = RESULTS_DIR / "cnn"


def configure_logging() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(LOGS_DIR / "cnn_training.log", encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )
    return logging.getLogger(__name__)


def load_actor_split_metadata() -> pd.DataFrame:
    """Reuse the Phase 2 assignments rather than creating a new split."""
    metadata = pd.read_csv(PROCESSED_DIR / "metadata.csv")
    assignments_path = RESULTS_DIR / "classical" / "split_assignments.csv"
    if not assignments_path.exists():
        raise FileNotFoundError("Missing Phase 2 split assignments. Run: python -m src.train_classical_models")
    assignments = pd.read_csv(assignments_path)[["path", "split"]]
    data = metadata.merge(assignments, on="path", how="left", validate="one_to_one")
    if data["split"].isna().any() or len(data) != len(metadata):
        raise ValueError("Phase 2 split assignments do not cover every metadata row")
    actor_sets = {name: set(data.loc[data["split"] == name, "actor"]) for name in ("train", "validation", "test")}
    if actor_sets["train"] & actor_sets["validation"] or actor_sets["train"] & actor_sets["test"] or actor_sets["validation"] & actor_sets["test"]:
        raise ValueError("Actor leakage found in saved split assignments")
    return data


def metric_summary(y_true: np.ndarray, y_predicted: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_predicted, average="weighted", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_predicted)),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
    }


def save_history_plots(history: dict[str, list[float]]) -> None:
    for metric, title, filename in (("accuracy", "Training and validation accuracy", "training_validation_accuracy.png"), ("loss", "Training and validation loss", "training_validation_loss.png")):
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(history[metric], label="training")
        axis.plot(history[f"val_{metric}"], label="validation")
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(metric.replace("_", " ").title())
        axis.legend()
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(RESULT_DIR / filename, dpi=180)
        plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20, help="Epochs for a new run, or additional epochs with --resume (default: 20)")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size (default: 16)")
    parser.add_argument("--seed", type=int, default=SEED, help="Reproducibility seed (default: 42)")
    parser.add_argument("--rebuild-spectrograms", action="store_true", help="Regenerate cached log-Mel files")
    parser.add_argument("--resume", action="store_true", help="Continue from the saved CNN checkpoint; useful for bounded CPU sessions")
    args = parser.parse_args()

    # Set before importing TensorFlow to keep CPU execution reproducible where supported.
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    import tensorflow as tf

    logger = configure_logging()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(args.seed)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        logger.warning("TensorFlow thread configuration was already initialized")
    logger.info("TensorFlow %s; devices=%s", tf.__version__, tf.config.list_physical_devices())

    data = cache_spectrograms(load_actor_split_metadata(), rebuild=args.rebuild_spectrograms)
    labels = sorted(data["emotion"].unique().tolist())
    label_to_index = {label: index for index, label in enumerate(labels)}
    split_tensors: dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame]] = {}
    for split_name in ("train", "validation", "test"):
        subset = data.loc[data["split"] == split_name].reset_index(drop=True)
        X = load_spectrogram_tensor(subset["spectrogram_path"].tolist())
        y = subset["emotion"].map(label_to_index).to_numpy(dtype=np.int32)
        split_tensors[split_name] = (X, y, subset)
        logger.info("%s tensor=%s, actors=%s, approximate memory=%.1f MiB", split_name, X.shape, sorted(subset["actor"].unique().tolist()), X.nbytes / 1024**2)

    X_train, y_train, _ = split_tensors["train"]
    X_validation, y_validation, _ = split_tensors["validation"]
    X_test, y_test, test_metadata = split_tensors["test"]
    config = {
        "seed": args.seed,
        "labels": labels,
        "spectrogram_preprocessing": SPECTROGRAM_CONFIG,
        "input_shape": list(X_train.shape[1:]),
        "batch_size": args.batch_size,
        "requested_epochs_this_run": args.epochs,
        "split_source": str((RESULTS_DIR / "classical" / "split_assignments.csv").resolve()),
    }

    final_model_path = MODEL_DIR / "cnn_ravdess.keras"
    history_path = RESULT_DIR / "training_history.json"
    previous_history: dict[str, list[float]] = {}
    initial_epoch = 0
    if args.resume:
        if not final_model_path.exists() or not history_path.exists():
            raise FileNotFoundError("Cannot resume: saved CNN model or history is missing")
        model = tf.keras.models.load_model(final_model_path)
        previous_history = json.loads(history_path.read_text(encoding="utf-8"))
        initial_epoch = len(previous_history.get("loss", []))
        logger.info("Resuming saved CNN from epoch %s for %s additional epochs", initial_epoch, args.epochs)
    else:
        model = build_cnn(tuple(X_train.shape[1:]), len(labels), seed=args.seed)
    model.summary(print_fn=logger.info)
    checkpoint_path = MODEL_DIR / "best_cnn.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, min_delta=0.001, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_loss", mode="min", save_best_only=True),
        tf.keras.callbacks.CSVLogger(RESULT_DIR / "training_history.csv", append=args.resume),
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation),
        initial_epoch=initial_epoch,
        epochs=initial_epoch + args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
        shuffle=True,
    )
    history_data = {name: previous_history.get(name, []) + [float(value) for value in values] for name, values in history.history.items()}
    config["completed_epochs"] = len(history_data["loss"])
    (MODEL_DIR / "preprocessing_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (RESULT_DIR / "training_history.json").write_text(json.dumps(history_data, indent=2), encoding="utf-8")
    save_history_plots(history_data)

    best_model = tf.keras.models.load_model(checkpoint_path)
    best_model.save(final_model_path)
    probability_matrix = best_model.predict(X_test, batch_size=args.batch_size, verbose=0)
    prediction_indices = probability_matrix.argmax(axis=1)
    predicted_labels = np.asarray([labels[index] for index in prediction_indices])
    actual_labels = np.asarray([labels[index] for index in y_test])
    metrics = metric_summary(actual_labels, predicted_labels)
    metrics["epochs_ran"] = len(history_data["loss"])
    (RESULT_DIR / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report = classification_report(actual_labels, predicted_labels, labels=labels, zero_division=0)
    (RESULT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    predictions = test_metadata[["path", "actor", "emotion"]].copy()
    predictions["prediction"] = predicted_labels
    predictions.to_csv(RESULT_DIR / "test_predictions.csv", index=False)

    matrix = confusion_matrix(actual_labels, predicted_labels, labels=labels)
    display = ConfusionMatrixDisplay(matrix, display_labels=labels)
    figure, axis = plt.subplots(figsize=(9, 7))
    display.plot(ax=axis, cmap="Purples", colorbar=False, xticks_rotation=45)
    axis.set_title("CNN — Actor-held-out test set")
    figure.tight_layout()
    figure.savefig(RESULT_DIR / "confusion_matrix.png", dpi=180)
    plt.close(figure)

    source_prediction, source_probabilities = predict_audio(final_model_path, labels, test_metadata.iloc[0]["path"])
    inference_check = {"audio_path": test_metadata.iloc[0]["path"], "actual_emotion": test_metadata.iloc[0]["emotion"], "predicted_emotion": source_prediction, "probabilities": {label: float(probability) for label, probability in zip(labels, source_probabilities, strict=True)}}
    (RESULT_DIR / "inference_check.json").write_text(json.dumps(inference_check, indent=2), encoding="utf-8")
    logger.info("CNN test accuracy=%.4f weighted F1=%.4f; epochs=%s", metrics["accuracy"], metrics["f1_weighted"], metrics["epochs_ran"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
