"""Create reproducible Phase 4 generalisation tables, figures, and analysis."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import DATA_DIR, RESULTS_DIR

ANALYSIS_DIR = RESULTS_DIR / "analysis"
CLASSICAL_DIR = RESULTS_DIR / "classical"
CNN_DIR = RESULTS_DIR / "cnn"
MODELS = ("svm", "random_forest", "knn", "cnn")


def model_title(name: str) -> str:
    return {"svm": "SVM", "random_forest": "Random Forest", "knn": "KNN", "cnn": "CNN"}[name]


def markdown_table(frame: pd.DataFrame) -> str:
    """Create a small dependency-free Markdown table for report artifacts."""
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for row in frame.itertuples(index=False, name=None):
        formatted = [f"{value:.4f}" if isinstance(value, float) else str(value) for value in row]
        rows.append("| " + " | ".join(formatted) + " |")
    return "\n".join([header, divider, *rows])


def load_predictions(name: str) -> pd.DataFrame:
    path = (CNN_DIR / "test_predictions.csv") if name == "cnn" else (CLASSICAL_DIR / f"{name}_test_predictions.csv")
    frame = pd.read_csv(path)
    required = {"emotion", "prediction", "actor"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"{path} is missing {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path} is empty")
    return frame


def generate_confusion_matrix(name: str, predictions: pd.DataFrame, labels: list[str]) -> None:
    matrix = confusion_matrix(predictions["emotion"], predictions["prediction"], labels=labels)
    figure, axis = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(matrix, display_labels=labels).plot(ax=axis, cmap="Blues", colorbar=False, xticks_rotation=45)
    axis.set_title(f"{model_title(name)}: unseen-speaker test set")
    figure.tight_layout()
    figure.savefig(ANALYSIS_DIR / f"{name}_confusion_matrix.png", dpi=220)
    plt.close(figure)


def emotion_errors(name: str, predictions: pd.DataFrame, labels: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    report = classification_report(predictions["emotion"], predictions["prediction"], labels=labels, output_dict=True, zero_division=0)
    per_emotion = pd.DataFrame(
        [
            {"model": name, "emotion": emotion, "precision": report[emotion]["precision"], "recall": report[emotion]["recall"], "f1": report[emotion]["f1-score"], "support": int(report[emotion]["support"])}
            for emotion in labels
        ]
    )
    errors = predictions.loc[predictions["emotion"] != predictions["prediction"], ["emotion", "prediction"]].value_counts().rename("count").reset_index()
    errors.insert(0, "model", name)
    return per_emotion, errors


def create_comparison_plot(comparison: pd.DataFrame) -> None:
    plot = comparison.set_index("model").loc[[model_title(name) for name in MODELS], ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]]
    plot = plot.rename(columns={"accuracy": "Accuracy", "precision_weighted": "Weighted precision", "recall_weighted": "Weighted recall", "f1_weighted": "Weighted F1"})
    axis = plot.plot(kind="bar", figsize=(10, 5.8), ylim=(0, 1), rot=0, color=["#4C78A8", "#F58518", "#54A24B", "#E45756"])
    axis.set_ylabel("Score")
    axis.set_title("Speaker-independent held-out test performance")
    axis.legend(loc="upper right")
    axis.grid(axis="y", alpha=0.25)
    axis.figure.tight_layout()
    axis.figure.savefig(ANALYSIS_DIR / "four_model_comparison.png", dpi=220)
    plt.close(axis.figure)


def create_error_plot(errors: pd.DataFrame) -> None:
    top = errors.sort_values(["model", "count"], ascending=[True, False]).groupby("model").head(3).copy()
    top["pair"] = top["emotion"] + " → " + top["prediction"]
    figure, axis = plt.subplots(figsize=(11, 6))
    for offset, (name, group) in enumerate(top.groupby("model")):
        axis.barh([f"{model_title(name)}: {pair}" for pair in group["pair"]], group["count"], label=model_title(name))
    axis.set_xlabel("Misclassified recordings")
    axis.set_title("Three most frequent emotion confusions per model")
    figure.tight_layout()
    figure.savefig(ANALYSIS_DIR / "frequent_emotion_confusions.png", dpi=220)
    plt.close(figure)


def create_grouped_cv_plot() -> None:
    summary_path = ANALYSIS_DIR / "classical_grouped_cv_summary.csv"
    if not summary_path.exists():
        return
    summary = pd.read_csv(summary_path).set_index("model").loc[["svm", "random_forest", "knn"]]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar([model_title(name) for name in summary.index], summary["f1_weighted_mean"], yerr=summary["f1_weighted_std"], capsize=5, color=["#4C78A8", "#F58518", "#54A24B"])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Weighted F1")
    axis.set_title("Six-fold grouped speaker CV (classical models)")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(ANALYSIS_DIR / "classical_grouped_cv_f1.png", dpi=220)
    plt.close(figure)


def write_analysis(comparison: pd.DataFrame, errors: pd.DataFrame, labels: list[str], cross_corpus: dict[str, str]) -> None:
    best = comparison.sort_values("f1_weighted", ascending=False).iloc[0]
    weakest = comparison.sort_values("f1_weighted", ascending=True).iloc[0]
    lines = [
        "# Phase 4: Generalisation and Experimental Analysis",
        "",
        "## Evaluation design",
        "",
        "The primary comparison is speaker-independent. Every model was evaluated on exactly 240 RAVDESS recordings from actors 01, 09, 17, and 19. These actors were absent from the 16-actor training partition and four-actor validation partition. This guards against speaker leakage.",
        "",
        "Ordinary random record-level evaluation was intentionally not run: it can put recordings from the same actor into training and test partitions and is therefore not a valid estimate of unseen-speaker performance.",
        "",
        "## Held-out comparison",
        "",
        markdown_table(comparison),
        "",
        f"By weighted F1, **{best['model']}** was best ({best['f1_weighted']:.4f}) and **{weakest['model']}** was weakest ({weakest['f1_weighted']:.4f}) on unseen speakers.",
        "",
        "The CNN was trained from scratch on log-Mel spectrograms for six CPU epochs; its result should be interpreted as a compact baseline rather than an optimized deep-learning benchmark. The classical models use fixed-size summaries of acoustic features. All results are about acted RAVDESS labels, not wellbeing or clinical states.",
        "",
        "## Emotion-specific errors",
        "",
    ]
    for name in MODELS:
        model_errors = errors.loc[errors["model"] == name].sort_values("count", ascending=False).head(3)
        pairs = "; ".join(f"{row.emotion} → {row.prediction} ({row.count})" for row in model_errors.itertuples()) or "No errors"
        lines.append(f"- **{model_title(name)}:** {pairs}.")
    lines += [
        "",
        "Per-emotion precision, recall, and F1 are stored in `emotion_level_metrics.csv`; confusion matrices should be used together with those scores because small class supports, especially neutral, can make a single metric unstable.",
        "",
        "## Grouped cross-validation",
        "",
        "A six-fold GroupKFold evaluation is provided for the three classical models. Each fold holds out whole actors. It is supplementary: hyperparameters remain the configurations selected during Phase 2, so the fixed 4-actor test comparison above remains the primary four-model comparison.",
        "",
        "## Cross-corpus generalisation",
        "",
        f"**Status: {cross_corpus['status']}.** {cross_corpus['reason']}",
        "",
        "A valid cross-corpus experiment requires a separately licensed corpus with compatible emotion labels and speaker identifiers. It must be kept external to RAVDESS model selection and reported as a distinct evaluation; no cross-corpus result is claimed here.",
        "",
        "## Phase 5 preparation",
        "",
        "Use the saved figures, tables, methodology statement, and ethical limitations to assemble the final report and Streamlit presentation. Discuss cultural and language scope (acted North American English), actor/sample limits, label ambiguity, and the unsuitability of these models for hiring, education, monitoring, or diagnosis.",
    ]
    (ANALYSIS_DIR / "EXPERIMENTAL_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    frames = {name: load_predictions(name) for name in MODELS}
    labels = sorted(set().union(*(set(frame["emotion"]) | set(frame["prediction"]) for frame in frames.values())))
    # Consistency check: four models must use identical recording identities and labels.
    baseline = frames["svm"][["path", "actor", "emotion"]].sort_values("path").reset_index(drop=True)
    for name, frame in frames.items():
        candidate = frame[["path", "actor", "emotion"]].sort_values("path").reset_index(drop=True)
        if not baseline.equals(candidate):
            raise ValueError(f"{model_title(name)} did not use the same primary held-out recordings")

    metrics_rows: list[dict[str, object]] = []
    per_emotion_frames: list[pd.DataFrame] = []
    error_frames: list[pd.DataFrame] = []
    for name, frame in frames.items():
        report = classification_report(frame["emotion"], frame["prediction"], labels=labels, output_dict=True, zero_division=0)
        metrics_rows.append({"model": model_title(name), "accuracy": report["accuracy"], "precision_weighted": report["weighted avg"]["precision"], "recall_weighted": report["weighted avg"]["recall"], "f1_weighted": report["weighted avg"]["f1-score"], "test_recordings": len(frame), "unseen_test_actors": ", ".join(map(str, sorted(frame["actor"].unique())) )})
        per_emotion, errors = emotion_errors(name, frame, labels)
        per_emotion_frames.append(per_emotion)
        error_frames.append(errors)
        generate_confusion_matrix(name, frame, labels)

    comparison = pd.DataFrame(metrics_rows)
    comparison.to_csv(ANALYSIS_DIR / "four_model_speaker_independent_comparison.csv", index=False)
    (ANALYSIS_DIR / "four_model_speaker_independent_comparison.md").write_text(markdown_table(comparison), encoding="utf-8")
    per_emotion = pd.concat(per_emotion_frames, ignore_index=True)
    errors = pd.concat(error_frames, ignore_index=True)
    per_emotion.to_csv(ANALYSIS_DIR / "emotion_level_metrics.csv", index=False)
    errors.to_csv(ANALYSIS_DIR / "frequent_emotion_confusions.csv", index=False)
    create_comparison_plot(comparison)
    create_error_plot(errors)
    create_grouped_cv_plot()

    external_audio = DATA_DIR / "external"
    cross_corpus = {
        "status": "not performed",
        "reason": "No separately labelled compatible external emotion-speech corpus is present in data/external; the project contains only RAVDESS.",
        "ordinary_random_evaluation": "not performed intentionally because it would mix speakers across partitions and inflate apparent generalisation.",
        "speaker_independent_evaluation": "completed using actors 01, 09, 17, and 19 as an untouched common test partition.",
        "external_data_directory_exists": external_audio.exists(),
    }
    (ANALYSIS_DIR / "cross_corpus_status.json").write_text(json.dumps(cross_corpus, indent=2), encoding="utf-8")
    write_analysis(comparison, errors, labels, cross_corpus)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
