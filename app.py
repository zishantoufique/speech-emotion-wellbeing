"""Interactive educational dashboard for the trained speech-emotion models.

The app performs inference only. It never trains or overwrites model artifacts.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.predictors import classical_prediction, cnn_prediction, model_availability

ROOT = Path(__file__).parent
PROCESSED = ROOT / "data" / "processed"
ANALYSIS_RESULTS = ROOT / "results" / "analysis"
MODEL_LABELS = {"svm": "SVM", "random_forest": "Random Forest", "knn": "K-Nearest Neighbors", "cnn": "CNN (log-Mel)"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def show_waveform(audio_path: Path) -> None:
    signal, sample_rate = librosa.load(audio_path, sr=22_050, mono=True)
    times = np.arange(len(signal)) / sample_rate
    figure, axis = plt.subplots(figsize=(10, 2.8))
    axis.plot(times, signal, color="#4C78A8", linewidth=0.6)
    axis.set(title="Uploaded audio waveform", xlabel="Time (seconds)", ylabel="Amplitude")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)


def show_acoustic_features(features: dict[str, np.ndarray]) -> None:
    st.subheader("Acoustic features")
    st.caption("These are extracted with the same Phase 1 settings used to train the classical models.")
    feature_stats = pd.DataFrame({"Feature": ["Pitch (Hz)", "RMS energy", "Zero-crossing rate", "Spectral centroid (Hz)", "Spectral rolloff (Hz)", "Spectral bandwidth (Hz)"], "Mean": [float(np.mean(features["pitch_hz"][features["pitch_hz"] > 0])) if np.any(features["pitch_hz"] > 0) else 0.0, float(np.mean(features["rms_energy"])), float(np.mean(features["zero_crossing_rate"])), float(np.mean(features["spectral_centroid"])), float(np.mean(features["spectral_rolloff"])), float(np.mean(features["spectral_bandwidth"]))]})
    st.dataframe(feature_stats.style.format({"Mean": "{:.3f}"}), hide_index=True, use_container_width=True)
    st.line_chart(pd.DataFrame({"Pitch (Hz)": features["pitch_hz"], "RMS energy": features["rms_energy"]}))
    st.caption("Frame-level pitch and RMS energy")
    st.line_chart(pd.DataFrame(features["mfcc"].T, columns=[f"MFCC {index}" for index in range(1, features["mfcc"].shape[0] + 1)]))
    st.caption("Frame-level MFCC coefficients")


def prediction_panel() -> None:
    st.header("Try a trained model")
    st.caption("This is an educational prediction from acted RAVDESS-style speech. It does not diagnose wellbeing, mental health, or a person’s internal emotional state.")
    uploaded = st.file_uploader("Upload a short WAV recording", type=["wav"], help="WAV is supported to match the dataset and training pipeline.")
    available = model_availability()
    options = [name for name, exists in available.items() if exists]
    if not options:
        st.error("No trained models were found. Run the Phase 2 and Phase 3 training scripts first.")
        return
    model_name = st.selectbox("Select trained model", options, format_func=lambda name: MODEL_LABELS[name])
    if uploaded is None:
        st.info("Upload a WAV file to enable playback, feature extraction, and prediction.")
        return
    st.audio(uploaded.getvalue(), format="audio/wav")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=ROOT) as temporary:
        temporary.write(uploaded.getvalue())
        temporary_path = Path(temporary.name)
    try:
        show_waveform(temporary_path)
        if st.button("Predict emotion", type="primary"):
            with st.spinner(f"Running {MODEL_LABELS[model_name]} with its saved preprocessing pipeline..."):
                if model_name == "cnn":
                    prediction, probabilities = cnn_prediction(temporary_path)
                    features = None
                else:
                    prediction, probabilities, features = classical_prediction(model_name, temporary_path)
            st.success(f"Predicted dataset emotion label: **{prediction}**")
            if probabilities:
                st.subheader("Model confidence")
                st.bar_chart(pd.Series(probabilities).sort_values(ascending=False), horizontal=True)
                st.caption("Confidence is the model’s relative score over RAVDESS classes, not evidence of a real person’s emotional state.")
            if features is not None:
                show_acoustic_features(features)
            else:
                st.info("The CNN consumes a fixed-size log-Mel spectrogram. Select a classical model to inspect engineered acoustic features.")
    except Exception as exc:
        st.error(f"The audio could not be processed: {exc}")
    finally:
        temporary_path.unlink(missing_ok=True)


def performance_panel() -> None:
    st.header("Experimental results")
    comparison_path = ANALYSIS_RESULTS / "four_model_speaker_independent_comparison.csv"
    if not comparison_path.exists():
        st.info("Run `python -m src.run_experimental_analysis` to generate comparison artifacts.")
        return
    st.caption("All models below used the same 240 recordings from unseen actors 01, 09, 17, and 19.")
    comparison = pd.read_csv(comparison_path)
    st.dataframe(comparison.style.format({column: "{:.4f}" for column in ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]}), hide_index=True, use_container_width=True)
    for filename, caption in (("four_model_comparison.png", "Speaker-independent four-model comparison"), ("classical_grouped_cv_f1.png", "Six-fold grouped speaker CV (classical models)"), ("frequent_emotion_confusions.png", "Most frequent emotion confusions")):
        path = ANALYSIS_RESULTS / filename
        if path.exists():
            st.image(str(path), caption=caption, use_container_width=True)
    selected = st.selectbox("Choose a model confusion matrix", ["svm", "random_forest", "knn", "cnn"], format_func=lambda name: MODEL_LABELS[name], key="matrix_model")
    matrix_path = ANALYSIS_RESULTS / f"{selected}_confusion_matrix.png"
    if matrix_path.exists():
        st.image(str(matrix_path), caption=f"{MODEL_LABELS[selected]} on the unseen-speaker test set", use_container_width=True)
    status_path = ANALYSIS_RESULTS / "cross_corpus_status.json"
    if status_path.exists():
        status = read_json(status_path)
        st.warning(f"Cross-corpus evaluation: {status['status']}. {status['reason']}")


def process_panel() -> None:
    st.header("How prediction works")
    st.markdown("""1. A WAV file is loaded as mono audio at 22,050 Hz.
2. **Classical models:** MFCCs, pitch, energy, zero-crossing rate, and spectral features are extracted and summarised into the same 77 features used in training.
3. **CNN:** silence is trimmed, audio is padded/truncated to four seconds, and a normalized 64-band log-Mel spectrogram is created with the saved Phase 3 settings.
4. The selected saved model predicts one of eight RAVDESS acted emotion labels.

The app only loads existing models and performs inference. It does not retrain, tune, or modify model files.""")
    st.subheader("Limits and ethical boundary")
    st.markdown("""The system is an educational experiment using acted North American English recordings. A speech signal reflects vocal style, recording conditions, and social context—not someone’s true internal state. It must not be used for recruitment, education assessment, workplace monitoring, surveillance, diagnosis, or wellbeing decisions.""")


st.set_page_config(page_title="Speech Emotion & Wellbeing", page_icon="🎙️", layout="wide")
st.title("Speech Emotion & Wellbeing Signal Recognition")
st.caption("Educational speech emotion classification experiment • RAVDESS • actor-independent evaluation")
summary_path = PROCESSED / "dataset_summary.json"
if summary_path.exists():
    summary = read_json(summary_path)
    first, second, third = st.columns(3)
    first.metric("Validated RAVDESS recordings", summary["valid_recordings"])
    second.metric("Emotion classes", len(summary["emotion_distribution"]))
    third.metric("Unseen-speaker test recordings", 240)

tab_predict, tab_results, tab_process = st.tabs(["Prediction", "Results", "Method & ethics"])
with tab_predict:
    prediction_panel()
with tab_results:
    performance_panel()
with tab_process:
    process_panel()
