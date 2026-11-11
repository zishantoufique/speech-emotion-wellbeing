"""Audio preprocessing and acoustic feature extraction with librosa."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from .config import HOP_LENGTH, N_MFCC, SAMPLE_RATE


def extract_features(audio_path: Path) -> dict[str, np.ndarray]:
    """Load one recording as mono and return aligned frame-level acoustic features."""
    signal, sample_rate = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    if signal.size == 0:
        raise ValueError("Audio signal is empty after loading")
    if not np.isfinite(signal).all():
        raise ValueError("Audio contains non-finite samples")

    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    pitches, magnitudes = librosa.piptrack(y=signal, sr=sample_rate, hop_length=HOP_LENGTH)
    pitch = np.zeros(pitches.shape[1], dtype=np.float32)
    voiced = magnitudes.max(axis=0) > 0
    pitch[voiced] = pitches[magnitudes.argmax(axis=0)[voiced], np.flatnonzero(voiced)]

    return {
        "mfcc": mfcc.astype(np.float32),
        "pitch_hz": pitch,
        "rms_energy": librosa.feature.rms(y=signal, hop_length=HOP_LENGTH)[0].astype(np.float32),
        "zero_crossing_rate": librosa.feature.zero_crossing_rate(signal, hop_length=HOP_LENGTH)[0].astype(np.float32),
        "spectral_centroid": librosa.feature.spectral_centroid(y=signal, sr=sample_rate, hop_length=HOP_LENGTH)[0].astype(np.float32),
        "spectral_rolloff": librosa.feature.spectral_rolloff(y=signal, sr=sample_rate, hop_length=HOP_LENGTH)[0].astype(np.float32),
        "spectral_bandwidth": librosa.feature.spectral_bandwidth(y=signal, sr=sample_rate, hop_length=HOP_LENGTH)[0].astype(np.float32),
        "sample_rate": np.array(sample_rate, dtype=np.int32),
        "duration_seconds": np.array(signal.size / sample_rate, dtype=np.float32),
    }


def save_features(output_path: Path, features: dict[str, np.ndarray]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **features)
