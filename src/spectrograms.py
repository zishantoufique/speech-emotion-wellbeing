"""Deterministic log-Mel preprocessing shared by CNN training and inference."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import PROCESSED_DIR, SAMPLE_RATE

SPECTROGRAM_DIR = PROCESSED_DIR / "spectrograms"
SPECTROGRAM_CONFIG = {
    "sample_rate": SAMPLE_RATE,
    "duration_seconds": 4.0,
    "trim_top_db": 30,
    "n_fft": 1024,
    "hop_length": 256,
    "n_mels": 64,
    "fmax_hz": 8000,
    "normalization": "per_recording_zscore",
}


def load_audio_for_cnn(audio_path: str | Path) -> np.ndarray:
    """Load mono speech, trim silence, then pad/truncate to a fixed duration."""
    signal, _ = librosa.load(audio_path, sr=SPECTROGRAM_CONFIG["sample_rate"], mono=True)
    if signal.size == 0 or not np.isfinite(signal).all():
        raise ValueError(f"Audio is empty or non-finite: {audio_path}")
    signal, _ = librosa.effects.trim(signal, top_db=SPECTROGRAM_CONFIG["trim_top_db"])
    target_samples = int(SPECTROGRAM_CONFIG["sample_rate"] * SPECTROGRAM_CONFIG["duration_seconds"])
    return librosa.util.fix_length(signal, size=target_samples).astype(np.float32)


def signal_to_log_mel(signal: np.ndarray) -> np.ndarray:
    """Return a normalized [mel bands, frames, channel] log-Mel image."""
    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=SPECTROGRAM_CONFIG["sample_rate"],
        n_fft=SPECTROGRAM_CONFIG["n_fft"],
        hop_length=SPECTROGRAM_CONFIG["hop_length"],
        n_mels=SPECTROGRAM_CONFIG["n_mels"],
        fmax=SPECTROGRAM_CONFIG["fmax_hz"],
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    deviation = float(log_mel.std())
    if deviation < 1e-8:
        raise ValueError("Log-Mel spectrogram has near-zero variance")
    normalized = (log_mel - float(log_mel.mean())) / deviation
    if not np.isfinite(normalized).all():
        raise ValueError("Log-Mel spectrogram contains non-finite values")
    return normalized[..., np.newaxis].astype(np.float32)


def audio_to_log_mel(audio_path: str | Path) -> np.ndarray:
    return signal_to_log_mel(load_audio_for_cnn(audio_path))


def cache_spectrograms(metadata: pd.DataFrame, rebuild: bool = False) -> pd.DataFrame:
    """Create one reusable `.npy` log-Mel array per original WAV recording."""
    required = {"path", "emotion", "actor"}
    if missing := required.difference(metadata.columns):
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")
    SPECTROGRAM_DIR.mkdir(parents=True, exist_ok=True)
    cached_paths: list[str] = []
    expected_shape: tuple[int, ...] | None = None
    failures: list[str] = []
    for item in tqdm(metadata.itertuples(index=False), total=len(metadata), desc="Caching log-Mel spectrograms", unit="recording"):
        source = Path(item.path)
        target = SPECTROGRAM_DIR / f"{source.stem}.npy"
        try:
            if rebuild or not target.exists():
                array = audio_to_log_mel(source)
                np.save(target, array)
            array = np.load(target, mmap_mode="r")
            if expected_shape is None:
                expected_shape = array.shape
            if array.shape != expected_shape or not np.isfinite(array).all():
                raise ValueError(f"unexpected cached shape or values: {array.shape}")
            cached_paths.append(str(target.resolve()))
        except (OSError, ValueError) as exc:
            failures.append(f"{source}: {exc}")
    if failures:
        raise ValueError("Spectrogram cache failed for %d recordings. Examples:\n%s" % (len(failures), "\n".join(failures[:5])))
    output = metadata.copy()
    output["spectrogram_path"] = cached_paths
    return output


def load_spectrogram_tensor(paths: list[str]) -> np.ndarray:
    """Load cached arrays as a finite float32 tensor suitable for Keras."""
    tensor = np.stack([np.load(path).astype(np.float32) for path in paths])
    if tensor.ndim != 4 or not np.isfinite(tensor).all():
        raise ValueError("CNN input tensor is malformed or non-finite")
    return tensor
