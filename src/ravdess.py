"""RAVDESS discovery, filename parsing, and audio validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import soundfile as sf

from .config import EMOTION_NAMES, INTENSITY_NAMES, STATEMENT_NAMES


@dataclass(frozen=True)
class Recording:
    path: Path
    modality: str
    vocal_channel: str
    emotion_code: str
    emotion: str
    intensity_code: str
    intensity: str
    statement_code: str
    statement: str
    repetition: int
    actor: int
    sample_rate: int
    frames: int
    duration_seconds: float

    def to_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["path"] = str(self.path)
        return row


def parse_ravdess_filename(path: Path) -> dict[str, object]:
    """Parse an official RAVDESS WAV filename and reject malformed/non-speech files."""
    parts = path.stem.split("-")
    if path.suffix.lower() != ".wav" or len(parts) != 7:
        raise ValueError("Expected a 7-part .wav RAVDESS filename")
    if parts[0] != "03" or parts[1] != "01":
        raise ValueError("Only speech modality (03-01) is supported in this project")
    modality, vocal_channel, emotion, intensity, statement, repetition, actor = parts
    if emotion not in EMOTION_NAMES:
        raise ValueError(f"Unknown emotion code: {emotion}")
    if intensity not in INTENSITY_NAMES:
        raise ValueError(f"Unknown intensity code: {intensity}")
    if statement not in STATEMENT_NAMES:
        raise ValueError(f"Unknown statement code: {statement}")
    try:
        repetition_int, actor_int = int(repetition), int(actor)
    except ValueError as exc:
        raise ValueError("Repetition and actor must be numbers") from exc
    if repetition_int not in (1, 2) or not 1 <= actor_int <= 24:
        raise ValueError("Unexpected repetition or actor identifier")
    if emotion == "01" and intensity != "01":
        raise ValueError("Neutral RAVDESS speech has only normal intensity")
    return {
        "modality": modality,
        "vocal_channel": vocal_channel,
        "emotion_code": emotion,
        "emotion": EMOTION_NAMES[emotion],
        "intensity_code": intensity,
        "intensity": INTENSITY_NAMES[intensity],
        "statement_code": statement,
        "statement": STATEMENT_NAMES[statement],
        "repetition": repetition_int,
        "actor": actor_int,
    }


def validate_recording(path: Path) -> Recording:
    """Parse metadata and ensure the WAV header is readable and non-empty."""
    metadata = parse_ravdess_filename(path)
    try:
        info = sf.info(path)
    except RuntimeError as exc:
        raise ValueError(f"Unreadable audio: {exc}") from exc
    if info.frames <= 0 or info.samplerate <= 0:
        raise ValueError("Audio contains no usable samples")
    return Recording(
        path=path.resolve(),
        sample_rate=info.samplerate,
        frames=info.frames,
        duration_seconds=round(info.frames / info.samplerate, 5),
        **metadata,
    )


def discover_wav_files(dataset_dir: Path) -> Iterator[Path]:
    yield from sorted(dataset_dir.rglob("*.wav"))

