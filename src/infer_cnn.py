"""Predict an emotion label for one WAV with the saved Phase 3 CNN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cnn_model import predict_audio
from .config import MODELS_DIR, RESULTS_DIR


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_path", type=Path, help="Path to a WAV recording")
    parser.add_argument("--save", type=Path, help="Optional JSON output path")
    args = parser.parse_args()
    config = json.loads((MODELS_DIR / "cnn" / "preprocessing_config.json").read_text(encoding="utf-8"))
    label, probabilities = predict_audio(MODELS_DIR / "cnn" / "cnn_ravdess.keras", config["labels"], args.audio_path)
    result = {"audio_path": str(args.audio_path.resolve()), "predicted_emotion": label, "probabilities": {name: float(value) for name, value in zip(config["labels"], probabilities, strict=True)}}
    text = json.dumps(result, indent=2)
    print(text)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
