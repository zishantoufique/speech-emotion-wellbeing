"""Validate RAVDESS speech WAVs, extract features, and write project summaries."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from .audio_features import extract_features, save_features
from .config import FEATURES_DIR, PROCESSED_DIR, RAVDESS_DIR
from .ravdess import discover_wav_files, validate_recording


def reset_feature_directory() -> None:
    if FEATURES_DIR.exists():
        shutil.rmtree(FEATURES_DIR)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep-existing", action="store_true", help="Do not clear old feature archives first")
    args = parser.parse_args()

    wav_paths = list(discover_wav_files(RAVDESS_DIR)) if RAVDESS_DIR.exists() else []
    if not wav_paths:
        print(f"No WAV files found under {RAVDESS_DIR}. Run: python -m src.download_dataset")
        return 1

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if not args.keep_existing:
        reset_feature_directory()

    valid_rows: list[dict[str, object]] = []
    invalid_rows: list[dict[str, str]] = []
    for wav_path in tqdm(wav_paths, desc="Validating and extracting", unit="recording"):
        try:
            recording = validate_recording(wav_path)
            feature_path = FEATURES_DIR / f"{wav_path.stem}.npz"
            features = extract_features(wav_path)
            save_features(feature_path, features)
            row = recording.to_dict()
            row["feature_path"] = str(feature_path.resolve())
            row["feature_frames"] = int(features["mfcc"].shape[1])
            valid_rows.append(row)
        except Exception as exc:  # Continue processing after corrupt/missing/malformed audio.
            invalid_rows.append({"path": str(wav_path.resolve()), "reason": str(exc)})

    metadata_path = PROCESSED_DIR / "metadata.csv"
    invalid_path = PROCESSED_DIR / "invalid_files.csv"
    write_csv(metadata_path, valid_rows)
    if invalid_rows:
        write_csv(invalid_path, invalid_rows)
    else:
        invalid_path.unlink(missing_ok=True)

    distribution = dict(sorted(Counter(row["emotion"] for row in valid_rows).items()))
    summary = {
        "source_directory": str(RAVDESS_DIR.resolve()),
        "raw_wav_files_discovered": len(wav_paths),
        "valid_recordings": len(valid_rows),
        "invalid_recordings": len(invalid_rows),
        "emotion_distribution": distribution,
        "metadata_file": str(metadata_path.resolve()),
        "feature_directory": str(FEATURES_DIR.resolve()),
    }
    summary_path = PROCESSED_DIR / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0 if valid_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())

