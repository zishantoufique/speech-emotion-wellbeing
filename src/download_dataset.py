"""Download and safely extract the official RAVDESS speech archive."""

from __future__ import annotations

import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from .config import RAVDESS_DIR, RAVDESS_DOWNLOAD_URL, RAVDESS_ZIP_PATH


def manual_instructions() -> str:
    return (
        "Manual download required:\n"
        "1. Open https://zenodo.org/records/1188976\n"
        "2. Download Audio_Speech_Actors_01-24.zip\n"
        f"3. Save it as: {RAVDESS_ZIP_PATH}\n"
        "4. Run: python -m src.download_dataset\n"
    )


def safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if not target.is_relative_to(destination):
            raise ValueError(f"Unsafe ZIP member: {member.filename}")
    archive.extractall(destination)


def main() -> int:
    RAVDESS_DIR.parent.mkdir(parents=True, exist_ok=True)
    existing = list(RAVDESS_DIR.rglob("*.wav")) if RAVDESS_DIR.exists() else []
    if existing:
        print(f"Dataset already available: {len(existing)} WAV files in {RAVDESS_DIR}")
        return 0

    partial_path = RAVDESS_ZIP_PATH.with_suffix(".zip.part")
    if RAVDESS_ZIP_PATH.exists() and not zipfile.is_zipfile(RAVDESS_ZIP_PATH):
        # Retain a partial network download and resume it on the next run.
        RAVDESS_ZIP_PATH.replace(partial_path)

    if not RAVDESS_ZIP_PATH.exists():
        try:
            offset = partial_path.stat().st_size if partial_path.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            request = urllib.request.Request(RAVDESS_DOWNLOAD_URL, headers=headers)
            print(f"Downloading official RAVDESS speech archive from Zenodo (resuming at {offset:,} bytes)...\n{RAVDESS_DOWNLOAD_URL}")
            with urllib.request.urlopen(request, timeout=60) as response:
                # A server that ignores Range returns the complete archive (200), so restart safely.
                append = offset > 0 and response.status == 206
                mode = "ab" if append else "wb"
                with open(partial_path, mode) as output:
                    shutil.copyfileobj(response, output)
            partial_path.replace(RAVDESS_ZIP_PATH)
        except Exception as exc:
            print(
                f"Automatic download paused or failed: {exc}\n"
                f"A partial archive is retained at {partial_path} so a rerun can resume it.\n\n"
                f"{manual_instructions()}",
                file=sys.stderr,
            )
            return 1

    try:
        with zipfile.ZipFile(RAVDESS_ZIP_PATH) as archive:
            print(f"Extracting {len(archive.infolist())} archive entries to {RAVDESS_DIR}...")
            safe_extract(archive, RAVDESS_DIR)
    except (OSError, zipfile.BadZipFile, ValueError) as exc:
        print(f"Could not extract archive: {exc}\n\n{manual_instructions()}", file=sys.stderr)
        return 1
    print("RAVDESS speech dataset is ready. Raw audio has not been modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

