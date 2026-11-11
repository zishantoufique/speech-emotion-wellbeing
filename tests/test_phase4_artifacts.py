"""Integrity checks for the actual Phase 4 speaker-independent artifacts."""

import json
import unittest
from pathlib import Path

import pandas as pd


class Phase4ArtifactTests(unittest.TestCase):
    def test_all_models_share_an_unseen_speaker_test_set(self) -> None:
        root = Path(__file__).resolve().parents[1]
        comparison = pd.read_csv(root / "results" / "analysis" / "four_model_speaker_independent_comparison.csv")
        self.assertEqual(set(comparison["model"]), {"SVM", "Random Forest", "KNN", "CNN"})
        self.assertTrue((comparison["test_recordings"] == 240).all())
        self.assertTrue((comparison["unseen_test_actors"] == "1, 9, 17, 19").all())

    def test_cross_corpus_status_is_honest(self) -> None:
        root = Path(__file__).resolve().parents[1]
        status = json.loads((root / "results" / "analysis" / "cross_corpus_status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["status"], "not performed")
        self.assertIn("RAVDESS", status["reason"])


if __name__ == "__main__":
    unittest.main()
