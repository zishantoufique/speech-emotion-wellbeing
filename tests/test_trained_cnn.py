"""Integration test for the saved CNN model using a real RAVDESS WAV."""

import json
import importlib.util
import unittest
from pathlib import Path

import pandas as pd

from src.cnn_model import predict_audio


@unittest.skipUnless(importlib.util.find_spec("tensorflow"), "requires the isolated .venvcnn TensorFlow environment")
class TrainedCnnTests(unittest.TestCase):
    def test_saved_cnn_predicts_a_real_recording(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / "models" / "cnn" / "preprocessing_config.json").read_text(encoding="utf-8"))
        audio_path = pd.read_csv(root / "data" / "processed" / "metadata.csv").iloc[0]["path"]
        label, probabilities = predict_audio(root / "models" / "cnn" / "cnn_ravdess.keras", config["labels"], audio_path)
        self.assertIn(label, config["labels"])
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
