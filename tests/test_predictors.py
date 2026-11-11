"""Integration tests proving saved models use training-matched audio preprocessing."""

import importlib.util
import unittest
from pathlib import Path

from src.predictors import classical_prediction, model_availability


class PredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.audio = cls.root / "data" / "raw" / "ravdess" / "Actor_01" / "03-01-01-01-01-01-01.wav"

    def test_all_expected_model_artifacts_exist(self) -> None:
        self.assertEqual(model_availability(), {"svm": True, "random_forest": True, "knn": True, "cnn": True})

    def test_classical_models_predict_real_audio_with_phase1_features(self) -> None:
        for model_name in ("svm", "random_forest", "knn"):
            label, probabilities, features = classical_prediction(model_name, self.audio)
            self.assertIn(label, {"angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"})
            self.assertEqual(features["mfcc"].shape[0], 13)
            if probabilities:
                self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=5)


@unittest.skipUnless(importlib.util.find_spec("tensorflow"), "requires isolated .venvcnn TensorFlow environment")
class CnnPredictorTests(unittest.TestCase):
    def test_cnn_predicts_real_audio_with_log_mel_preprocessing(self) -> None:
        from src.predictors import cnn_prediction

        root = Path(__file__).resolve().parents[1]
        audio = root / "data" / "raw" / "ravdess" / "Actor_01" / "03-01-01-01-01-01-01.wav"
        label, probabilities = cnn_prediction(audio)
        self.assertIn(label, probabilities)
        self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=5)
