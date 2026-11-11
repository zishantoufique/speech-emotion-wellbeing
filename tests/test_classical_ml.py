"""Fast tests for Phase 2 feature, split, training, and inference utilities."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.base import clone

from src.classical_ml import build_model_specs, make_actor_splits, predict_feature_row, summarise_feature_archive


class ClassicalMlTests(unittest.TestCase):
    def test_summarise_archive(self) -> None:
        local_temp_root = Path(__file__).parent / "_temporary_test_files"
        local_temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=local_temp_root) as directory:
            path = Path(directory) / "sample.npz"
            np.savez(path, mfcc=np.ones((13, 4)), pitch_hz=np.array([0, 100, 110, 120]), rms_energy=np.ones(4), zero_crossing_rate=np.ones(4), spectral_centroid=np.ones(4), spectral_rolloff=np.ones(4), spectral_bandwidth=np.ones(4))
            row = summarise_feature_archive(path)
        self.assertEqual(len(row), 77)
        self.assertTrue(np.isfinite(list(row.values())).all())
        self.assertAlmostEqual(row["pitch_voiced_fraction"], 0.75)

    def test_actor_splits_are_disjoint(self) -> None:
        actors = np.repeat(np.arange(1, 25), 3)
        splits = make_actor_splits(actors, seed=42)
        groups = {name: set(actors[index]) for name, index in splits.items()}
        self.assertFalse(groups["train"] & groups["validation"])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["validation"] & groups["test"])
        self.assertEqual(sum(len(index) for index in splits.values()), len(actors))

    def test_train_and_infer_svm_pipeline(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.normal(size=(24, 4))
        y = np.array(["happy"] * 12 + ["sad"] * 12)
        model = clone(build_model_specs(seed=42)["svm"].estimator).set_params(model__kernel="linear", model__C=1.0).fit(X, y)
        result = predict_feature_row(model, {f"feature_{index}": float(value) for index, value in enumerate(X[0])}, [f"feature_{index}" for index in range(4)])
        self.assertIn(result, {"happy", "sad"})


if __name__ == "__main__":
    unittest.main()
