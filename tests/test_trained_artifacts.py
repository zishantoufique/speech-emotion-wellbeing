"""Integration test for the real Phase 2 saved models and feature schema."""

import json
import unittest
from pathlib import Path

import joblib
import pandas as pd

from src.classical_ml import predict_feature_row, summarise_feature_archive


class TrainedArtifactTests(unittest.TestCase):
    def test_saved_models_predict_a_real_feature_archive(self) -> None:
        root = Path(__file__).resolve().parents[1]
        model_dir = root / "models" / "classical"
        schema = json.loads((model_dir / "feature_schema.json").read_text(encoding="utf-8"))
        metadata = pd.read_csv(root / "data" / "processed" / "metadata.csv")
        feature_row = summarise_feature_archive(metadata.iloc[0]["feature_path"])
        for model_name in ("svm", "random_forest", "knn"):
            model = joblib.load(model_dir / f"{model_name}_pipeline.joblib")
            prediction = predict_feature_row(model, feature_row, schema["feature_columns"])
            self.assertIn(prediction, schema["target_labels"])


if __name__ == "__main__":
    unittest.main()
