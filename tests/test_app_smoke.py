"""Static smoke checks for the user-facing Streamlit application."""

import unittest
from pathlib import Path


class AppSmokeTests(unittest.TestCase):
    def test_app_includes_upload_and_inference_paths(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
        for required in ("st.file_uploader", "st.audio", "classical_prediction", "cnn_prediction", "show_waveform", "Experimental results"):
            self.assertIn(required, source)


if __name__ == "__main__":
    unittest.main()
