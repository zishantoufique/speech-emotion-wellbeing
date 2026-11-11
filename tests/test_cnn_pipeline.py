"""Tests for deterministic CNN spectrogram preprocessing and architecture."""

import unittest

import numpy as np

from src.spectrograms import SPECTROGRAM_CONFIG, signal_to_log_mel


class CnnPreprocessingTests(unittest.TestCase):
    def test_log_mel_shape_and_values(self) -> None:
        samples = int(SPECTROGRAM_CONFIG["sample_rate"] * SPECTROGRAM_CONFIG["duration_seconds"])
        signal = np.sin(np.linspace(0, 200 * np.pi, samples, dtype=np.float32))
        spectrogram = signal_to_log_mel(signal)
        self.assertEqual(spectrogram.shape[0], SPECTROGRAM_CONFIG["n_mels"])
        self.assertEqual(spectrogram.shape[-1], 1)
        self.assertTrue(np.isfinite(spectrogram).all())
        self.assertAlmostEqual(float(spectrogram.mean()), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
