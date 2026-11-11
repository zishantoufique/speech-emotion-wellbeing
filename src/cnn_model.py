"""Small CNN architecture and inference helper for RAVDESS log-Mel inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .spectrograms import audio_to_log_mel


def build_cnn(input_shape: tuple[int, int, int], class_count: int, seed: int = 42) -> Any:
    """Build an intentionally compact CNN from scratch; no pretrained layers."""
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.30, seed=seed),
            tf.keras.layers.Dense(class_count, activation="softmax"),
        ],
        name="ravdess_log_mel_cnn",
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def predict_audio(model_path: str | Path, labels: list[str], audio_path: str | Path) -> tuple[str, np.ndarray]:
    """Load the saved model and predict one raw WAV using training-matched preprocessing."""
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path)
    input_tensor = audio_to_log_mel(audio_path)[np.newaxis, ...]
    probabilities = model.predict(input_tensor, verbose=0)[0]
    return labels[int(np.argmax(probabilities))], probabilities
