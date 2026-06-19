from __future__ import annotations

from pathlib import Path

import numpy as np

from tismir.encoders.audio import audio_encoders
from tismir.encoders.base import EmbeddingSequence
from tismir.utils.audio import get_wav_duration


class PlaceholderAudioEncoder:
    """Deterministic dependency-free audio encoder for pipeline tests.

    This backend is not musically meaningful. It exists to exercise the
    preprocessing contract before integrating foundation model encoders.
    """

    name = "placeholder"

    def __init__(self, output_dim: int = 16, frame_rate: float = 2.0, **_: object) -> None:
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if frame_rate <= 0:
            raise ValueError("frame_rate must be positive")
        self.output_dim = output_dim
        self.frame_rate = frame_rate

    def encode(self, audio_path: str | Path) -> EmbeddingSequence:
        duration = get_wav_duration(audio_path)
        hop = 1.0 / self.frame_rate
        times = np.arange(0.0, duration, hop, dtype=np.float32)
        if len(times) == 0:
            times = np.asarray([0.0], dtype=np.float32)

        normalized = times / max(duration, hop)
        features = []
        for index in range(self.output_dim):
            frequency = index // 2 + 1
            if index % 2 == 0:
                features.append(np.sin(2.0 * np.pi * frequency * normalized))
            else:
                features.append(np.cos(2.0 * np.pi * frequency * normalized))
        embeddings = np.stack(features, axis=1).astype(np.float32)

        return EmbeddingSequence(
            embeddings=embeddings,
            times=times,
            metadata={
                "encoder": self.name,
                "output_dim": self.output_dim,
                "frame_rate": self.frame_rate,
                "duration": duration,
            },
        )


audio_encoders.register("placeholder", PlaceholderAudioEncoder)
