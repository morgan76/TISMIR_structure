from __future__ import annotations

import hashlib

import numpy as np

from tismir.encoders.text import text_encoders


class PlaceholderTextEncoder:
    """Deterministic dependency-free text encoder for tests and pipeline checks."""

    name = "placeholder"

    def __init__(self, output_dim: int = 16, **_: object) -> None:
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        self.output_dim = output_dim

    def encode(self, labels: list[str]) -> np.ndarray:
        embeddings = []
        for label in labels:
            digest = hashlib.sha256(label.encode("utf-8")).digest()
            values = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
            values = np.resize(values, self.output_dim)
            values = (values / 127.5) - 1.0
            norm = np.linalg.norm(values)
            if norm > 0:
                values = values / norm
            embeddings.append(values)
        if not embeddings:
            return np.zeros((0, self.output_dim), dtype=np.float32)
        return np.stack(embeddings, axis=0).astype(np.float32)


text_encoders.register("placeholder", PlaceholderTextEncoder)
