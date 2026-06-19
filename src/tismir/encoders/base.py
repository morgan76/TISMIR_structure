from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class EmbeddingSequence:
    """Time-indexed embedding sequence."""

    embeddings: np.ndarray
    times: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 2:
            raise ValueError("embeddings must have shape [num_frames, dim]")
        if self.times.ndim != 1:
            raise ValueError("times must have shape [num_frames]")
        if len(self.embeddings) != len(self.times):
            raise ValueError("embeddings and times must have the same length")


class AudioEncoder(Protocol):
    name: str
    output_dim: int

    def encode(self, audio_path: str | Path) -> EmbeddingSequence:
        """Encode an audio file into dense frame embeddings."""


class TextEncoder(Protocol):
    name: str
    output_dim: int

    def encode(self, labels: list[str]) -> np.ndarray:
        """Encode text labels into embeddings with shape [num_labels, dim]."""
