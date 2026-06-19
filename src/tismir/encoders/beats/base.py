from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class BeatTrackingResult:
    beats: np.ndarray
    downbeats: np.ndarray | None = None
    confidence: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.beats.ndim != 1:
            raise ValueError("beats must have shape [num_beats]")
        if np.any(np.diff(self.beats) <= 0):
            raise ValueError("beats must be strictly increasing")
        if self.downbeats is not None and self.downbeats.ndim != 1:
            raise ValueError("downbeats must have shape [num_downbeats]")


class BeatTracker(Protocol):
    name: str

    def track(self, audio_path: str | Path) -> BeatTrackingResult:
        """Estimate beat and optional downbeat positions in seconds."""
