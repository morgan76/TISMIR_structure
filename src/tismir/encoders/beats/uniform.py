from __future__ import annotations

from pathlib import Path

import numpy as np

from tismir.encoders.beats import beat_trackers
from tismir.encoders.beats.base import BeatTrackingResult
from tismir.utils.audio import get_wav_duration


class UniformBeatTracker:
    """Uniform beat grid for development and tests."""

    name = "uniform"

    def __init__(self, beat_period: float = 0.5, estimate_downbeats: bool = True, **_: object) -> None:
        if beat_period <= 0:
            raise ValueError("beat_period must be positive")
        self.beat_period = beat_period
        self.estimate_downbeats = estimate_downbeats

    def track(self, audio_path: str | Path) -> BeatTrackingResult:
        duration = get_wav_duration(audio_path)
        beats = np.arange(0.0, duration, self.beat_period, dtype=np.float32)
        if len(beats) == 0:
            beats = np.asarray([0.0], dtype=np.float32)

        downbeats = beats[::4].copy() if self.estimate_downbeats else None
        return BeatTrackingResult(
            beats=beats,
            downbeats=downbeats,
            metadata={
                "beat_tracker": self.name,
                "beat_period": self.beat_period,
                "estimate_downbeats": self.estimate_downbeats,
                "duration": duration,
            },
        )


beat_trackers.register("uniform", UniformBeatTracker)
