from __future__ import annotations

from pathlib import Path

import numpy as np

from tismir.encoders.beats import beat_trackers
from tismir.encoders.beats.base import BeatTrackingResult


class BeatThisTracker:
    """BeatThis backend using ``beat_this.inference.File2Beats``."""

    name = "beat_this"

    def __init__(
        self,
        checkpoint_path: str = "final0",
        device: str = "cpu",
        dbn: bool = False,
        float16: bool = False,
        **_: object,
    ) -> None:
        try:
            from beat_this.inference import File2Beats
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "BeatThis is not installed. Install it with "
                "`python -m pip install beat-this` or `python -m pip install -e '.[beat]'`."
            ) from exc

        self.checkpoint_path = checkpoint_path
        self.device = device
        self.dbn = dbn
        self.float16 = float16
        self._file_to_beats = File2Beats(
            checkpoint_path=checkpoint_path,
            device=device,
            dbn=dbn,
            float16=float16,
        )

    def track(self, audio_path: str | Path) -> BeatTrackingResult:
        beats, downbeats = self._file_to_beats(str(audio_path))
        beats = np.asarray(beats, dtype=np.float32)
        downbeats = np.asarray(downbeats, dtype=np.float32)
        if downbeats.size == 0:
            downbeats = None

        return BeatTrackingResult(
            beats=beats,
            downbeats=downbeats,
            metadata={
                "beat_tracker": self.name,
                "checkpoint_path": self.checkpoint_path,
                "device": self.device,
                "dbn": self.dbn,
                "float16": self.float16,
            },
        )


beat_trackers.register("beat_this", BeatThisTracker)
