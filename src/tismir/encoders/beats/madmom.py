from __future__ import annotations

from pathlib import Path

import numpy as np

from tismir.encoders.beats import beat_trackers
from tismir.encoders.beats.base import BeatTrackingResult


class MadmomBeatTracker:
    """madmom beat/downbeat tracking backend."""

    name = "madmom"

    def __init__(
        self,
        estimate_downbeats: bool = True,
        fps: int = 100,
        min_bpm: float = 55.0,
        max_bpm: float = 215.0,
        beats_per_bar: tuple[int, ...] = (3, 4),
        **_: object,
    ) -> None:
        try:
            import madmom.features.beats as beats_module
            import madmom.features.downbeats as downbeats_module
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "madmom is not installed. For modern Python versions, try "
                "`python -m pip install git+https://github.com/CPJKU/madmom.git`."
            ) from exc

        self.estimate_downbeats = estimate_downbeats
        self.fps = fps
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.beats_per_bar = tuple(beats_per_bar)

        if estimate_downbeats:
            self._activation_processor = downbeats_module.RNNDownBeatProcessor()
            self._tracking_processor = downbeats_module.DBNDownBeatTrackingProcessor(
                beats_per_bar=list(self.beats_per_bar),
                fps=fps,
                min_bpm=min_bpm,
                max_bpm=max_bpm,
            )
        else:
            self._activation_processor = beats_module.RNNBeatProcessor()
            self._tracking_processor = beats_module.DBNBeatTrackingProcessor(
                fps=fps,
                min_bpm=min_bpm,
                max_bpm=max_bpm,
            )

    def track(self, audio_path: str | Path) -> BeatTrackingResult:
        activations = self._activation_processor(str(audio_path))
        tracked = self._tracking_processor(activations)

        if self.estimate_downbeats:
            tracked = np.asarray(tracked)
            beats = tracked[:, 0].astype(np.float32)
            downbeats = tracked[tracked[:, 1] == 1, 0].astype(np.float32)
        else:
            beats = np.asarray(tracked, dtype=np.float32)
            downbeats = None

        if downbeats is not None and downbeats.size == 0:
            downbeats = None

        return BeatTrackingResult(
            beats=beats,
            downbeats=downbeats,
            metadata={
                "beat_tracker": self.name,
                "estimate_downbeats": self.estimate_downbeats,
                "fps": self.fps,
                "min_bpm": self.min_bpm,
                "max_bpm": self.max_bpm,
                "beats_per_bar": self.beats_per_bar,
            },
        )


beat_trackers.register("madmom", MadmomBeatTracker)
