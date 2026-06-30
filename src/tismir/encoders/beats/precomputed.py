from __future__ import annotations

from pathlib import Path

import numpy as np

from tismir.encoders.beats import beat_trackers
from tismir.encoders.beats.base import BeatTrackingResult


class PrecomputedBeatTracker:
    """Load beat/downbeat positions from precomputed ``.npy`` files."""

    name = "precomputed"

    def __init__(
        self,
        root: str | Path,
        dataset: str | None = None,
        beats_filename: str = "beats.npy",
        downbeats_filename: str = "downbeats.npy",
        **_: object,
    ) -> None:
        self.root = Path(root)
        self.dataset = dataset
        self.beats_filename = beats_filename
        self.downbeats_filename = downbeats_filename

    def track(self, audio_path: str | Path) -> BeatTrackingResult:
        audio_path = Path(audio_path)
        track_id = audio_path.stem
        track_dir = self._track_dir(track_id)
        beats_path = track_dir / self.beats_filename
        if not beats_path.exists():
            raise FileNotFoundError(f"Missing precomputed beats: {beats_path}")

        beats = np.load(beats_path).astype(np.float32, copy=False)
        downbeats_path = track_dir / self.downbeats_filename
        downbeats = None
        if downbeats_path.exists():
            downbeats = np.load(downbeats_path).astype(np.float32, copy=False)

        return BeatTrackingResult(
            beats=beats,
            downbeats=downbeats,
            metadata={
                "beat_tracker": self.name,
                "root": str(self.root),
                "dataset": self.dataset,
                "track_id": track_id,
                "beats_path": str(beats_path),
                "downbeats_path": str(downbeats_path) if downbeats_path.exists() else None,
            },
        )

    def _track_dir(self, track_id: str) -> Path:
        if self.dataset is not None:
            return self.root / self.dataset / track_id

        direct = self.root / track_id
        if direct.exists():
            return direct

        matches = sorted(self.root.glob(f"*/{track_id}"))
        if len(matches) == 1:
            return matches[0]
        if not matches:
            return direct
        raise ValueError(
            f"Multiple precomputed beat directories match track '{track_id}': {matches}"
        )


beat_trackers.register("precomputed", PrecomputedBeatTracker)
