from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class Track:
    """Dataset-agnostic pointer to a track and its structure annotation."""

    track_id: str
    audio_path: Path
    jams_path: Path
    dataset: str
    split: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Section:
    """A labeled time interval in seconds."""

    start: float
    end: float
    label: str
    confidence: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start
