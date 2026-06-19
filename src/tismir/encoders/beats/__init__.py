"""Beat tracker interfaces and registry."""

from tismir.encoders.beats.base import BeatTracker, BeatTrackingResult
from tismir.encoders.registry import EncoderRegistry

beat_trackers: EncoderRegistry[BeatTracker] = EncoderRegistry()

__all__ = ["BeatTracker", "BeatTrackingResult", "beat_trackers"]
