"""Beat tracker interfaces and registry."""

from tismir.encoders.beats.base import BeatTracker, BeatTrackingResult
from tismir.encoders.registry import EncoderRegistry

beat_trackers: EncoderRegistry[BeatTracker] = EncoderRegistry()

__all__ = ["BeatTracker", "BeatTrackingResult", "beat_trackers"]

# Register lightweight built-in backends.
from tismir.encoders.beats import beat_this  # noqa: E402,F401
from tismir.encoders.beats import madmom  # noqa: E402,F401
from tismir.encoders.beats import uniform  # noqa: E402,F401
