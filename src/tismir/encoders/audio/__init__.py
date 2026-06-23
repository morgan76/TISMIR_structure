"""Audio encoder registry."""

from tismir.encoders.registry import EncoderRegistry

audio_encoders: EncoderRegistry = EncoderRegistry()

__all__ = ["audio_encoders"]

# Register lightweight built-in backends.
from tismir.encoders.audio import dac  # noqa: E402,F401
from tismir.encoders.audio import mert  # noqa: E402,F401
from tismir.encoders.audio import placeholder  # noqa: E402,F401
