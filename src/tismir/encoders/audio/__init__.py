"""Audio encoder registry."""

from tismir.encoders.registry import EncoderRegistry

audio_encoders: EncoderRegistry = EncoderRegistry()

__all__ = ["audio_encoders"]
