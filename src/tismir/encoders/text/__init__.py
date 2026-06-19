"""Text encoder registry."""

from tismir.encoders.registry import EncoderRegistry

text_encoders: EncoderRegistry = EncoderRegistry()

__all__ = ["text_encoders"]
