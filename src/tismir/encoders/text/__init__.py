"""Text encoder registry."""

from tismir.encoders.registry import EncoderRegistry

text_encoders: EncoderRegistry = EncoderRegistry()

__all__ = ["text_encoders"]

# Register built-in backends.
from tismir.encoders.text import placeholder  # noqa: E402,F401
from tismir.encoders.text import sentence_transformers  # noqa: E402,F401
