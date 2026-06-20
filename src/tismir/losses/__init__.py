"""Loss functions."""

from tismir.losses.contrastive import (
    audio_audio_supervised_contrastive,
    audio_to_text_infonce,
    text_to_audio_infonce,
)
from tismir.losses.frame_label import frame_label_cross_entropy

__all__ = [
    "audio_audio_supervised_contrastive",
    "audio_to_text_infonce",
    "frame_label_cross_entropy",
    "text_to_audio_infonce",
]
