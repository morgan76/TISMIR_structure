"""Loss functions."""

from tismir.losses.contrastive import (
    audio_audio_supervised_contrastive,
    audio_to_text_infonce,
    text_to_audio_infonce,
)
from tismir.losses.frame_label import (
    boundary_prediction_loss,
    cross_similarity_matching_loss,
    frame_label_cross_entropy,
    pairwise_probability_loss,
    pairwise_structure_relation_loss,
    pairwise_structure_loss,
    token_uniformity_loss,
)

__all__ = [
    "audio_audio_supervised_contrastive",
    "audio_to_text_infonce",
    "boundary_prediction_loss",
    "cross_similarity_matching_loss",
    "frame_label_cross_entropy",
    "pairwise_probability_loss",
    "pairwise_structure_relation_loss",
    "pairwise_structure_loss",
    "text_to_audio_infonce",
    "token_uniformity_loss",
]
