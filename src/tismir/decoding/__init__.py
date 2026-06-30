"""Segmentation decoding utilities."""

from tismir.decoding.jams import save_segments_jams
from tismir.decoding.segments import (
    boundary_times_from_intervals,
    decode_label_indices,
    merge_frame_labels,
    merge_frame_labels_with_boundary_scores,
    remove_short_segments,
    smooth_logits,
    viterbi_decode,
)

__all__ = [
    "decode_label_indices",
    "boundary_times_from_intervals",
    "merge_frame_labels",
    "merge_frame_labels_with_boundary_scores",
    "remove_short_segments",
    "save_segments_jams",
    "smooth_logits",
    "viterbi_decode",
]
