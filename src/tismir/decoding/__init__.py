"""Segmentation decoding utilities."""

from tismir.decoding.jams import save_segments_jams
from tismir.decoding.segments import (
    boundary_peak_decode_segments,
    boundary_peak_decoding_config,
    boundary_peak_split_indices,
    boundary_times_from_intervals,
    decode_boundary_peak_indices,
    decode_label_indices,
    merge_frame_labels,
    merge_frame_labels_with_boundary_scores,
    remove_short_segments,
    smooth_logits,
    viterbi_decode,
)

__all__ = [
    "boundary_peak_decode_segments",
    "boundary_peak_decoding_config",
    "boundary_peak_split_indices",
    "decode_label_indices",
    "decode_boundary_peak_indices",
    "boundary_times_from_intervals",
    "merge_frame_labels",
    "merge_frame_labels_with_boundary_scores",
    "remove_short_segments",
    "save_segments_jams",
    "smooth_logits",
    "viterbi_decode",
]
