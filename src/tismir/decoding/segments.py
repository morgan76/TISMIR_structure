from __future__ import annotations

from collections.abc import Sequence


def merge_frame_labels(
    intervals: Sequence[tuple[float, float]],
    labels: Sequence[str],
) -> list[tuple[float, float, str]]:
    """Merge consecutive intervals that have the same label."""

    if len(intervals) != len(labels):
        raise ValueError("intervals and labels must have the same length")
    if not intervals:
        return []

    merged: list[tuple[float, float, str]] = []
    current_start, current_end = intervals[0]
    current_label = labels[0]

    for (start, end), label in zip(intervals[1:], labels[1:]):
        if label == current_label:
            current_end = end
        else:
            merged.append((current_start, current_end, current_label))
            current_start, current_end, current_label = start, end, label

    merged.append((current_start, current_end, current_label))
    return merged
