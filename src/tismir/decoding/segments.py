from __future__ import annotations

from collections.abc import Sequence

import numpy as np


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


def smooth_logits(logits: np.ndarray, window: int = 1, mode: str = "mean") -> np.ndarray:
    """Smooth frame-label logits over time."""

    if window <= 1:
        return logits
    if window % 2 == 0:
        raise ValueError("smoothing window must be odd")
    if mode not in {"mean", "median"}:
        raise ValueError("smoothing mode must be one of: mean, median")
    if logits.ndim != 2:
        raise ValueError("logits must have shape [time, labels]")

    radius = window // 2
    padded = np.pad(logits, ((radius, radius), (0, 0)), mode="edge")
    smoothed = []
    for index in range(len(logits)):
        chunk = padded[index : index + window]
        if mode == "mean":
            smoothed.append(chunk.mean(axis=0))
        else:
            smoothed.append(np.median(chunk, axis=0))
    return np.stack(smoothed, axis=0)


def remove_short_segments(
    segments: Sequence[tuple[float, float, str]],
    min_duration: float = 0.0,
) -> list[tuple[float, float, str]]:
    """Merge segments shorter than ``min_duration`` into a neighbor."""

    merged = list(segments)
    if min_duration <= 0 or len(merged) <= 1:
        return merged

    changed = True
    while changed and len(merged) > 1:
        changed = False
        for index, (start, end, label) in enumerate(merged):
            if end - start >= min_duration:
                continue
            if index == 0:
                next_start, next_end, next_label = merged[index + 1]
                merged[index + 1] = (start, next_end, next_label)
                del merged[index]
            elif index == len(merged) - 1:
                prev_start, prev_end, prev_label = merged[index - 1]
                merged[index - 1] = (prev_start, end, prev_label)
                del merged[index]
            else:
                prev_start, prev_end, prev_label = merged[index - 1]
                next_start, next_end, next_label = merged[index + 1]
                prev_duration = prev_end - prev_start
                next_duration = next_end - next_start
                if prev_label == next_label:
                    merged[index - 1 : index + 2] = [(prev_start, next_end, prev_label)]
                elif prev_duration >= next_duration:
                    merged[index - 1] = (prev_start, end, prev_label)
                    del merged[index]
                else:
                    merged[index + 1] = (start, next_end, next_label)
                    del merged[index]
            changed = True
            break

    return merge_frame_labels(
        [(start, end) for start, end, _ in merged],
        [label for _, _, label in merged],
    )
