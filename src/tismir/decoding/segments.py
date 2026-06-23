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


def decode_label_indices(
    logits: np.ndarray,
    strategy: str = "argmax",
    transition_penalty: float = 0.0,
) -> np.ndarray:
    """Decode a label index sequence from frame-label logits."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape [time, labels]")
    if logits.shape[0] == 0:
        return np.asarray([], dtype=int)
    if strategy == "argmax":
        return logits.argmax(axis=-1).astype(int)
    if strategy == "viterbi":
        return viterbi_decode(logits, transition_penalty=transition_penalty)
    raise ValueError("strategy must be one of: argmax, viterbi")


def viterbi_decode(logits: np.ndarray, transition_penalty: float = 0.0) -> np.ndarray:
    """Decode logits with a constant penalty for changing labels."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape [time, labels]")
    if logits.shape[0] == 0:
        return np.asarray([], dtype=int)
    if transition_penalty < 0:
        raise ValueError("transition_penalty must be non-negative")

    scores = np.asarray(logits, dtype=np.float64)
    num_frames, num_labels = scores.shape
    backpointers = np.zeros((num_frames, num_labels), dtype=np.int64)
    best_scores = scores[0].copy()

    for frame_index in range(1, num_frames):
        stay_scores = best_scores
        switch_scores = best_scores - transition_penalty
        previous_scores = np.broadcast_to(switch_scores[:, np.newaxis], (num_labels, num_labels)).copy()
        previous_scores[np.arange(num_labels), np.arange(num_labels)] = stay_scores
        backpointers[frame_index] = previous_scores.argmax(axis=0)
        best_scores = previous_scores.max(axis=0) + scores[frame_index]

    path = np.zeros(num_frames, dtype=np.int64)
    path[-1] = int(best_scores.argmax())
    for frame_index in range(num_frames - 1, 0, -1):
        path[frame_index - 1] = backpointers[frame_index, path[frame_index]]
    return path


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
