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


def merge_frame_labels_with_boundary_scores(
    intervals: Sequence[tuple[float, float]],
    labels: Sequence[str],
    boundary_probabilities: Sequence[float] | np.ndarray | None,
    threshold: float = 0.5,
    mode: str = "peaks",
) -> list[tuple[float, float, str]]:
    """Merge frame labels while letting boundary scores split same-label runs."""

    if boundary_probabilities is None:
        return merge_frame_labels(intervals, labels)
    if len(intervals) != len(labels):
        raise ValueError("intervals and labels must have the same length")
    if not intervals:
        return []
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("boundary threshold must be between 0 and 1")

    boundary_probabilities = np.asarray(boundary_probabilities, dtype=float)
    if boundary_probabilities.shape != (len(intervals) - 1,):
        raise ValueError("boundary_probabilities must have shape [time - 1]")
    split_mask = _boundary_split_mask(
        boundary_probabilities,
        threshold=threshold,
        mode=mode,
    )
    boundary_times = boundary_times_from_intervals(intervals)

    merged: list[tuple[float, float, str]] = []
    current_start, current_end = intervals[0]
    current_label = labels[0]

    for index, ((start, end), label) in enumerate(zip(intervals[1:], labels[1:]), start=1):
        label_changed = label != current_label
        boundary_split = bool(split_mask[index - 1])
        if label_changed or boundary_split:
            split_time = start if label_changed else float(boundary_times[index - 1])
            split_time = min(max(split_time, current_start), end)
            if split_time > current_start:
                merged.append((current_start, split_time, current_label))
            current_start = split_time
            current_label = label
        current_end = end

    if current_end > current_start:
        merged.append((current_start, current_end, current_label))
    return merged


def boundary_times_from_intervals(
    intervals: Sequence[tuple[float, float]],
) -> np.ndarray:
    """Return midpoint times between consecutive beat-token positions."""

    if len(intervals) < 2:
        return np.asarray([], dtype=float)
    starts = np.asarray([start for start, _ in intervals], dtype=float)
    return (starts[:-1] + starts[1:]) * 0.5


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


def _boundary_split_mask(
    probabilities: np.ndarray,
    threshold: float,
    mode: str,
) -> np.ndarray:
    if mode not in {"threshold", "peaks"}:
        raise ValueError("boundary decoding mode must be one of: threshold, peaks")
    if mode == "threshold":
        return probabilities >= threshold
    mask = np.zeros(probabilities.shape, dtype=bool)
    if len(probabilities) == 1:
        mask[0] = probabilities[0] >= threshold
        return mask
    for index in range(1, len(probabilities) - 1):
        if (
            probabilities[index] >= threshold
            and probabilities[index] > probabilities[index - 1]
            and probabilities[index] > probabilities[index + 1]
        ):
            mask[index] = True
    return mask


def decode_label_indices(
    logits: np.ndarray,
    strategy: str = "argmax",
    transition_penalty: float = 0.0,
    boundary_probabilities: Sequence[float] | np.ndarray | None = None,
    boundary_weight: float = 0.0,
    boundary_eps: float = 1e-4,
) -> np.ndarray:
    """Decode a label index sequence from frame-label logits."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape [time, labels]")
    if logits.shape[0] == 0:
        return np.asarray([], dtype=int)
    if strategy == "argmax":
        return logits.argmax(axis=-1).astype(int)
    if strategy == "viterbi":
        return viterbi_decode(
            logits,
            transition_penalty=transition_penalty,
            boundary_probabilities=boundary_probabilities,
            boundary_weight=boundary_weight,
            boundary_eps=boundary_eps,
        )
    raise ValueError("strategy must be one of: argmax, viterbi")


def viterbi_decode(
    logits: np.ndarray,
    transition_penalty: float = 0.0,
    boundary_probabilities: Sequence[float] | np.ndarray | None = None,
    boundary_weight: float = 0.0,
    boundary_eps: float = 1e-4,
) -> np.ndarray:
    """Decode logits with optional boundary-probability transition priors."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape [time, labels]")
    if logits.shape[0] == 0:
        return np.asarray([], dtype=int)
    if transition_penalty < 0:
        raise ValueError("transition_penalty must be non-negative")
    if boundary_weight < 0:
        raise ValueError("boundary_weight must be non-negative")
    if not 0.0 < boundary_eps < 0.5:
        raise ValueError("boundary_eps must be between 0 and 0.5")

    scores = np.asarray(logits, dtype=np.float64)
    num_frames, num_labels = scores.shape
    boundary_probabilities = _boundary_prior_probabilities(
        boundary_probabilities,
        num_frames=num_frames,
        weight=boundary_weight,
        eps=boundary_eps,
    )
    backpointers = np.zeros((num_frames, num_labels), dtype=np.int64)
    best_scores = scores[0].copy()

    for frame_index in range(1, num_frames):
        stay_scores = best_scores
        switch_scores = best_scores - transition_penalty
        if boundary_probabilities is not None:
            boundary_probability = boundary_probabilities[frame_index - 1]
            stay_scores = stay_scores + boundary_weight * np.log1p(-boundary_probability)
            switch_scores = switch_scores + boundary_weight * np.log(boundary_probability)
        previous_scores = np.broadcast_to(switch_scores[:, np.newaxis], (num_labels, num_labels)).copy()
        previous_scores[np.arange(num_labels), np.arange(num_labels)] = stay_scores
        backpointers[frame_index] = previous_scores.argmax(axis=0)
        best_scores = previous_scores.max(axis=0) + scores[frame_index]

    path = np.zeros(num_frames, dtype=np.int64)
    path[-1] = int(best_scores.argmax())
    for frame_index in range(num_frames - 1, 0, -1):
        path[frame_index - 1] = backpointers[frame_index, path[frame_index]]
    return path


def _boundary_prior_probabilities(
    boundary_probabilities: Sequence[float] | np.ndarray | None,
    num_frames: int,
    weight: float,
    eps: float,
) -> np.ndarray | None:
    if boundary_probabilities is None or weight == 0.0:
        return None
    probabilities = np.asarray(boundary_probabilities, dtype=np.float64)
    if probabilities.shape != (num_frames - 1,):
        raise ValueError("boundary_probabilities must have shape [time - 1]")
    return np.clip(probabilities, eps, 1.0 - eps)


def remove_short_segments(
    segments: Sequence[tuple[float, float, str]],
    min_duration: float = 0.0,
    merge_same_label_neighbors: bool = True,
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

    if not merge_same_label_neighbors:
        return merged
    return merge_frame_labels(
        [(start, end) for start, end, _ in merged],
        [label for _, _, label in merged],
    )
