from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def build_beat_intervals(beats: np.ndarray, track_duration: float) -> list[tuple[float, float]]:
    """Build beat intervals using track duration as the final boundary."""

    if beats.ndim != 1:
        raise ValueError("beats must have shape [num_beats]")
    if len(beats) == 0:
        raise ValueError("beats must not be empty")
    if np.any(np.diff(beats) <= 0):
        raise ValueError("beats must be strictly increasing")
    if track_duration <= beats[-1]:
        raise ValueError("track_duration must be greater than the final beat")

    boundaries = np.concatenate([beats, np.asarray([track_duration], dtype=beats.dtype)])
    return [(float(boundaries[i]), float(boundaries[i + 1])) for i in range(len(beats))]


def mean_pool_to_intervals(
    embeddings: np.ndarray,
    times: np.ndarray,
    intervals: Sequence[tuple[float, float]],
    empty: str = "nearest",
) -> np.ndarray:
    """Mean-pool dense frame embeddings over time intervals.

    ``times`` are interpreted as frame timestamps in seconds. Frames exactly on
    an interval end are assigned to the following interval.
    """

    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [num_frames, dim]")
    if times.ndim != 1:
        raise ValueError("times must have shape [num_frames]")
    if len(embeddings) != len(times):
        raise ValueError("embeddings and times must have the same length")
    if empty not in {"nearest", "zeros", "raise"}:
        raise ValueError("empty must be one of: nearest, zeros, raise")

    pooled = []
    for start, end in intervals:
        mask = (times >= start) & (times < end)
        if np.any(mask):
            pooled.append(embeddings[mask].mean(axis=0))
            continue

        if empty == "raise":
            raise ValueError(f"No frames found in interval ({start}, {end})")
        if empty == "zeros":
            pooled.append(np.zeros(embeddings.shape[1], dtype=embeddings.dtype))
        else:
            center = 0.5 * (start + end)
            idx = int(np.argmin(np.abs(times - center)))
            pooled.append(embeddings[idx])

    return np.stack(pooled, axis=0)
