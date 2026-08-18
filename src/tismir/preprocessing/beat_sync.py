from __future__ import annotations

from collections.abc import Sequence

import numpy as np

#: Pooling methods supported by :func:`pool_frames_to_intervals`.
POOLING_METHODS = ("mean", "max", "energy_weighted", "multi_stat", "first", "last")

#: Statistics available to the ``multi_stat`` pooling method.
MULTI_STAT_CHOICES = ("mean", "max", "std", "min")


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

    return pool_frames_to_intervals(embeddings, times, intervals, method="mean", empty=empty)


def pool_frames_to_intervals(
    embeddings: np.ndarray,
    times: np.ndarray,
    intervals: Sequence[tuple[float, float]],
    method: str = "mean",
    empty: str = "nearest",
    *,
    temperature: float = 1.0,
    stats: Sequence[str] | None = None,
) -> np.ndarray:
    """Pool dense frame embeddings over time intervals with a selectable reduction.

    Every method shares the same interval-membership convention (frames in
    ``[start, end)``; a frame exactly on an interval end belongs to the following
    interval) and the same empty-interval policy (``empty``):

    * ``mean`` -- average of in-interval frames.
    * ``max`` -- element-wise maximum of in-interval frames.
    * ``energy_weighted`` -- weighted average with softmax weights over per-frame
      L2 norms (``weights = softmax(||f||_2 / temperature)``); an attention-like,
      parameter-free pool.
    * ``multi_stat`` -- concatenation of the statistics in ``stats`` (default
      ``["mean", "max", "std"]``), producing an output of ``dim * len(stats)``.
    * ``first`` / ``last`` -- the in-interval frame nearest the interval start /
      just before the interval end.

    ``empty`` controls behaviour for intervals with no frames: ``nearest`` uses
    the single frame closest to the interval centre, ``zeros`` emits a zero
    vector, and ``raise`` raises ``ValueError``.
    """

    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [num_frames, dim]")
    if times.ndim != 1:
        raise ValueError("times must have shape [num_frames]")
    if len(embeddings) != len(times):
        raise ValueError("embeddings and times must have the same length")
    if empty not in {"nearest", "zeros", "raise"}:
        raise ValueError("empty must be one of: nearest, zeros, raise")
    if method not in POOLING_METHODS:
        raise ValueError(f"Unsupported pooling method: {method}. Choose from {POOLING_METHODS}.")

    resolved_stats = _resolve_stats(stats) if method == "multi_stat" else None
    output_dim = embeddings.shape[1] * (len(resolved_stats) if resolved_stats else 1)

    pooled = []
    for start, end in intervals:
        mask = (times >= start) & (times < end)
        if np.any(mask):
            frames = embeddings[mask]
            frame_times = times[mask]
            pooled.append(
                _reduce(
                    frames,
                    frame_times,
                    interval=(start, end),
                    method=method,
                    temperature=temperature,
                    stats=resolved_stats,
                )
            )
            continue

        if empty == "raise":
            raise ValueError(f"No frames found in interval ({start}, {end})")
        if empty == "zeros":
            pooled.append(np.zeros(output_dim, dtype=embeddings.dtype))
        else:
            center = 0.5 * (start + end)
            idx = int(np.argmin(np.abs(times - center)))
            frame = embeddings[idx : idx + 1]
            pooled.append(
                _reduce(
                    frame,
                    times[idx : idx + 1],
                    interval=(start, end),
                    method=method,
                    temperature=temperature,
                    stats=resolved_stats,
                )
            )

    return np.stack(pooled, axis=0)


def _reduce(
    frames: np.ndarray,
    frame_times: np.ndarray,
    *,
    interval: tuple[float, float],
    method: str,
    temperature: float,
    stats: Sequence[str] | None,
) -> np.ndarray:
    """Reduce the frames belonging to a single interval to one pooled vector."""

    if method == "mean":
        return frames.mean(axis=0)
    if method == "max":
        return frames.max(axis=0)
    if method == "energy_weighted":
        return _energy_weighted_mean(frames, temperature=temperature)
    if method == "first":
        return frames[int(np.argmin(frame_times))]
    if method == "last":
        return frames[int(np.argmax(frame_times))]
    if method == "multi_stat":
        assert stats is not None
        return np.concatenate([_single_stat(frames, name) for name in stats], axis=0)
    raise ValueError(f"Unsupported pooling method: {method}")


def _energy_weighted_mean(frames: np.ndarray, *, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    energies = np.linalg.norm(frames.astype(np.float64, copy=False), axis=1) / temperature
    # Softmax with max-subtraction for numerical stability.
    weights = np.exp(energies - energies.max())
    total = weights.sum()
    if total <= 0 or not np.isfinite(total):
        weights = np.ones_like(weights)
        total = float(len(weights))
    weights = weights / total
    return (weights[:, None] * frames).sum(axis=0).astype(frames.dtype, copy=False)


def _single_stat(frames: np.ndarray, name: str) -> np.ndarray:
    if name == "mean":
        return frames.mean(axis=0)
    if name == "max":
        return frames.max(axis=0)
    if name == "min":
        return frames.min(axis=0)
    if name == "std":
        # Population std; zero for single-frame intervals.
        return frames.std(axis=0)
    raise ValueError(f"Unsupported multi_stat statistic: {name}. Choose from {MULTI_STAT_CHOICES}.")


def _resolve_stats(stats: Sequence[str] | None) -> tuple[str, ...]:
    if stats is None:
        resolved: tuple[str, ...] = ("mean", "max", "std")
    else:
        resolved = tuple(str(name) for name in stats)
    if not resolved:
        raise ValueError("multi_stat pooling requires at least one statistic")
    invalid = [name for name in resolved if name not in MULTI_STAT_CHOICES]
    if invalid:
        raise ValueError(f"Unsupported multi_stat statistics: {invalid}. Choose from {MULTI_STAT_CHOICES}.")
    return resolved
