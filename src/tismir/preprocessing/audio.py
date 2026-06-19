from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tismir.data.schemas import Track
from tismir.encoders.audio import audio_encoders
from tismir.encoders.beats import BeatTrackingResult, beat_trackers
from tismir.io import save_array, save_json
from tismir.preprocessing.beat_sync import build_beat_intervals, mean_pool_to_intervals


@dataclass(frozen=True)
class AudioPreprocessingResult:
    track_id: str
    output_dir: str
    dense_shape: tuple[int, int]
    beat_sync_shape: tuple[int, int]
    num_beats: int


def preprocess_track_audio(
    track: Track,
    output_root: str | Path,
    audio_encoder_name: str,
    audio_encoder_params: dict[str, Any] | None,
    beat_tracker_name: str,
    beat_tracker_params: dict[str, Any] | None,
    pooling: dict[str, Any] | None = None,
) -> AudioPreprocessingResult:
    """Precompute dense and beat-synchronous embeddings for one track."""

    pooling = {} if pooling is None else dict(pooling)
    method = pooling.get("method", "mean")
    if method != "mean":
        raise ValueError(f"Unsupported pooling method: {method}")

    audio_encoder = audio_encoders.build(audio_encoder_name, **(audio_encoder_params or {}))
    beat_tracker = beat_trackers.build(beat_tracker_name, **(beat_tracker_params or {}))

    dense = audio_encoder.encode(track.audio_path)
    beat_result = beat_tracker.track(track.audio_path)
    duration = float(dense.metadata.get("duration", 0.0))
    if duration <= 0:
        duration = float(beat_result.metadata.get("duration", 0.0))
    if duration <= 0:
        raise ValueError(f"Could not determine duration for {track.audio_path}")
    beat_result = _trim_beats_to_duration(beat_result, duration)

    intervals = build_beat_intervals(beat_result.beats, track_duration=duration)
    beat_sync = mean_pool_to_intervals(
        dense.embeddings,
        dense.times,
        intervals,
        empty=pooling.get("empty", "nearest"),
    )

    output_dir = Path(output_root) / audio_encoder_name / track.dataset / track.track_id
    keep_dense = bool(pooling.get("keep_dense", True))
    if keep_dense:
        save_array(output_dir / "dense.npy", dense.embeddings)
        save_array(output_dir / "dense_times.npy", dense.times)
    save_array(output_dir / "beats.npy", beat_result.beats)
    if beat_result.downbeats is not None:
        save_array(output_dir / "downbeats.npy", beat_result.downbeats)
    save_array(output_dir / "beat_sync.npy", beat_sync)

    metadata = {
        "track": {
            "track_id": track.track_id,
            "dataset": track.dataset,
            "audio_path": str(track.audio_path),
            "jams_path": str(track.jams_path),
            "split": track.split,
        },
        "audio_encoder": dense.metadata,
        "beat_tracker": beat_result.metadata,
        "pooling": {
            "method": method,
            "empty": pooling.get("empty", "nearest"),
            "keep_dense": keep_dense,
            "interval_convention": "[beat_i, beat_{i+1}), final interval ends at track duration",
        },
        "outputs": {
            "dense_shape": tuple(dense.embeddings.shape),
            "beat_sync_shape": tuple(beat_sync.shape),
            "num_beats": int(len(beat_result.beats)),
            "duration": duration,
        },
    }
    save_json(output_dir / "metadata.json", metadata)

    return AudioPreprocessingResult(
        track_id=track.track_id,
        output_dir=str(output_dir),
        dense_shape=tuple(dense.embeddings.shape),
        beat_sync_shape=tuple(beat_sync.shape),
        num_beats=int(len(beat_result.beats)),
    )


def result_to_dict(result: AudioPreprocessingResult) -> dict[str, Any]:
    return asdict(result)


def _trim_beats_to_duration(beat_result: BeatTrackingResult, duration: float) -> BeatTrackingResult:
    """Drop beat/downbeat times outside the embedding duration."""

    keep = beat_result.beats < duration
    beats = beat_result.beats[keep]
    if len(beats) == 0:
        beats = np.asarray([0.0], dtype=np.float32)

    downbeats = beat_result.downbeats
    if downbeats is not None:
        downbeats = downbeats[downbeats < duration]
        if len(downbeats) == 0:
            downbeats = None

    metadata = dict(beat_result.metadata)
    metadata["original_num_beats"] = int(len(beat_result.beats))
    metadata["num_beats_after_duration_trim"] = int(len(beats))
    metadata["duration_trim"] = duration
    return BeatTrackingResult(
        beats=beats,
        downbeats=downbeats,
        confidence=beat_result.confidence,
        metadata=metadata,
    )
