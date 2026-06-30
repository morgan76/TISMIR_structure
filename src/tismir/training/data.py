from __future__ import annotations

import json
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tismir.data.annotations import (
    assign_intervals_to_grid,
    assign_intervals_to_adjusted_timeline,
    concrete_annotation_processing_choices,
    silence_label_in,
    is_random_annotation_processing,
    label_base,
)
from tismir.data.filters import filter_tracks_by_annotation_content
from tismir.data.jams import load_processed_structure_sections, unique_labels
from tismir.data.manifest import load_manifest
from tismir.data.schemas import Track
from tismir.preprocessing.beat_sync import build_beat_intervals


@dataclass(frozen=True)
class TrainingExample:
    track_id: str
    dataset: str
    audio: np.ndarray
    text: np.ndarray
    targets: np.ndarray
    base_targets: np.ndarray
    segment_targets: np.ndarray
    labels: list[str]
    beat_intervals: list[tuple[float, float]]


class StructureEmbeddingDataset:
    """Load audio/text embeddings and beat-level section targets."""

    def __init__(
        self,
        manifest: str | Path,
        audio_embedding_root: str | Path,
        audio_encoder: str,
        text_embedding_root: str | Path,
        text_encoder: str,
        audio_embedding_key: str = "beat_sync",
        namespace: str = "segment_open",
        candidate_label_strategy: str = "dataset_labels",
        ignore_index: int = -100,
        annotation_processing: str | dict[str, Any] | None = None,
        beat_subsampling: bool | dict[str, Any] | None = None,
        track_filter: bool | dict[str, Any] | None = None,
    ) -> None:
        if candidate_label_strategy not in {"dataset_labels", "track_labels"}:
            raise ValueError("candidate_label_strategy must be one of: dataset_labels, track_labels")
        self.audio_embedding_root = Path(audio_embedding_root)
        self.audio_encoder = audio_encoder
        self.text_embedding_root = Path(text_embedding_root)
        self.text_encoder = text_encoder
        self.audio_embedding_key = audio_embedding_key
        self.namespace = namespace
        self.candidate_label_strategy = candidate_label_strategy
        self.ignore_index = ignore_index
        self.annotation_processing = annotation_processing
        self.beat_subsampling = beat_subsampling
        self.track_filter = track_filter
        self.tracks = filter_tracks_by_annotation_content(
            load_manifest(manifest),
            namespace=self.namespace,
            annotation_processing=self.annotation_processing,
            track_filter=self.track_filter,
        )
        if not self.tracks:
            raise ValueError("No tracks remain after applying data.track_filter")
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.tracks)

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch for deterministic per-track annotation sampling."""

        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> TrainingExample:
        track = self.tracks[index]
        annotation_processing = _resolve_annotation_processing_for_example(
            self.annotation_processing,
            epoch=self.epoch,
            index=index,
            track_id=track.track_id,
        )
        return load_training_example(
            track=track,
            audio_embedding_root=self.audio_embedding_root,
            audio_encoder=self.audio_encoder,
            text_embedding_root=self.text_embedding_root,
            text_encoder=self.text_encoder,
            audio_embedding_key=self.audio_embedding_key,
            namespace=self.namespace,
            candidate_label_strategy=self.candidate_label_strategy,
            ignore_index=self.ignore_index,
            annotation_processing=annotation_processing,
            beat_subsampling=self.beat_subsampling,
        )


def load_training_example(
    track: Track,
    audio_embedding_root: str | Path,
    audio_encoder: str,
    text_embedding_root: str | Path,
    text_encoder: str,
    audio_embedding_key: str = "beat_sync",
    namespace: str = "segment_open",
    candidate_label_strategy: str = "dataset_labels",
    ignore_index: int = -100,
    annotation_processing: str | dict[str, Any] | None = None,
    beat_subsampling: bool | dict[str, Any] | None = None,
) -> TrainingExample:
    """Load one training example from precomputed embeddings and JAMS."""

    audio_dir = Path(audio_embedding_root) / audio_encoder / track.dataset / track.track_id
    text_dir = Path(text_embedding_root) / text_encoder / track.dataset

    audio = np.load(audio_dir / f"{audio_embedding_key}.npy").astype(np.float32, copy=False)
    beats = np.load(audio_dir / "beats.npy").astype(np.float32, copy=False)
    metadata = _load_json(audio_dir / "metadata.json")
    duration = float(metadata["outputs"]["duration"])
    beat_intervals = build_beat_intervals(beats, track_duration=duration)
    if len(audio) != len(beat_intervals):
        raise ValueError(
            f"Audio/beat length mismatch for {track.track_id}: "
            f"{len(audio)} audio frames vs {len(beat_intervals)} beat intervals"
        )
    beat_subsampling_config = _beat_subsampling_config(beat_subsampling)
    audio, beat_intervals, beat_subsampling_applied = _apply_beat_subsampling(
        audio=audio,
        beat_intervals=beat_intervals,
        beats=beats,
        config=beat_subsampling_config,
    )

    labels_payload = _load_json(text_dir / "labels.json")
    dataset_labels = list(labels_payload["labels"])
    dataset_text = np.load(text_dir / "embeddings.npy").astype(np.float32, copy=False)
    if len(dataset_labels) != len(dataset_text):
        raise ValueError(f"Label/text embedding mismatch in {text_dir}")

    sections = load_processed_structure_sections(
        track.jams_path,
        namespace=namespace,
        annotation_processing=annotation_processing,
    )
    labels, text = _select_candidate_labels(
        sections=sections,
        dataset_labels=dataset_labels,
        dataset_text=dataset_text,
        strategy=candidate_label_strategy,
        text_dir=text_dir,
    )
    silence_label = silence_label_in(labels)
    if beat_subsampling_applied and beat_subsampling_config["target_assignment"] == "max_overlap":
        targets = assign_intervals_to_grid(
            beat_intervals,
            sections,
            labels=labels,
            unknown_label=silence_label,
            no_overlap_value=ignore_index,
        )
    else:
        target_position = (
            "center"
            if beat_subsampling_config["target_assignment"] == "adjusted_center"
            else "start"
        )
        targets = assign_intervals_to_adjusted_timeline(
            beat_intervals,
            sections,
            duration=duration,
            labels=labels,
            no_overlap_value=ignore_index,
            position=target_position,
        )
    if len(targets) != len(audio):
        raise ValueError(
            f"Target/audio length mismatch for {track.track_id}: "
            f"{len(targets)} targets vs {len(audio)} audio frames"
        )

    base_targets = _base_targets(targets, labels=labels, ignore_index=ignore_index)
    segment_targets = _segment_targets(targets, ignore_index=ignore_index)

    return TrainingExample(
        track_id=track.track_id,
        dataset=track.dataset,
        audio=audio,
        text=text,
        targets=targets,
        base_targets=base_targets,
        segment_targets=segment_targets,
        labels=labels,
        beat_intervals=beat_intervals,
    )


def _select_candidate_labels(
    sections,
    dataset_labels: list[str],
    dataset_text: np.ndarray,
    strategy: str,
    text_dir: Path,
) -> tuple[list[str], np.ndarray]:
    if strategy == "dataset_labels":
        return dataset_labels, dataset_text
    if strategy != "track_labels":
        raise ValueError("candidate_label_strategy must be one of: dataset_labels, track_labels")

    label_to_index = {label: index for index, label in enumerate(dataset_labels)}
    labels = unique_labels(sections)
    silence_label = silence_label_in(dataset_labels)
    if silence_label is not None and silence_label not in labels:
        labels.append(silence_label)
    missing_labels = [label for label in labels if label not in label_to_index]
    if missing_labels:
        raise KeyError(f"Labels {missing_labels} are not in the precomputed text labels at {text_dir}")
    indices = [label_to_index[label] for label in labels]
    return labels, dataset_text[indices]


def collate_training_examples(examples: list[TrainingExample]) -> dict[str, Any]:
    """Collate variable-length examples for PyTorch training."""

    if not examples:
        raise ValueError("Cannot collate an empty batch")

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install torch to collate training examples.") from exc

    labels = examples[0].labels
    if any(example.labels != labels for example in examples):
        raise ValueError("All examples in a batch must share the same label set")

    max_length = max(len(example.audio) for example in examples)
    audio_dim = examples[0].audio.shape[1]
    audio = torch.zeros((len(examples), max_length, audio_dim), dtype=torch.float32)
    targets = torch.full((len(examples), max_length), -100, dtype=torch.long)
    base_targets = torch.full((len(examples), max_length), -100, dtype=torch.long)
    segment_targets = torch.full((len(examples), max_length), -100, dtype=torch.long)
    mask = torch.zeros((len(examples), max_length), dtype=torch.bool)

    for index, example in enumerate(examples):
        length = len(example.audio)
        audio[index, :length] = torch.from_numpy(example.audio)
        targets[index, :length] = torch.from_numpy(example.targets.astype(np.int64))
        base_targets[index, :length] = torch.from_numpy(example.base_targets.astype(np.int64))
        segment_targets[index, :length] = torch.from_numpy(example.segment_targets.astype(np.int64))
        mask[index, :length] = True

    text = torch.from_numpy(examples[0].text)
    return {
        "track_ids": [example.track_id for example in examples],
        "datasets": [example.dataset for example in examples],
        "audio": audio,
        "text": text,
        "targets": targets,
        "base_targets": base_targets,
        "segment_targets": segment_targets,
        "mask": mask,
        "labels": labels,
    }


def _base_targets(targets: np.ndarray, labels: list[str], ignore_index: int) -> np.ndarray:
    bases = [label_base(label) for label in labels]
    base_to_index = {base: index for index, base in enumerate(dict.fromkeys(bases))}
    label_to_base_index = np.asarray(
        [base_to_index[base] for base in bases],
        dtype=np.int64,
    )
    output = np.full(targets.shape, ignore_index, dtype=np.int64)
    valid = targets != ignore_index
    output[valid] = label_to_base_index[targets[valid].astype(np.int64)]
    return output


def _segment_targets(targets: np.ndarray, ignore_index: int) -> np.ndarray:
    output = np.full(targets.shape, ignore_index, dtype=np.int64)
    current_segment = -1
    previous_target = ignore_index
    for index, target in enumerate(targets.astype(np.int64)):
        if target == ignore_index:
            previous_target = ignore_index
            continue
        if target != previous_target:
            current_segment += 1
        output[index] = current_segment
        previous_target = target
    return output


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _apply_beat_subsampling(
    audio: np.ndarray,
    beat_intervals: list[tuple[float, float]],
    beats: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, list[tuple[float, float]], bool]:
    if not config["enabled"]:
        return audio, beat_intervals, False

    factor = config["factor"]
    if factor <= 1:
        return audio, beat_intervals, False

    median_bpm = _median_bpm(beats)
    if not np.isfinite(median_bpm) or median_bpm <= config["bpm_threshold"]:
        return audio, beat_intervals, False

    pooled_audio = []
    pooled_intervals: list[tuple[float, float]] = []
    for start in range(0, len(audio), factor):
        end = min(start + factor, len(audio))
        chunk = audio[start:end]
        if config["pooling"] == "first":
            pooled_audio.append(chunk[0])
        else:
            pooled_audio.append(chunk.mean(axis=0))
        pooled_intervals.append((beat_intervals[start][0], beat_intervals[end - 1][1]))

    return np.stack(pooled_audio).astype(np.float32, copy=False), pooled_intervals, True


def _beat_subsampling_config(value: bool | dict[str, Any] | None) -> dict[str, Any]:
    if value in (None, False):
        return {
            "enabled": False,
            "bpm_threshold": 140.0,
            "factor": 2,
            "pooling": "mean",
            "target_assignment": "max_overlap",
        }
    if value is True:
        value = {}
    if not isinstance(value, dict):
        raise TypeError("beat_subsampling must be a boolean, mapping, or null")
    enabled = bool(value.get("enabled", True))
    factor = int(value.get("factor", 2))
    if factor < 1:
        raise ValueError("beat_subsampling.factor must be >= 1")
    bpm_threshold = float(
        value.get(
            "bpm_threshold",
            value.get("bpm_cap", value.get("max_bpm", 140.0)),
        )
    )
    if bpm_threshold <= 0:
        raise ValueError("beat_subsampling.bpm_threshold must be positive")
    pooling = str(value.get("pooling", value.get("method", "mean"))).lower()
    if pooling not in {"mean", "first"}:
        raise ValueError("beat_subsampling.pooling must be one of: mean, first")
    target_assignment = str(
        value.get("target_assignment", value.get("assignment", "max_overlap"))
    )
    if target_assignment not in {"max_overlap", "adjusted_start", "adjusted_center"}:
        raise ValueError(
            "beat_subsampling.target_assignment must be one of: "
            "max_overlap, adjusted_start, adjusted_center"
        )
    return {
        "enabled": enabled,
        "bpm_threshold": bpm_threshold,
        "factor": factor,
        "pooling": pooling,
        "target_assignment": target_assignment,
    }


def _median_bpm(beats: np.ndarray) -> float:
    if len(beats) < 2:
        return float("nan")
    intervals = np.diff(beats.astype(np.float64, copy=False))
    intervals = intervals[intervals > 0]
    if len(intervals) == 0:
        return float("nan")
    return float(60.0 / np.median(intervals))


def _resolve_annotation_processing_for_example(
    annotation_processing: str | dict[str, Any] | None,
    epoch: int,
    index: int,
    track_id: str,
) -> str | dict[str, Any] | None:
    if not is_random_annotation_processing(annotation_processing):
        return annotation_processing
    assert isinstance(annotation_processing, dict)
    choices = concrete_annotation_processing_choices(annotation_processing)
    seed = int(annotation_processing.get("seed", 0))
    rng = random.Random(_stable_epoch_seed(seed, epoch, index, track_id))
    return choices[rng.randrange(len(choices))]


def _stable_epoch_seed(seed: int, epoch: int, index: int, track_id: str) -> int:
    payload = f"{seed}:{epoch}:{index}:{track_id}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)
