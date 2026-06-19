from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tismir.data.annotations import assign_intervals_to_adjusted_timeline
from tismir.data.jams import load_structure_sections
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
    ) -> None:
        if candidate_label_strategy != "dataset_labels":
            raise ValueError("Only 'dataset_labels' is currently implemented")
        self.tracks = load_manifest(manifest)
        self.audio_embedding_root = Path(audio_embedding_root)
        self.audio_encoder = audio_encoder
        self.text_embedding_root = Path(text_embedding_root)
        self.text_encoder = text_encoder
        self.audio_embedding_key = audio_embedding_key
        self.namespace = namespace
        self.ignore_index = ignore_index

    def __len__(self) -> int:
        return len(self.tracks)

    def __getitem__(self, index: int) -> TrainingExample:
        track = self.tracks[index]
        return load_training_example(
            track=track,
            audio_embedding_root=self.audio_embedding_root,
            audio_encoder=self.audio_encoder,
            text_embedding_root=self.text_embedding_root,
            text_encoder=self.text_encoder,
            audio_embedding_key=self.audio_embedding_key,
            namespace=self.namespace,
            ignore_index=self.ignore_index,
        )


def load_training_example(
    track: Track,
    audio_embedding_root: str | Path,
    audio_encoder: str,
    text_embedding_root: str | Path,
    text_encoder: str,
    audio_embedding_key: str = "beat_sync",
    namespace: str = "segment_open",
    ignore_index: int = -100,
) -> TrainingExample:
    """Load one training example from precomputed embeddings and JAMS."""

    audio_dir = Path(audio_embedding_root) / audio_encoder / track.dataset / track.track_id
    text_dir = Path(text_embedding_root) / text_encoder / track.dataset

    audio = np.load(audio_dir / f"{audio_embedding_key}.npy").astype(np.float32)
    beats = np.load(audio_dir / "beats.npy").astype(np.float32)
    metadata = _load_json(audio_dir / "metadata.json")
    duration = float(metadata["outputs"]["duration"])
    beat_intervals = build_beat_intervals(beats, track_duration=duration)

    labels_payload = _load_json(text_dir / "labels.json")
    labels = list(labels_payload["labels"])
    text = np.load(text_dir / "embeddings.npy").astype(np.float32)
    if len(labels) != len(text):
        raise ValueError(f"Label/text embedding mismatch in {text_dir}")

    sections = load_structure_sections(track.jams_path, namespace=namespace)
    targets = assign_intervals_to_adjusted_timeline(
        beat_intervals,
        sections,
        duration=duration,
        labels=labels,
        no_overlap_value=ignore_index,
    )
    if len(targets) != len(audio):
        raise ValueError(
            f"Target/audio length mismatch for {track.track_id}: "
            f"{len(targets)} targets vs {len(audio)} audio frames"
        )

    return TrainingExample(
        track_id=track.track_id,
        dataset=track.dataset,
        audio=audio,
        text=text,
        targets=targets,
        labels=labels,
        beat_intervals=beat_intervals,
    )


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
    mask = torch.zeros((len(examples), max_length), dtype=torch.bool)

    for index, example in enumerate(examples):
        length = len(example.audio)
        audio[index, :length] = torch.from_numpy(example.audio)
        targets[index, :length] = torch.from_numpy(example.targets.astype(np.int64))
        mask[index, :length] = True

    text = torch.from_numpy(examples[0].text)
    return {
        "track_ids": [example.track_id for example in examples],
        "datasets": [example.dataset for example in examples],
        "audio": audio,
        "text": text,
        "targets": targets,
        "mask": mask,
        "labels": labels,
    }


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
