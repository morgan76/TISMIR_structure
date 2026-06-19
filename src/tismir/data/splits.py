from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from tismir.data.manifest import save_manifest
from tismir.data.schemas import Track


def split_tracks(
    tracks: Iterable[Track],
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    seed: int = 0,
    shuffle: bool = True,
) -> dict[str, list[Track]]:
    """Split tracks deterministically and set each track's split field."""

    tracks = list(tracks)
    if not tracks:
        raise ValueError("Cannot split an empty track list")
    ratios = {
        "train": float(train_ratio),
        "val": float(val_ratio),
        "test": float(test_ratio),
    }
    if any(value < 0 for value in ratios.values()):
        raise ValueError("Split ratios must be non-negative")
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("At least one split ratio must be positive")

    indices = list(range(len(tracks)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)

    normalized = {name: value / total for name, value in ratios.items()}
    val_count = int(round(len(tracks) * normalized["val"]))
    test_count = int(round(len(tracks) * normalized["test"]))
    train_count = len(tracks) - val_count - test_count
    if train_count < 0:
        raise ValueError("Split ratios produce negative train count")

    split_indices = {
        "train": indices[:train_count],
        "val": indices[train_count : train_count + val_count],
        "test": indices[train_count + val_count :],
    }
    return {
        split: [replace(tracks[index], split=split) for index in selected]
        for split, selected in split_indices.items()
        if selected
    }


def save_split_manifests(
    splits: dict[str, list[Track]],
    output_dir: str | Path,
    name: str,
    suffix: str = ".local",
) -> dict[str, Path]:
    """Save each split as ``{name}_{split}{suffix}.jsonl``."""

    output_dir = Path(output_dir)
    paths: dict[str, Path] = {}
    for split, tracks in splits.items():
        path = output_dir / f"{name}_{split}{suffix}.jsonl"
        save_manifest(path, tracks)
        paths[split] = path
    return paths
