from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from tismir.data.schemas import Track


def load_manifest(path: str | Path) -> list[Track]:
    """Load a JSONL manifest into dataset-agnostic track records."""

    path = Path(path)
    tracks: list[Track] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
            tracks.append(track_from_record(record, base_dir=path.parent))
    return tracks


def save_manifest(path: str | Path, tracks: Iterable[Track]) -> None:
    """Save track records as JSONL."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for track in tracks:
            handle.write(json.dumps(track_to_record(track), sort_keys=True))
            handle.write("\n")


def track_from_record(record: dict, base_dir: str | Path | None = None) -> Track:
    """Create a :class:`Track` from a manifest dictionary."""

    base = Path("." if base_dir is None else base_dir)
    audio_path = _resolve_path(record["audio_path"], base)
    jams_path = _resolve_path(record["jams_path"], base)
    return Track(
        track_id=str(record["track_id"]),
        audio_path=audio_path,
        jams_path=jams_path,
        dataset=str(record["dataset"]),
        split=None if record.get("split") is None else str(record["split"]),
        metadata=dict(record.get("metadata", {})),
    )


def track_to_record(track: Track) -> dict:
    """Convert a :class:`Track` to a JSON-serializable manifest dictionary."""

    return {
        "track_id": track.track_id,
        "audio_path": str(track.audio_path),
        "jams_path": str(track.jams_path),
        "dataset": track.dataset,
        "split": track.split,
        "metadata": dict(track.metadata),
    }


def _resolve_path(path: str, base_dir: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return base_dir / candidate
