from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from tismir.data.jams import load_structure_sections
from tismir.data.schemas import Section, Track


@dataclass(frozen=True)
class TrackValidationResult:
    track_id: str
    dataset: str
    audio_path: str
    jams_path: str
    valid: bool
    num_sections: int = 0
    duration: float | None = None
    labels: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class DatasetSummary:
    num_tracks: int
    num_valid_tracks: int
    num_invalid_tracks: int
    num_sections: int
    duration: float | None
    label_counts: dict[str, int] = field(default_factory=dict)
    invalid_tracks: list[TrackValidationResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["invalid_tracks"] = [asdict(track) for track in self.invalid_tracks]
        return data


def validate_track(track: Track, namespace: str = "segment_open") -> TrackValidationResult:
    """Validate a track record and its JAMS section annotation."""

    errors: list[str] = []
    if not track.audio_path.exists():
        errors.append(f"Missing audio file: {track.audio_path}")
    if not track.jams_path.exists():
        errors.append(f"Missing JAMS file: {track.jams_path}")
        return TrackValidationResult(
            track_id=track.track_id,
            dataset=track.dataset,
            audio_path=str(track.audio_path),
            jams_path=str(track.jams_path),
            valid=False,
            errors=tuple(errors),
        )

    sections: list[Section] = []
    try:
        sections = load_structure_sections(track.jams_path, namespace=namespace)
    except Exception as exc:  # pragma: no cover - specific exception types vary by JAMS
        errors.append(f"Could not load JAMS sections: {exc}")

    if not sections:
        errors.append(f"No sections found in namespace '{namespace}'")

    for index, section in enumerate(sections):
        if section.end <= section.start:
            errors.append(f"Section {index} has non-positive duration")
        if not section.label:
            errors.append(f"Section {index} has an empty label")

    duration = max((section.end for section in sections), default=None)
    labels = tuple(section.label for section in sections)
    return TrackValidationResult(
        track_id=track.track_id,
        dataset=track.dataset,
        audio_path=str(track.audio_path),
        jams_path=str(track.jams_path),
        valid=not errors,
        num_sections=len(sections),
        duration=duration,
        labels=labels,
        errors=tuple(errors),
    )


def summarize_validation(results: Iterable[TrackValidationResult]) -> DatasetSummary:
    """Summarize validation results across a manifest."""

    results = list(results)
    label_counts: Counter[str] = Counter()
    total_sections = 0
    total_duration = 0.0
    has_duration = False
    invalid_tracks: list[TrackValidationResult] = []

    for result in results:
        if not result.valid:
            invalid_tracks.append(result)
            continue
        label_counts.update(result.labels)
        total_sections += result.num_sections
        if result.duration is not None:
            total_duration += result.duration
            has_duration = True

    return DatasetSummary(
        num_tracks=len(results),
        num_valid_tracks=len(results) - len(invalid_tracks),
        num_invalid_tracks=len(invalid_tracks),
        num_sections=total_sections,
        duration=total_duration if has_duration else None,
        label_counts=dict(sorted(label_counts.items())),
        invalid_tracks=invalid_tracks,
    )


def format_summary(summary: DatasetSummary, max_labels: int = 20) -> str:
    """Format a compact human-readable dataset summary."""

    lines = [
        f"Tracks: {summary.num_valid_tracks}/{summary.num_tracks} valid",
        f"Sections: {summary.num_sections}",
    ]
    if summary.duration is not None:
        lines.append(f"Annotated duration: {summary.duration / 60.0:.2f} min")
    lines.append(f"Distinct labels: {len(summary.label_counts)}")

    most_common = sorted(summary.label_counts.items(), key=lambda item: (-item[1], item[0]))[:max_labels]
    if most_common:
        lines.append("Top labels:")
        for label, count in most_common:
            lines.append(f"  {label}: {count}")

    if summary.invalid_tracks:
        lines.append("Invalid tracks:")
        for track in summary.invalid_tracks:
            joined_errors = "; ".join(track.errors)
            lines.append(f"  {track.track_id}: {joined_errors}")

    return "\n".join(lines)


def write_summary(path: str | Path, summary: DatasetSummary) -> None:
    """Write a dataset summary JSON file."""

    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
