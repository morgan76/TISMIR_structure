from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from tismir.data.annotations import (
    concrete_annotation_processing_choices,
    is_random_annotation_processing,
)
from tismir.data.jams import load_processed_structure_sections, unique_labels
from tismir.data.schemas import Track


def track_filter_config(value: bool | dict[str, Any] | None) -> dict[str, Any]:
    """Parse optional manifest-track filtering based on processed annotations."""

    if value in (None, False):
        return {
            "enabled": False,
            "min_useful_labels": 0,
            "min_segments": 0,
            "ignore_labels": set(),
            "random_policy_mode": "any",
        }
    if value is True:
        value = {}
    if not isinstance(value, dict):
        raise TypeError("track_filter must be a boolean, mapping, or null")

    min_useful_labels = int(
        value.get(
            "min_useful_labels",
            value.get("min_non_silence_labels", value.get("min_labels", 0)),
        )
    )
    min_segments = int(value.get("min_segments", 0))
    if min_useful_labels < 0:
        raise ValueError("track_filter.min_useful_labels must be >= 0")
    if min_segments < 0:
        raise ValueError("track_filter.min_segments must be >= 0")

    ignore_labels = value.get("ignore_labels", ())
    if isinstance(ignore_labels, (str, bytes)) or not isinstance(ignore_labels, Sequence):
        raise TypeError("track_filter.ignore_labels must be a sequence of labels")

    random_policy_mode = str(value.get("random_policy_mode", "any")).lower()
    if random_policy_mode not in {"any", "all"}:
        raise ValueError("track_filter.random_policy_mode must be one of: any, all")

    return {
        "enabled": bool(value.get("enabled", True)),
        "min_useful_labels": min_useful_labels,
        "min_segments": min_segments,
        "ignore_labels": {_canonical_label(label) for label in ignore_labels},
        "random_policy_mode": random_policy_mode,
    }


def filter_tracks_by_annotation_content(
    tracks: Sequence[Track],
    namespace: str,
    annotation_processing: str | dict[str, Any] | None,
    track_filter: bool | dict[str, Any] | None,
) -> list[Track]:
    """Return tracks whose processed annotations pass the configured filter."""

    config = track_filter_config(track_filter)
    if not config["enabled"]:
        return list(tracks)
    return [
        track
        for track in tracks
        if track_passes_annotation_filter(
            track,
            namespace=namespace,
            annotation_processing=annotation_processing,
            config=config,
        )
    ]


def track_passes_annotation_filter(
    track: Track,
    namespace: str,
    annotation_processing: str | dict[str, Any] | None,
    config: dict[str, Any],
) -> bool:
    """Return whether a track has enough usable processed annotation content."""

    choices = _annotation_filter_choices(annotation_processing)
    decisions = [
        _annotation_choice_passes(
            track,
            namespace=namespace,
            annotation_processing=choice,
            config=config,
        )
        for choice in choices
    ]
    if config["random_policy_mode"] == "all":
        return all(decisions)
    return any(decisions)


def _annotation_filter_choices(
    annotation_processing: str | dict[str, Any] | None,
) -> list[str | dict[str, Any] | None]:
    if is_random_annotation_processing(annotation_processing):
        assert isinstance(annotation_processing, dict)
        return concrete_annotation_processing_choices(annotation_processing)
    return [annotation_processing]


def _annotation_choice_passes(
    track: Track,
    namespace: str,
    annotation_processing: str | dict[str, Any] | None,
    config: dict[str, Any],
) -> bool:
    sections = load_processed_structure_sections(
        track.jams_path,
        namespace=namespace,
        annotation_processing=annotation_processing,
    )
    labels = unique_labels(sections)
    useful_labels = [
        label
        for label in labels
        if _canonical_label(label) not in config["ignore_labels"]
    ]
    return (
        len(useful_labels) >= int(config["min_useful_labels"])
        and len(sections) >= int(config["min_segments"])
    )


def _canonical_label(label: str) -> str:
    return " ".join(str(label).strip().lower().replace("_", " ").replace("-", " ").split())
