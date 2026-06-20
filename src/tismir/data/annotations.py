from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from tismir.data.schemas import Section

ANNOTATION_PROCESSING_POLICIES = {
    "keep",
    "merge",
    "enumerate_all_occurrences",
    "enumerate_consecutive_repeats",
}


def process_sections(
    sections: Sequence[Section],
    annotation_processing: str | dict[str, Any] | None = None,
) -> list[Section]:
    """Apply optional section-label processing without modifying source JAMS."""

    config = _annotation_processing_config(annotation_processing)
    policy = config["policy"]
    sections = list(sections)
    if policy == "keep":
        return sections
    if policy == "merge":
        return merge_consecutive_same_label_sections(sections)
    if policy == "enumerate_all_occurrences":
        return enumerate_section_occurrences(
            sections,
            labels_to_enumerate=_repeated_labels(sections),
            start_index=config["start_index"],
            separator=config["separator"],
        )
    if policy == "enumerate_consecutive_repeats":
        return enumerate_section_occurrences(
            sections,
            labels_to_enumerate=_consecutively_repeated_labels(sections),
            start_index=config["start_index"],
            separator=config["separator"],
        )
    raise ValueError(f"Unknown annotation processing policy: {policy}")


def merge_consecutive_same_label_sections(sections: Sequence[Section]) -> list[Section]:
    """Merge adjacent sections when their labels are identical."""

    merged: list[Section] = []
    for section in sections:
        if merged and merged[-1].label == section.label:
            previous = merged[-1]
            merged[-1] = replace(
                previous,
                end=section.end,
                confidence=_merge_confidence(previous.confidence, section.confidence),
            )
        else:
            merged.append(section)
    return merged


def enumerate_section_occurrences(
    sections: Sequence[Section],
    labels_to_enumerate: set[str],
    start_index: int = 1,
    separator: str = " ",
) -> list[Section]:
    """Append chronological occurrence numbers to selected labels."""

    if start_index < 0:
        raise ValueError("annotation_processing.start_index must be non-negative")
    counts: dict[str, int] = {}
    processed: list[Section] = []
    for section in sections:
        if section.label not in labels_to_enumerate:
            processed.append(section)
            continue
        occurrence = counts.get(section.label, 0) + start_index
        counts[section.label] = counts.get(section.label, 0) + 1
        processed.append(replace(section, label=f"{section.label}{separator}{occurrence}"))
    return processed


def _annotation_processing_config(value: str | dict[str, Any] | None) -> dict[str, Any]:
    if value in (None, False):
        value = "keep"
    if isinstance(value, str):
        value = {"policy": value}
    if not isinstance(value, dict):
        raise TypeError("annotation_processing must be a string, mapping, or null")
    policy = str(value.get("policy", value.get("consecutive_same_label", "keep")))
    if policy not in ANNOTATION_PROCESSING_POLICIES:
        raise ValueError(
            "annotation_processing.policy must be one of: "
            + ", ".join(sorted(ANNOTATION_PROCESSING_POLICIES))
        )
    return {
        "policy": policy,
        "separator": str(value.get("separator", " ")),
        "start_index": int(value.get("start_index", 1)),
    }


def _repeated_labels(sections: Sequence[Section]) -> set[str]:
    counts: dict[str, int] = {}
    for section in sections:
        counts[section.label] = counts.get(section.label, 0) + 1
    return {label for label, count in counts.items() if count > 1}


def _consecutively_repeated_labels(sections: Sequence[Section]) -> set[str]:
    repeated: set[str] = set()
    previous_label = None
    for section in sections:
        if section.label == previous_label:
            repeated.add(section.label)
        previous_label = section.label
    return repeated


def _merge_confidence(left: float | None, right: float | None) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    return 0.5 * (left + right)


def assign_intervals_to_grid(
    intervals: Sequence[tuple[float, float]],
    sections: Sequence[Section],
    labels: Sequence[str] | None = None,
    unknown_label: str | None = None,
    no_overlap_value: int | str | None = None,
) -> np.ndarray:
    """Assign each time interval to the section label with maximum overlap.

    Parameters
    ----------
    intervals:
        Sequence of ``(start, end)`` pairs in seconds.
    sections:
        Ground-truth structure sections.
    labels:
        Candidate label set. If provided, outputs are integer indices into this
        sequence. If omitted, outputs are label strings.
    unknown_label:
        Candidate label to use when no annotated section overlaps an interval.
    no_overlap_value:
        Value to emit when no section overlaps an interval. This is useful for
        training targets where unannotated frames should be ignored.
    """

    label_to_index = None if labels is None else {label: idx for idx, label in enumerate(labels)}
    assignments: list[int | str] = []

    for start, end in intervals:
        best_label = None
        best_overlap = 0.0
        for section in sections:
            overlap = _overlap(start, end, section.start, section.end)
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = section.label

        if best_label is None:
            best_label = unknown_label

        if best_label is None:
            if no_overlap_value is not None:
                assignments.append(no_overlap_value)
                continue
            raise ValueError(f"No section overlaps interval ({start}, {end})")

        if label_to_index is None:
            assignments.append(best_label)
        else:
            if best_label not in label_to_index:
                if unknown_label is None or unknown_label not in label_to_index:
                    raise KeyError(f"Label '{best_label}' is not in the candidate label set")
                best_label = unknown_label
            assignments.append(label_to_index[best_label])

    dtype = np.int64 if label_to_index is not None else object
    return np.asarray(assignments, dtype=dtype)


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_intervals_to_adjusted_timeline(
    intervals: Sequence[tuple[float, float]],
    sections: Sequence[Section],
    duration: float,
    labels: Sequence[str] | None = None,
    no_overlap_value: int | str | None = None,
    position: str = "start",
) -> np.ndarray:
    """Assign intervals using a LinkSeg-style adjusted annotation timeline.

    This follows the mapping pattern used in LinkSeg: annotations are first
    adjusted to cover ``[0, duration]`` with ``mir_eval.util.adjust_intervals``.
    Beat-synchronous frames are then assigned by a representative time point.

    Synthetic ``__T_MIN``/``__T_MAX`` labels introduced by ``mir_eval`` are
    mapped to an existing silence-like candidate label when possible; otherwise
    ``no_overlap_value`` is emitted if provided.
    """

    if position not in {"start", "center"}:
        raise ValueError("position must be one of: start, center")
    adjusted_intervals, adjusted_labels = adjust_sections_to_timeline(sections, duration)
    points = np.asarray(
        [
            start if position == "start" else 0.5 * (start + end)
            for start, end in intervals
        ],
        dtype=np.float64,
    )
    assigned_labels = assign_points_to_timeline(points, adjusted_intervals, adjusted_labels)
    return encode_timeline_labels(
        assigned_labels,
        labels=labels,
        no_overlap_value=no_overlap_value,
    )


def adjust_sections_to_timeline(
    sections: Sequence[Section],
    duration: float,
) -> tuple[np.ndarray, list[str]]:
    """Adjust section intervals to cover the song timeline."""

    if duration <= 0:
        raise ValueError("duration must be positive")
    if not sections:
        raise ValueError("sections must not be empty")
    try:
        import mir_eval
    except ImportError as exc:  # pragma: no cover - installed through JAMS
        raise ImportError("mir_eval is required for adjusted timeline mapping.") from exc

    intervals = np.asarray([(section.start, section.end) for section in sections], dtype=np.float64)
    labels = [section.label for section in sections]
    adjusted_intervals, adjusted_labels = mir_eval.util.adjust_intervals(
        intervals,
        labels,
        t_min=0.0,
        t_max=float(duration),
    )
    return np.asarray(adjusted_intervals, dtype=np.float64), list(adjusted_labels)


def assign_points_to_timeline(
    points: np.ndarray,
    intervals: np.ndarray,
    labels: Sequence[str],
) -> list[str]:
    """Assign each point to the active label in an adjusted timeline."""

    if intervals.ndim != 2 or intervals.shape[1] != 2:
        raise ValueError("intervals must have shape [num_intervals, 2]")
    if len(intervals) != len(labels):
        raise ValueError("intervals and labels must have the same length")

    starts = intervals[:, 0]
    ends = intervals[:, 1]
    assigned: list[str] = []
    for point in points:
        index = int(np.searchsorted(starts, point, side="right") - 1)
        index = min(max(index, 0), len(labels) - 1)
        if point >= ends[index] and index + 1 < len(labels):
            index += 1
        assigned.append(labels[index])
    return assigned


def encode_timeline_labels(
    assigned_labels: Sequence[str],
    labels: Sequence[str] | None = None,
    no_overlap_value: int | str | None = None,
) -> np.ndarray:
    """Encode assigned timeline labels as strings or candidate-label indices."""

    if labels is None:
        return np.asarray(list(assigned_labels), dtype=object)

    label_to_index = {label: idx for idx, label in enumerate(labels)}
    encoded: list[int | str] = []
    for label in assigned_labels:
        mapped_label = _map_synthetic_boundary_label(label, labels)
        if mapped_label in label_to_index:
            encoded.append(label_to_index[mapped_label])
        elif no_overlap_value is not None:
            encoded.append(no_overlap_value)
        else:
            raise KeyError(f"Label '{label}' is not in the candidate label set")
    return np.asarray(encoded, dtype=np.int64 if isinstance(encoded[0], int) else object)


def _map_synthetic_boundary_label(label: str, candidate_labels: Sequence[str]) -> str:
    if label not in {"__T_MIN", "__T_MAX"}:
        return label

    lowercase_to_label = {candidate.lower(): candidate for candidate in candidate_labels}
    for candidate in ("silence", "nothing", "silent", "no music"):
        if candidate in lowercase_to_label:
            return lowercase_to_label[candidate]
    return label
