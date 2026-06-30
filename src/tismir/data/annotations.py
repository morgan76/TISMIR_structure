from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
import re
from typing import Any

import numpy as np

from tismir.data.schemas import Section

ANNOTATION_PROCESSING_POLICIES = {
    "keep",
    "merge",
    "base_labels",
    "enumerate_all_occurrences",
    "enumerate_base_occurrences",
    "enumerate_consecutive_repeats",
    "salami_function_merge",
    "salami_function_occurrences",
    "salami_function_projected_lower",
}
RANDOM_ANNOTATION_PROCESSING_POLICIES = {"random", "sample"}
SALAMI_FUNCTION_POLICIES = {
    "salami_function_merge",
    "salami_function_occurrences",
    "salami_function_projected_lower",
}
SYNTHETIC_SILENCE_LABEL = "silence"


def process_sections(
    sections: Sequence[Section],
    annotation_processing: str | dict[str, Any] | None = None,
) -> list[Section]:
    """Apply optional section-label processing without modifying source JAMS."""

    config = _annotation_processing_config(annotation_processing)
    policy = config["policy"]
    sections = list(sections)
    if config["replace_no_function"]:
        sections = replace_no_function_sections(
            sections,
            label=config["no_function_label"],
            skip_labels=config["no_function_skip_labels"],
        )
    if policy == "salami_function_merge":
        return merge_consecutive_same_label_sections(sections)
    if policy == "salami_function_occurrences":
        merged = merge_consecutive_same_label_sections(sections)
        skip_bases = {label_base(label) for label in config["occurrence_skip_labels"]}
        return enumerate_section_base_occurrences(
            merged,
            bases_to_enumerate=_repeated_label_bases(merged) - skip_bases,
            start_index=config["start_index"],
            separator=config["separator"],
        )
    if policy == "salami_function_projected_lower":
        raise ValueError(
            "salami_function_projected_lower requires access to both function and lower "
            "SALAMI namespaces; use load_processed_structure_sections()."
        )
    if policy == "keep":
        return sections
    if policy == "merge":
        return merge_consecutive_same_label_sections(sections)
    if policy == "base_labels":
        return merge_consecutive_same_label_sections(
            [replace(section, label=label_base(section.label)) for section in sections]
        )
    if policy == "enumerate_all_occurrences":
        return enumerate_section_occurrences(
            sections,
            labels_to_enumerate=_repeated_labels(sections),
            start_index=config["start_index"],
            separator=config["separator"],
        )
    if policy == "enumerate_base_occurrences":
        return enumerate_section_base_occurrences(
            sections,
            bases_to_enumerate=_repeated_label_bases(sections),
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


def is_random_annotation_processing(annotation_processing: Any) -> bool:
    """Return whether an annotation-processing config describes policy sampling."""

    if not isinstance(annotation_processing, dict):
        return False
    return str(annotation_processing.get("policy", "")).lower() in RANDOM_ANNOTATION_PROCESSING_POLICIES


def concrete_annotation_processing_choices(
    annotation_processing: dict[str, Any],
) -> list[str | dict[str, Any]]:
    """Return concrete annotation policies from a random policy config."""

    policy = str(annotation_processing.get("policy", "")).lower()
    if policy not in RANDOM_ANNOTATION_PROCESSING_POLICIES:
        raise ValueError("annotation_processing.policy must be random or sample")
    choices = annotation_processing.get("choices", annotation_processing.get("policies"))
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
        raise ValueError("random annotation_processing requires a non-empty choices list")

    concrete: list[str | dict[str, Any]] = []
    for choice in choices:
        concrete.append(_concrete_annotation_processing_choice(annotation_processing, choice))
    return concrete


def validation_annotation_processing_choice(
    annotation_processing: dict[str, Any],
) -> str | dict[str, Any]:
    """Return the deterministic validation policy for a random training config."""

    validation_choice = annotation_processing.get(
        "validation_policy",
        annotation_processing.get("default_policy"),
    )
    if validation_choice is None:
        choices = concrete_annotation_processing_choices(annotation_processing)
        if any(_annotation_processing_config(choice)["policy"] == "enumerate_base_occurrences" for choice in choices):
            validation_choice = "enumerate_base_occurrences"
        else:
            validation_choice = choices[0]
    return _concrete_annotation_processing_choice(annotation_processing, validation_choice)


def _concrete_annotation_processing_choice(
    parent: dict[str, Any],
    choice: str | dict[str, Any],
) -> str | dict[str, Any]:
    if isinstance(choice, str):
        config: dict[str, Any] = {"policy": choice}
    elif isinstance(choice, dict):
        config = dict(choice)
    else:
        raise TypeError("annotation_processing choices must be strings or mappings")

    policy = str(config.get("policy", "")).lower()
    if policy in RANDOM_ANNOTATION_PROCESSING_POLICIES:
        raise ValueError("nested random annotation_processing choices are not supported")
    for key in (
        "separator",
        "start_index",
        "replace_no_function",
        "salami_replace_no_function",
        "no_function_label",
        "no_function_skip_labels",
        "function_namespace",
        "lower_namespace",
        "projected_label_format",
        "projected_preserve_labels",
        "projected_function_policy",
        "merge_projected_lower",
        "occurrence_skip_labels",
        "annotation_selection",
    ):
        if key in parent and key not in config:
            config[key] = parent[key]
    _annotation_processing_config(config)
    return config


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


def ensure_silence_label(labels: Sequence[str]) -> list[str]:
    """Return labels with the synthetic silence class available."""

    labels = list(labels)
    if not any(_canonical_label(label) == SYNTHETIC_SILENCE_LABEL for label in labels):
        labels.append(SYNTHETIC_SILENCE_LABEL)
    return labels


def silence_label_in(labels: Sequence[str]) -> str | None:
    """Return the existing silence-like candidate label, if any."""

    lowercase_to_label = {candidate.lower(): candidate for candidate in labels}
    for candidate in ("silence", "nothing", "silent", "no music"):
        if candidate in lowercase_to_label:
            return lowercase_to_label[candidate]
    return None


def project_lower_sections_to_function_labels(
    function_sections: Sequence[Section],
    lower_sections: Sequence[Section],
    label_format: str = "{function} subsegment {lower}",
    preserve_labels: Sequence[str] = ("silence",),
    merge_consecutive: bool = True,
) -> list[Section]:
    """Label SALAMI lower-level sections by their overlapping functional section."""

    functions = list(function_sections)
    lower = list(lower_sections)
    preserve = {_canonical_label(label) for label in preserve_labels}
    projected: list[Section] = []
    for lower_section in lower:
        function = _best_overlapping_section(lower_section, functions)
        if function is None:
            derived_label = lower_section.label
        elif _canonical_label(function.label) in preserve:
            derived_label = function.label.lower()
        else:
            derived_label = label_format.format(
                function=function.label,
                function_base=label_base(function.label),
                lower=lower_section.label,
                lower_base=label_base(lower_section.label),
            )
        projected.append(replace(lower_section, label=derived_label))

    if merge_consecutive:
        return merge_consecutive_same_label_sections(projected)
    return projected


def replace_no_function_sections(
    sections: Sequence[Section],
    label: str = "no_function",
    skip_labels: Sequence[str] = (),
) -> list[Section]:
    """Replace SALAMI no-function placeholders with neighboring function labels."""

    sections = list(sections)
    skip = {_canonical_label(value) for value in skip_labels}
    placeholder = _canonical_label(label)
    replacements: list[str | None] = []
    previous: str | None = None
    for section in sections:
        current = _canonical_label(section.label)
        if current == placeholder:
            replacements.append(previous)
            continue
        replacements.append(section.label)
        if current not in skip:
            previous = section.label

    next_label: str | None = None
    for index in range(len(sections) - 1, -1, -1):
        current = _canonical_label(sections[index].label)
        if current == placeholder:
            if replacements[index] is None and next_label is not None:
                replacements[index] = next_label
            continue
        if current not in skip:
            next_label = sections[index].label

    return [
        replace(section, label=replacement) if replacement is not None else section
        for section, replacement in zip(sections, replacements)
    ]


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


def enumerate_section_base_occurrences(
    sections: Sequence[Section],
    bases_to_enumerate: set[str],
    start_index: int = 1,
    separator: str = " ",
) -> list[Section]:
    """Append chronological occurrence numbers after stripping existing markers."""

    if start_index < 0:
        raise ValueError("annotation_processing.start_index must be non-negative")
    counts: dict[str, int] = {}
    processed: list[Section] = []
    for section in sections:
        base = label_base(section.label)
        if base not in bases_to_enumerate:
            processed.append(section)
            continue
        occurrence = counts.get(base, 0) + start_index
        counts[base] = counts.get(base, 0) + 1
        processed.append(replace(section, label=f"{base}{separator}{occurrence}"))
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
    replace_no_function_default = policy in SALAMI_FUNCTION_POLICIES
    annotation_selection = str(value.get("annotation_selection", "first"))
    if annotation_selection not in {"first", "richest_function"}:
        raise ValueError("annotation_processing.annotation_selection must be one of: first, richest_function")
    projected_function_policy = str(value.get("projected_function_policy", "salami_function_merge"))
    if projected_function_policy not in {"salami_function_merge", "salami_function_occurrences"}:
        raise ValueError(
            "annotation_processing.projected_function_policy must be one of: "
            "salami_function_merge, salami_function_occurrences"
        )
    return {
        "policy": policy,
        "separator": str(value.get("separator", " ")),
        "start_index": int(value.get("start_index", 1)),
        "annotation_selection": annotation_selection,
        "replace_no_function": bool(
            value.get(
                "replace_no_function",
                value.get("salami_replace_no_function", replace_no_function_default),
            )
        ),
        "no_function_label": str(value.get("no_function_label", "no_function")),
        "no_function_skip_labels": tuple(value.get("no_function_skip_labels", ())),
        "function_namespace": str(value.get("function_namespace", "segment_salami_function")),
        "lower_namespace": str(value.get("lower_namespace", "segment_salami_lower")),
        "projected_label_format": str(
            value.get("projected_label_format", "{function} subsegment {lower}")
        ),
        "projected_preserve_labels": tuple(value.get("projected_preserve_labels", ("silence",))),
        "projected_function_policy": projected_function_policy,
        "merge_projected_lower": bool(value.get("merge_projected_lower", True)),
        "occurrence_skip_labels": tuple(
            value.get(
                "occurrence_skip_labels",
                ("silence",) if policy == "salami_function_occurrences" else (),
            )
        ),
    }


def annotation_processing_config(value: str | dict[str, Any] | None) -> dict[str, Any]:
    """Return normalized annotation-processing config."""

    return _annotation_processing_config(value)


def is_salami_projected_lower_policy(value: str | dict[str, Any] | None) -> bool:
    return _annotation_processing_config(value)["policy"] == "salami_function_projected_lower"


def source_namespace_for_annotation_processing(
    namespace: str,
    annotation_processing: str | dict[str, Any] | None,
) -> str:
    config = _annotation_processing_config(annotation_processing)
    if config["policy"] in SALAMI_FUNCTION_POLICIES:
        return config["function_namespace"]
    return namespace


def _repeated_labels(sections: Sequence[Section]) -> set[str]:
    counts: dict[str, int] = {}
    for section in sections:
        counts[section.label] = counts.get(section.label, 0) + 1
    return {label for label, count in counts.items() if count > 1}


def _repeated_label_bases(sections: Sequence[Section]) -> set[str]:
    counts: dict[str, int] = {}
    for section in sections:
        base = label_base(section.label)
        counts[base] = counts.get(base, 0) + 1
    return {label for label, count in counts.items() if count > 1}


def _consecutively_repeated_labels(sections: Sequence[Section]) -> set[str]:
    repeated: set[str] = set()
    previous_label = None
    for section in sections:
        if section.label == previous_label:
            repeated.add(section.label)
        previous_label = section.label
    return repeated


def label_base(label: str) -> str:
    """Return a normalized section base label without occurrence markers."""

    text = re.sub(r"[_\-]+", " ", label.strip().lower())
    text = re.sub(r"(?<=[a-z])(?=\d)", " ", text)
    text = re.sub(r"(?<=\d)(?=[a-z])", " ", text)
    text = " ".join(text.split())
    return re.sub(r"\s+[0-9]+(?:\s*[a-z])?$", "", text)


def _merge_confidence(left: float | None, right: float | None) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    return 0.5 * (left + right)


def _canonical_label(label: str) -> str:
    return re.sub(r"[\s_\-]+", "_", label.strip().lower())


def _best_overlapping_section(section: Section, candidates: Sequence[Section]) -> Section | None:
    best_section = None
    best_overlap = 0.0
    for candidate in candidates:
        overlap = _overlap(section.start, section.end, candidate.start, candidate.end)
        if overlap > best_overlap:
            best_overlap = overlap
            best_section = candidate
    return best_section


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

    return silence_label_in(candidate_labels) or label
