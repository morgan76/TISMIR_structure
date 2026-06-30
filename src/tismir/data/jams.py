from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Iterable

from tismir.data.annotations import (
    annotation_processing_config,
    is_salami_projected_lower_policy,
    merge_consecutive_same_label_sections,
    process_sections,
    project_lower_sections_to_function_labels,
    replace_no_function_sections,
    source_namespace_for_annotation_processing,
)
from tismir.data.schemas import Section


def load_structure_sections(
    jams_path: str | Path,
    namespace: str = "segment_open",
    annotation_index: int = 0,
) -> list[Section]:
    """Load section intervals from a JAMS file.

    The default namespace follows the JAMS convention for structural segments.
    Some datasets may use another segment namespace; callers can override it.
    """

    annotations = _load_annotations(jams_path, namespace)
    if not annotations:
        raise ValueError(f"No JAMS annotations found for namespace '{namespace}' in {jams_path}")
    if annotation_index < 0 or annotation_index >= len(annotations):
        raise IndexError(
            f"Annotation index {annotation_index} is out of range for namespace "
            f"'{namespace}' in {jams_path}; found {len(annotations)} annotations"
        )
    return _sections_from_annotation(annotations[annotation_index])


def load_structure_annotation_count(jams_path: str | Path, namespace: str = "segment_open") -> int:
    """Return the number of annotations in a JAMS namespace."""

    return len(_load_annotations(jams_path, namespace))


def load_processed_structure_sections(
    jams_path: str | Path,
    namespace: str = "segment_open",
    annotation_processing: str | dict[str, Any] | None = None,
) -> list[Section]:
    """Load JAMS sections and apply annotation processing.

    Most policies transform a single annotation namespace. SALAMI projected-lower
    labels are hierarchical: they project lower-level sections onto overlapping
    functional sections, so that policy loads both SALAMI namespaces.
    """

    config = annotation_processing_config(annotation_processing)
    if is_salami_projected_lower_policy(config):
        function_annotations = _load_annotations(jams_path, config["function_namespace"])
        function_index = _select_annotation_index(
            function_annotations,
            config=config,
            fallback_namespace=config["function_namespace"],
            jams_path=jams_path,
        )
        function_sections = _sections_from_annotation(function_annotations[function_index])
        function_processing = dict(config)
        function_processing["policy"] = config["projected_function_policy"]
        function_sections = process_sections(function_sections, annotation_processing=function_processing)
        lower_annotations = _load_annotations(jams_path, config["lower_namespace"])
        lower_index = min(function_index, len(lower_annotations) - 1)
        lower_sections = _sections_from_annotation(lower_annotations[lower_index])
        return project_lower_sections_to_function_labels(
            function_sections=function_sections,
            lower_sections=lower_sections,
            label_format=config["projected_label_format"],
            preserve_labels=config["projected_preserve_labels"],
            merge_consecutive=bool(config["merge_projected_lower"]),
        )

    source_namespace = source_namespace_for_annotation_processing(
        namespace,
        annotation_processing=config,
    )
    annotations = _load_annotations(jams_path, source_namespace)
    annotation_index = _select_annotation_index(
        annotations,
        config=config,
        fallback_namespace=source_namespace,
        jams_path=jams_path,
    )
    sections = _sections_from_annotation(annotations[annotation_index])
    return process_sections(sections, annotation_processing=config)


def unique_labels(sections: Iterable[Section]) -> list[str]:
    """Return labels in first-seen order."""

    labels: list[str] = []
    seen: set[str] = set()
    for section in sections:
        if section.label not in seen:
            labels.append(section.label)
            seen.add(section.label)
    return labels


def sections_to_intervals_labels(sections: Iterable[Section]):
    """Convert sections to mir_eval-style intervals and labels."""

    import numpy as np

    sections = list(sections)
    intervals = np.asarray([(section.start, section.end) for section in sections], dtype=float)
    labels = [section.label for section in sections]
    return intervals, labels


def _load_annotations(jams_path: str | Path, namespace: str):
    try:
        import jams
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError("Install the 'annotations' extra to read JAMS files.") from exc

    jam = jams.load(str(jams_path))
    annotations = jam.search(namespace=namespace)
    if not annotations:
        raise ValueError(f"No JAMS annotations found for namespace '{namespace}' in {jams_path}")
    return annotations


def _sections_from_annotation(annotation) -> list[Section]:
    sections: list[Section] = []
    for obs in annotation.data:
        start = float(obs.time)
        duration = float(obs.duration)
        value = obs.value
        if isinstance(value, dict):
            label = str(value.get("label", value.get("value", "")))
            metadata = value
        else:
            label = str(value)
            metadata = {}
        sections.append(
            Section(
                start=start,
                end=start + duration,
                label=label,
                confidence=None if obs.confidence is None else float(obs.confidence),
                metadata=metadata,
            )
        )
    return sections


def _select_annotation_index(
    annotations,
    config: dict[str, Any],
    fallback_namespace: str,
    jams_path: str | Path,
) -> int:
    selection = str(config.get("annotation_selection", "first"))
    if selection == "first":
        return 0
    if selection != "richest_function":
        raise ValueError(f"Unknown annotation selection strategy: {selection}")
    if not annotations:
        raise ValueError(f"No JAMS annotations found for namespace '{fallback_namespace}' in {jams_path}")
    scores = [
        _richest_function_score(_sections_from_annotation(annotation), config=config)
        for annotation in annotations
    ]
    return max(range(len(scores)), key=lambda index: scores[index])


def _richest_function_score(sections: list[Section], config: dict[str, Any]) -> tuple[int, int, int, int]:
    processed = sections
    if config.get("replace_no_function", False):
        processed = replace_no_function_sections(
            processed,
            label=config["no_function_label"],
            skip_labels=config["no_function_skip_labels"],
        )
    processed = merge_consecutive_same_label_sections(processed)
    labels = unique_labels(processed)
    useful_labels = [
        label
        for label in labels
        if _canonical_label(label) not in {"silence", "no function"}
    ]
    return (
        len(useful_labels),
        len(labels),
        len(processed),
        -len(sections),
    )


def _canonical_label(label: str) -> str:
    return " ".join(str(label).strip().lower().replace("_", " ").replace("-", " ").split())
