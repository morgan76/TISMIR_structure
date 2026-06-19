from __future__ import annotations

from pathlib import Path
from typing import Iterable

from tismir.data.schemas import Section


def load_structure_sections(jams_path: str | Path, namespace: str = "segment_open") -> list[Section]:
    """Load section intervals from a JAMS file.

    The default namespace follows the JAMS convention for structural segments.
    Some datasets may use another segment namespace; callers can override it.
    """

    try:
        import jams
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError("Install the 'annotations' extra to read JAMS files.") from exc

    jam = jams.load(str(jams_path))
    annotations = jam.search(namespace=namespace)
    if not annotations:
        raise ValueError(f"No JAMS annotations found for namespace '{namespace}' in {jams_path}")

    sections: list[Section] = []
    for obs in annotations[0].data:
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
