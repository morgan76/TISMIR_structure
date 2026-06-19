from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from tismir.data.schemas import Section


def assign_intervals_to_grid(
    intervals: Sequence[tuple[float, float]],
    sections: Sequence[Section],
    labels: Sequence[str] | None = None,
    unknown_label: str | None = None,
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
