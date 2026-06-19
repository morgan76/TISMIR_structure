from __future__ import annotations

from pathlib import Path
from typing import Sequence


def save_segments_jams(
    path: str | Path,
    segments: Sequence[tuple[float, float, str]],
    duration: float | None = None,
    namespace: str = "segment_open",
) -> None:
    """Save predicted segments to a JAMS file."""

    try:
        import jams
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install the 'annotations' extra to write JAMS files.") from exc

    jam = jams.JAMS()
    if duration is None and segments:
        duration = float(segments[-1][1])
    if duration is not None:
        jam.file_metadata.duration = float(duration)

    annotation = jams.Annotation(namespace=namespace)
    for start, end, label in segments:
        annotation.append(time=float(start), duration=float(end - start), value=label)
    jam.annotations.append(annotation)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    jam.save(str(path))
