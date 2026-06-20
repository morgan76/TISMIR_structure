from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tismir.data.annotations import process_sections
from tismir.data.jams import load_structure_sections, sections_to_intervals_labels
from tismir.data.manifest import load_manifest


DEFAULT_METRICS = [
    "Precision@0.5",
    "Recall@0.5",
    "F-measure@0.5",
    "Precision@3.0",
    "Recall@3.0",
    "F-measure@3.0",
    "Pairwise F-measure",
    "NCE F-measure",
]


@dataclass(frozen=True)
class TrackEvaluation:
    track_id: str
    dataset: str
    prediction_path: str
    scores: dict[str, float]


def evaluate_prediction_manifest(
    reference_manifest: str | Path,
    predictions_root: str | Path,
    namespace: str = "segment_open",
    trim: bool = True,
    reference_annotation_processing: str | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate predicted JAMS files against a reference manifest."""

    tracks = load_manifest(reference_manifest)
    results: list[TrackEvaluation] = []
    for track in tracks:
        prediction_path = Path(predictions_root) / track.dataset / f"{track.track_id}.jams"
        if not prediction_path.exists():
            raise FileNotFoundError(f"Missing prediction JAMS: {prediction_path}")
        scores = evaluate_pair(
            track.jams_path,
            prediction_path,
            namespace=namespace,
            trim=trim,
            reference_annotation_processing=reference_annotation_processing,
        )
        results.append(
            TrackEvaluation(
                track_id=track.track_id,
                dataset=track.dataset,
                prediction_path=str(prediction_path),
                scores=scores,
            )
        )

    return summarize_evaluations(results)


def evaluate_pair(
    reference_jams: str | Path,
    prediction_jams: str | Path,
    namespace: str = "segment_open",
    trim: bool = True,
    reference_annotation_processing: str | dict[str, Any] | None = None,
) -> dict[str, float]:
    """Evaluate one reference/prediction JAMS pair with mir_eval.segment."""

    try:
        import mir_eval
    except ImportError as exc:  # pragma: no cover - installed through JAMS
        raise ImportError("mir_eval is required for structure evaluation.") from exc

    reference_sections = load_structure_sections(reference_jams, namespace=namespace)
    reference_sections = process_sections(
        reference_sections,
        annotation_processing=reference_annotation_processing,
    )
    ref_intervals, ref_labels = sections_to_intervals_labels(reference_sections)
    est_intervals, est_labels = sections_to_intervals_labels(
        load_structure_sections(prediction_jams, namespace=namespace)
    )
    duration = max(float(ref_intervals[-1, 1]), float(est_intervals[-1, 1]))
    ref_intervals, ref_labels = mir_eval.util.adjust_intervals(
        ref_intervals,
        list(ref_labels),
        t_min=0.0,
        t_max=duration,
    )
    est_intervals, est_labels = mir_eval.util.adjust_intervals(
        est_intervals,
        list(est_labels),
        t_min=0.0,
        t_max=duration,
    )
    scores = mir_eval.segment.evaluate(
        ref_intervals,
        ref_labels,
        est_intervals,
        est_labels,
        trim=trim,
    )
    return {key: float(value) for key, value in scores.items()}


def summarize_evaluations(results: list[TrackEvaluation]) -> dict[str, Any]:
    """Aggregate per-track metrics into mean/std summaries."""

    if not results:
        raise ValueError("No evaluation results to summarize")

    metric_names = sorted(results[0].scores)
    summary = {}
    for metric in metric_names:
        values = np.asarray([result.scores[metric] for result in results], dtype=float)
        summary[metric] = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values)),
        }

    return {
        "num_tracks": len(results),
        "summary": summary,
        "tracks": [
            {
                "track_id": result.track_id,
                "dataset": result.dataset,
                "prediction_path": result.prediction_path,
                "scores": result.scores,
            }
            for result in results
        ],
    }


def save_evaluation(path: str | Path, evaluation: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(evaluation, handle, indent=2, sort_keys=True)
        handle.write("\n")


def format_evaluation(evaluation: dict[str, Any], metrics: list[str] | None = None) -> str:
    metrics = DEFAULT_METRICS if metrics is None else metrics
    lines = [f"Tracks: {evaluation['num_tracks']}"]
    summary = evaluation["summary"]
    for metric in metrics:
        if metric not in summary:
            continue
        lines.append(f"{metric}: {summary[metric]['mean']:.4f} +/- {summary[metric]['std']:.4f}")
    return "\n".join(lines)
