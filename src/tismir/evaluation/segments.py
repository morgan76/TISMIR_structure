from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tismir.data.jams import (
    load_processed_structure_sections,
    load_structure_sections,
    sections_to_intervals_labels,
)
from tismir.data.filters import filter_tracks_by_annotation_content
from tismir.data.manifest import load_manifest
from tismir.training.data import StructureEmbeddingDataset


DEFAULT_METRICS = [
    "Precision@0.5",
    "Recall@0.5",
    "F-measure@0.5",
    "Precision@3.0",
    "Recall@3.0",
    "F-measure@3.0",
    "Acc",
    "Balanced Acc",
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
    prediction_namespace: str | None = None,
    trim: bool = True,
    reference_annotation_processing: str | dict[str, Any] | None = None,
    audio_embedding_root: str | Path | None = None,
    audio_encoder: str | None = None,
    text_embedding_root: str | Path | None = None,
    text_encoder: str | None = None,
    audio_embedding_key: str = "beat_sync",
    candidate_label_strategy: str = "track_labels",
    beat_subsampling: bool | dict[str, Any] | None = None,
    reference_track_filter: bool | dict[str, Any] | None = None,
    ignore_index: int = -100,
) -> dict[str, Any]:
    """Evaluate predicted JAMS files against a reference manifest."""

    if prediction_namespace is None:
        prediction_namespace = namespace
    tracks = filter_tracks_by_annotation_content(
        load_manifest(reference_manifest),
        namespace=namespace,
        annotation_processing=reference_annotation_processing,
        track_filter=reference_track_filter,
    )
    frame_dataset = _frame_metric_dataset(
        reference_manifest=reference_manifest,
        namespace=namespace,
        reference_annotation_processing=reference_annotation_processing,
        audio_embedding_root=audio_embedding_root,
        audio_encoder=audio_encoder,
        text_embedding_root=text_embedding_root,
        text_encoder=text_encoder,
        audio_embedding_key=audio_embedding_key,
        candidate_label_strategy=candidate_label_strategy,
        beat_subsampling=beat_subsampling,
        reference_track_filter=reference_track_filter,
        ignore_index=ignore_index,
    )
    results: list[TrackEvaluation] = []
    for index, track in enumerate(tracks):
        prediction_path = Path(predictions_root) / track.dataset / f"{track.track_id}.jams"
        if not prediction_path.exists():
            raise FileNotFoundError(f"Missing prediction JAMS: {prediction_path}")
        scores = evaluate_pair(
            track.jams_path,
            prediction_path,
            namespace=namespace,
            prediction_namespace=prediction_namespace,
            trim=trim,
            reference_annotation_processing=reference_annotation_processing,
        )
        if frame_dataset is not None:
            example = frame_dataset[index]
            if example.track_id != track.track_id or example.dataset != track.dataset:
                raise ValueError(
                    "Frame metric dataset order does not match the reference manifest: "
                    f"{example.dataset}/{example.track_id} != {track.dataset}/{track.track_id}"
                )
            prediction_json = Path(predictions_root) / track.dataset / f"{track.track_id}.json"
            scores.update(
                _frame_label_scores(
                    prediction_json=prediction_json,
                    targets=example.targets,
                    labels=example.labels,
                    ignore_index=ignore_index,
                )
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


def _frame_metric_dataset(
    reference_manifest: str | Path,
    namespace: str,
    reference_annotation_processing: str | dict[str, Any] | None,
    audio_embedding_root: str | Path | None,
    audio_encoder: str | None,
    text_embedding_root: str | Path | None,
    text_encoder: str | None,
    audio_embedding_key: str,
    candidate_label_strategy: str,
    beat_subsampling: bool | dict[str, Any] | None,
    reference_track_filter: bool | dict[str, Any] | None,
    ignore_index: int,
) -> StructureEmbeddingDataset | None:
    required = (audio_embedding_root, audio_encoder, text_embedding_root, text_encoder)
    if all(value is None for value in required):
        return None
    if any(value is None for value in required):
        raise ValueError(
            "Frame accuracy metrics require audio_embedding_root, audio_encoder, "
            "text_embedding_root, and text_encoder."
        )
    return StructureEmbeddingDataset(
        manifest=reference_manifest,
        audio_embedding_root=audio_embedding_root,
        audio_encoder=str(audio_encoder),
        text_embedding_root=text_embedding_root,
        text_encoder=str(text_encoder),
        audio_embedding_key=audio_embedding_key,
        namespace=namespace,
        candidate_label_strategy=candidate_label_strategy,
        annotation_processing=reference_annotation_processing,
        beat_subsampling=beat_subsampling,
        track_filter=reference_track_filter,
        ignore_index=ignore_index,
    )


def _frame_label_scores(
    prediction_json: Path,
    targets: np.ndarray,
    labels: list[str],
    ignore_index: int,
) -> dict[str, float]:
    if not prediction_json.exists():
        raise FileNotFoundError(f"Missing prediction JSON for frame metrics: {prediction_json}")
    with prediction_json.open("r", encoding="utf-8") as handle:
        prediction = json.load(handle)

    predicted = _prediction_indices_for_labels(
        prediction=prediction,
        labels=labels,
        ignore_index=ignore_index,
    )
    if len(predicted) != len(targets):
        raise ValueError(
            f"Frame prediction/target length mismatch for {prediction_json}: "
            f"{len(predicted)} predictions vs {len(targets)} targets"
        )

    targets = targets.astype(np.int64, copy=False)
    valid = targets != ignore_index
    if not np.any(valid):
        return {"Acc": float("nan"), "Balanced Acc": float("nan")}

    valid_predictions = predicted[valid]
    valid_targets = targets[valid]
    acc = float(np.mean(valid_predictions == valid_targets))
    per_label_acc = [
        float(np.mean(valid_predictions[valid_targets == label_index] == label_index))
        for label_index in np.unique(valid_targets)
    ]
    return {
        "Acc": acc,
        "Balanced Acc": float(np.mean(per_label_acc)) if per_label_acc else float("nan"),
    }


def _prediction_indices_for_labels(
    prediction: dict[str, Any],
    labels: list[str],
    ignore_index: int,
) -> np.ndarray:
    raw_indices = np.asarray(prediction["frame_label_indices"], dtype=np.int64)
    prediction_labels = list(prediction.get("labels", []))
    if prediction_labels == labels:
        return raw_indices

    label_to_index = {label: index for index, label in enumerate(labels)}
    mapped = np.full(raw_indices.shape, ignore_index, dtype=np.int64)
    for output_index, label in enumerate(prediction_labels):
        if label in label_to_index:
            mapped[raw_indices == output_index] = label_to_index[label]
    return mapped


def evaluate_pair(
    reference_jams: str | Path,
    prediction_jams: str | Path,
    namespace: str = "segment_open",
    prediction_namespace: str | None = None,
    trim: bool = True,
    reference_annotation_processing: str | dict[str, Any] | None = None,
) -> dict[str, float]:
    """Evaluate one reference/prediction JAMS pair with mir_eval.segment."""

    try:
        import mir_eval
    except ImportError as exc:  # pragma: no cover - installed through JAMS
        raise ImportError("mir_eval is required for structure evaluation.") from exc

    reference_sections = load_processed_structure_sections(
        reference_jams,
        namespace=namespace,
        annotation_processing=reference_annotation_processing,
    )
    if prediction_namespace is None:
        prediction_namespace = namespace
    ref_intervals, ref_labels = sections_to_intervals_labels(reference_sections)
    est_intervals, est_labels = sections_to_intervals_labels(
        load_structure_sections(prediction_jams, namespace=prediction_namespace)
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
