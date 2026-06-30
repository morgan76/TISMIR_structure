#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from tismir.data.annotations import (
    is_random_annotation_processing,
    validation_annotation_processing_choice,
)
from tismir.decoding.segments import (
    decode_label_indices,
    merge_frame_labels,
    remove_short_segments,
    smooth_logits,
)
from tismir.models import build_model
from tismir.training.data import StructureEmbeddingDataset


METRICS = (
    "F-measure@0.5",
    "F-measure@3.0",
    "Pairwise F-measure",
    "NCE F-measure",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid-search decoding parameters for a trained structure model."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--audio-embedding-root", default=None)
    parser.add_argument("--audio-encoder", default=None)
    parser.add_argument("--text-embedding-root", default=None)
    parser.add_argument("--text-encoder", default=None)
    parser.add_argument("--audio-embedding-key", default=None)
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--candidate-label-strategy", default=None)
    parser.add_argument(
        "--annotation-policy",
        choices=[
            "keep",
            "merge",
            "base_labels",
            "enumerate_all_occurrences",
            "enumerate_base_occurrences",
            "enumerate_consecutive_repeats",
            "salami_function_merge",
            "salami_function_occurrences",
            "salami_function_projected_lower",
        ],
        default=None,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--smoothing-windows", nargs="+", type=int, default=[1, 5, 9, 13, 19])
    parser.add_argument("--smoothing-modes", nargs="+", choices=["mean", "median"], default=["mean"])
    parser.add_argument("--decoders", nargs="+", choices=["argmax", "viterbi"], default=["viterbi"])
    parser.add_argument(
        "--transition-penalties",
        nargs="+",
        type=float,
        default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
    )
    parser.add_argument(
        "--boundary-weights",
        nargs="+",
        type=float,
        default=[0.0],
        help=(
            "Viterbi boundary-prior weights. Values greater than zero use cached "
            "boundary-head probabilities when the checkpoint exposes them."
        ),
    )
    parser.add_argument(
        "--boundary-eps-values",
        nargs="+",
        type=float,
        default=[1e-4],
        help="Numerical clipping values for boundary probabilities.",
    )
    parser.add_argument("--min-segment-durations", nargs="+", type=float, default=[0.0])
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    results = run_grid_search(
        checkpoint_path=args.checkpoint,
        manifest=args.manifest,
        audio_embedding_root=args.audio_embedding_root,
        audio_encoder=args.audio_encoder,
        text_embedding_root=args.text_embedding_root,
        text_encoder=args.text_encoder,
        audio_embedding_key=args.audio_embedding_key,
        namespace=args.namespace,
        candidate_label_strategy=args.candidate_label_strategy,
        annotation_policy=args.annotation_policy,
        device=args.device,
        limit=args.limit,
        smoothing_windows=args.smoothing_windows,
        smoothing_modes=args.smoothing_modes,
        decoders=args.decoders,
        transition_penalties=args.transition_penalties,
        boundary_weights=args.boundary_weights,
        boundary_eps_values=args.boundary_eps_values,
        min_segment_durations=args.min_segment_durations,
        progress=args.progress,
    )
    save_results(args.output_json, results)
    if args.output_csv is not None:
        save_csv(args.output_csv, results["results"])
    print_top_results(results["results"])


def run_grid_search(
    checkpoint_path: str | Path,
    manifest: str | Path | None,
    audio_embedding_root: str | Path | None,
    audio_encoder: str | None,
    text_embedding_root: str | Path | None,
    text_encoder: str | None,
    audio_embedding_key: str | None,
    namespace: str | None,
    candidate_label_strategy: str | None,
    annotation_policy: str | None,
    device: str,
    limit: int | None,
    smoothing_windows: list[int],
    smoothing_modes: list[str],
    decoders: list[str],
    transition_penalties: list[float],
    boundary_weights: list[float],
    boundary_eps_values: list[float],
    min_segment_durations: list[float],
    progress: bool,
) -> dict[str, Any]:
    try:
        import mir_eval
    except ImportError as exc:  # pragma: no cover
        raise ImportError("mir_eval is required for decoding grid search.") from exc

    torch = _require_torch()
    device_obj = _resolve_device(device, torch)
    checkpoint = torch.load(checkpoint_path, map_location=device_obj)
    config = checkpoint["config"]
    data_config = _dataset_config(
        config=config,
        manifest=manifest,
        audio_embedding_root=audio_embedding_root,
        audio_encoder=audio_encoder,
        text_embedding_root=text_embedding_root,
        text_encoder=text_encoder,
        audio_embedding_key=audio_embedding_key,
        namespace=namespace,
        candidate_label_strategy=candidate_label_strategy,
        annotation_policy=annotation_policy,
    )
    ignore_index = int(data_config.get("ignore_index", -100))
    dataset = StructureEmbeddingDataset(**data_config)
    model = build_model(config.get("model", {}), checkpoint["audio_dim"], checkpoint["text_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device_obj)
    model.eval()

    cached = _cache_logits(
        model=model,
        dataset=dataset,
        device=device_obj,
        ignore_index=ignore_index,
        limit=limit,
        progress=progress,
        torch=torch,
    )
    grid = list(
        itertools.product(
            smoothing_windows,
            smoothing_modes,
            decoders,
            transition_penalties,
            boundary_weights,
            boundary_eps_values,
            min_segment_durations,
        )
    )
    rows: list[dict[str, Any]] = []
    iterator = _progress_iter(grid, enabled=progress, desc="decoding grid")
    for window, mode, decoder, penalty, boundary_weight, boundary_eps, min_duration in iterator:
        rows.append(
            _evaluate_setting(
                cached=cached,
                mir_eval=mir_eval,
                smoothing_window=window,
                smoothing_mode=mode,
                decoder=decoder,
                transition_penalty=penalty,
                boundary_weight=boundary_weight,
                boundary_eps=boundary_eps,
                min_segment_duration=min_duration,
            )
        )
    rows.sort(key=lambda row: row["F-measure@3.0_mean"], reverse=True)
    return {
        "checkpoint": str(checkpoint_path),
        "data": _jsonable(data_config),
        "num_tracks": len(cached),
        "grid_size": len(rows),
        "results": rows,
    }


def _dataset_config(
    config: dict[str, Any],
    manifest: str | Path | None,
    audio_embedding_root: str | Path | None,
    audio_encoder: str | None,
    text_embedding_root: str | Path | None,
    text_encoder: str | None,
    audio_embedding_key: str | None,
    namespace: str | None,
    candidate_label_strategy: str | None,
    annotation_policy: str | None,
) -> dict[str, Any]:
    data_config = dict(config["data"])
    validation_config = dict(config.get("validation", {}))
    data_config.update(validation_config.get("data", {}))
    if validation_config.get("manifest") is not None:
        data_config["manifest"] = validation_config["manifest"]

    overrides = {
        "manifest": manifest,
        "audio_embedding_root": audio_embedding_root,
        "audio_encoder": audio_encoder,
        "text_embedding_root": text_embedding_root,
        "text_encoder": text_encoder,
        "audio_embedding_key": audio_embedding_key,
        "namespace": namespace,
        "candidate_label_strategy": candidate_label_strategy,
    }
    for key, value in overrides.items():
        if value is not None:
            data_config[key] = value

    annotation_processing = data_config.get("annotation_processing")
    if annotation_policy is not None:
        annotation_processing = {"policy": annotation_policy}
    elif is_random_annotation_processing(annotation_processing):
        assert isinstance(annotation_processing, dict)
        annotation_processing = validation_annotation_processing_choice(annotation_processing)
    data_config["annotation_processing"] = annotation_processing
    return data_config


def _cache_logits(model, dataset, device, ignore_index: int, limit: int | None, progress: bool, torch):
    count = len(dataset) if limit is None else min(limit, len(dataset))
    cached = []
    iterator = _progress_iter(range(count), enabled=progress, desc="model logits")
    with torch.inference_mode():
        for index in iterator:
            example = dataset[index]
            audio = torch.from_numpy(example.audio).unsqueeze(0).to(device)
            text = torch.from_numpy(example.text).to(device)
            boundary_probabilities = None
            if hasattr(model, "extract_features"):
                features = model.extract_features(audio, text)
                logits = features["logits"][0].detach().cpu().numpy().astype(np.float32)
                boundary_logits = _last_boundary_logits(features)
                if boundary_logits is not None:
                    boundary_probabilities = _sigmoid(boundary_logits)
            else:
                logits = model(audio, text)[0].detach().cpu().numpy().astype(np.float32)
            reference_segments = _reference_segments_from_targets(
                intervals=example.beat_intervals,
                targets=example.targets,
                labels=example.labels,
                ignore_index=ignore_index,
            )
            if not reference_segments:
                continue
            cached.append(
                {
                    "track_id": example.track_id,
                    "dataset": example.dataset,
                    "logits": logits,
                    "boundary_probabilities": boundary_probabilities,
                    "labels": example.labels,
                    "beat_intervals": example.beat_intervals,
                    "reference_segments": reference_segments,
                }
            )
    if not cached:
        raise ValueError("No tracks could be cached for grid search")
    return cached


def _evaluate_setting(
    cached: list[dict[str, Any]],
    mir_eval,
    smoothing_window: int,
    smoothing_mode: str,
    decoder: str,
    transition_penalty: float,
    boundary_weight: float,
    boundary_eps: float,
    min_segment_duration: float,
) -> dict[str, Any]:
    metric_values = {name: [] for name in METRICS}
    segment_counts = []
    for item in cached:
        decoded_logits = smooth_logits(
            item["logits"],
            window=smoothing_window,
            mode=smoothing_mode,
        )
        item_boundary_probabilities = item.get("boundary_probabilities")
        label_indices = decode_label_indices(
            decoded_logits,
            strategy=decoder,
            transition_penalty=transition_penalty,
            boundary_probabilities=item_boundary_probabilities,
            boundary_weight=(
                boundary_weight
                if decoder == "viterbi" and item_boundary_probabilities is not None
                else 0.0
            ),
            boundary_eps=boundary_eps,
        )
        predicted_labels = [item["labels"][int(label_index)] for label_index in label_indices]
        predicted_segments = merge_frame_labels(item["beat_intervals"], predicted_labels)
        predicted_segments = remove_short_segments(
            predicted_segments,
            min_duration=min_segment_duration,
        )
        if not predicted_segments:
            continue
        segment_counts.append(len(predicted_segments))
        ref_intervals, ref_labels = _segments_to_arrays(item["reference_segments"])
        pred_intervals, pred_labels = _segments_to_arrays(predicted_segments)
        duration = max(float(ref_intervals[-1, 1]), float(pred_intervals[-1, 1]))
        ref_intervals, ref_labels = mir_eval.util.adjust_intervals(
            ref_intervals,
            list(ref_labels),
            t_min=0.0,
            t_max=duration,
        )
        pred_intervals, pred_labels = mir_eval.util.adjust_intervals(
            pred_intervals,
            list(pred_labels),
            t_min=0.0,
            t_max=duration,
        )
        _, _, f05 = mir_eval.segment.detection(
            ref_intervals,
            pred_intervals,
            window=0.5,
            trim=True,
        )
        _, _, f30 = mir_eval.segment.detection(
            ref_intervals,
            pred_intervals,
            window=3.0,
            trim=True,
        )
        _, _, pairwise_f = mir_eval.segment.pairwise(
            ref_intervals,
            ref_labels,
            pred_intervals,
            pred_labels,
        )
        _, _, nce_f = mir_eval.segment.nce(
            ref_intervals,
            ref_labels,
            pred_intervals,
            pred_labels,
        )
        metric_values["F-measure@0.5"].append(float(f05))
        metric_values["F-measure@3.0"].append(float(f30))
        metric_values["Pairwise F-measure"].append(float(pairwise_f))
        metric_values["NCE F-measure"].append(float(nce_f))

    row: dict[str, Any] = {
        "smoothing_window": int(smoothing_window),
        "smoothing_mode": smoothing_mode,
        "decoder": decoder,
        "transition_penalty": float(transition_penalty),
        "boundary_weight": float(boundary_weight),
        "boundary_eps": float(boundary_eps),
        "min_segment_duration": float(min_segment_duration),
        "num_tracks": max((len(values) for values in metric_values.values()), default=0),
        "predicted_segments_mean": _safe_mean(segment_counts),
        "predicted_segments_std": _safe_std(segment_counts),
    }
    for metric, values in metric_values.items():
        row[f"{metric}_mean"] = _safe_mean(values)
        row[f"{metric}_std"] = _safe_std(values)
    return row


def _last_boundary_logits(features: dict[str, Any]) -> np.ndarray | None:
    boundary_logits = features.get("boundary_logits")
    if not boundary_logits:
        return None
    return boundary_logits[-1][0].detach().cpu().numpy().astype(np.float32)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _reference_segments_from_targets(
    intervals: list[tuple[float, float]],
    targets: np.ndarray,
    labels: list[str],
    ignore_index: int,
) -> list[tuple[float, float, str]]:
    segments: list[tuple[float, float, str]] = []
    current: tuple[float, float, str] | None = None
    for interval, target in zip(intervals, targets):
        target_index = int(target)
        if target_index == ignore_index:
            if current is not None:
                segments.append(current)
                current = None
            continue
        start, end = interval
        label = labels[target_index]
        if current is not None and current[2] == label and np.isclose(current[1], start):
            current = (current[0], end, label)
        else:
            if current is not None:
                segments.append(current)
            current = (start, end, label)
    if current is not None:
        segments.append(current)
    return segments


def _segments_to_arrays(
    segments: list[tuple[float, float, str]],
) -> tuple[np.ndarray, list[str]]:
    intervals = np.asarray([(start, end) for start, end, _ in segments], dtype=float)
    labels = [label for _, _, label in segments]
    return intervals, labels


def save_results(path: str | Path, results: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
        handle.write("\n")


def save_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_top_results(rows: list[dict[str, Any]], top_k: int = 10) -> None:
    print(f"Grid points: {len(rows)}")
    for rank, row in enumerate(rows[:top_k], start=1):
        print(
            f"{rank:02d} "
            f"F@3={row['F-measure@3.0_mean']:.4f} "
            f"F@0.5={row['F-measure@0.5_mean']:.4f} "
            f"Pairwise={row['Pairwise F-measure_mean']:.4f} "
            f"NCE={row['NCE F-measure_mean']:.4f} "
            f"segments={row['predicted_segments_mean']:.1f} "
            f"window={row['smoothing_window']} "
            f"mode={row['smoothing_mode']} "
            f"decoder={row['decoder']} "
            f"penalty={row['transition_penalty']} "
            f"boundary_weight={row['boundary_weight']} "
            f"boundary_eps={row['boundary_eps']} "
            f"min_dur={row['min_segment_duration']}"
        )


def _progress_iter(iterable, enabled: bool, desc: str):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, desc=desc, dynamic_ncols=True)


def _safe_mean(values) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.nanmean(np.asarray(values, dtype=float)))


def _safe_std(values) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.nanstd(np.asarray(values, dtype=float)))


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _resolve_device(device: str, torch):
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install torch to run decoding grid search.") from exc
    return torch


if __name__ == "__main__":
    main()
