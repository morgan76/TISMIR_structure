#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from tismir.data.annotations import (
    concrete_annotation_processing_choices,
    is_random_annotation_processing,
)
from tismir.data.manifest import load_manifest, save_manifest
from tismir.evaluation import evaluate_prediction_manifest, format_evaluation, save_evaluation
from tismir.inference import run_baseline_inference
from tismir.io import load_yaml
from tismir.diagnostics import run_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run inference/evaluation/diagnostics for every annotation policy choice "
            "in a training config and create per-track side-by-side comparison plots."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--comparison-limit", type=int, default=None)
    parser.add_argument("--diagnostic-max-plots", type=int, default=0)
    parser.add_argument("--audio-audio-max-frames", type=int, default=512)
    parser.add_argument("--skip-runs", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    data_config = dict(config.get("data", {}))
    validation_config = dict(config.get("validation", {}))
    segmentation_config = dict(validation_config.get("segmentation", {}))
    reference_namespace = data_config.get("namespace", "segment_open")
    prediction_namespace = data_config.get("prediction_namespace", "segment_open")

    checkpoint = Path(args.checkpoint) if args.checkpoint else _default_checkpoint(config)
    manifest = Path(
        args.manifest
        or validation_config.get("manifest")
        or data_config.get("manifest")
    )
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = _limited_manifest(manifest, output_dir=output_dir, limit=args.limit)

    policies = _annotation_policy_choices(data_config.get("annotation_processing"))
    policy_records = []
    for policy in policies:
        policy_name = _policy_name(policy)
        policy_dir = output_dir / policy_name
        infer_dir = policy_dir / "infer"
        diagnostics_dir = policy_dir / "diagnostics"
        evaluation_path = policy_dir / "evaluation.json"
        policy_dir.mkdir(parents=True, exist_ok=True)

        if not args.skip_runs:
            print(f"\n== {policy_name}: inference ==")
            run_baseline_inference(
                checkpoint_path=checkpoint,
                manifest=run_manifest,
                audio_embedding_root=data_config["audio_embedding_root"],
                audio_encoder=data_config["audio_encoder"],
                text_embedding_root=data_config["text_embedding_root"],
                text_encoder=data_config["text_encoder"],
                audio_embedding_key=data_config.get("audio_embedding_key", "beat_sync"),
                namespace=reference_namespace,
                prediction_namespace=prediction_namespace,
                output_dir=infer_dir,
                device=args.device,
                limit=None,
                candidate_label_strategy=data_config.get(
                    "candidate_label_strategy",
                    "track_labels",
                ),
                annotation_processing=policy,
                beat_subsampling=data_config.get("beat_subsampling"),
                track_filter=data_config.get("track_filter"),
                smoothing_window=int(segmentation_config.get("smoothing_window", 7)),
                smoothing_mode=str(segmentation_config.get("smoothing_mode", "mean")),
                decoder=str(segmentation_config.get("decoder", "viterbi")),
                transition_penalty=float(segmentation_config.get("transition_penalty", 8.0)),
                min_segment_duration=float(segmentation_config.get("min_segment_duration", 0.0)),
            )

            print(f"\n== {policy_name}: evaluation ==")
            evaluation = evaluate_prediction_manifest(
                reference_manifest=run_manifest,
                predictions_root=infer_dir,
                namespace=reference_namespace,
                prediction_namespace=prediction_namespace,
                trim=bool(segmentation_config.get("trim", True)),
                reference_annotation_processing=policy,
                audio_embedding_root=data_config["audio_embedding_root"],
                audio_encoder=data_config["audio_encoder"],
                text_embedding_root=data_config["text_embedding_root"],
                text_encoder=data_config["text_encoder"],
                audio_embedding_key=data_config.get("audio_embedding_key", "beat_sync"),
                candidate_label_strategy=data_config.get(
                    "candidate_label_strategy",
                    "track_labels",
                ),
                beat_subsampling=data_config.get("beat_subsampling"),
                reference_track_filter=data_config.get("track_filter"),
                ignore_index=int(data_config.get("ignore_index", -100)),
            )
            print(format_evaluation(evaluation))
            save_evaluation(evaluation_path, evaluation)

            print(f"\n== {policy_name}: diagnostics ==")
            run_diagnostics(
                checkpoint_path=checkpoint,
                manifest=run_manifest,
                audio_embedding_root=data_config["audio_embedding_root"],
                audio_encoder=data_config["audio_encoder"],
                text_embedding_root=data_config["text_embedding_root"],
                text_encoder=data_config["text_encoder"],
                audio_embedding_key=data_config.get("audio_embedding_key", "beat_sync"),
                namespace=reference_namespace,
                output_dir=diagnostics_dir,
                device=args.device,
                limit=None,
                candidate_label_strategy=data_config.get(
                    "candidate_label_strategy",
                    "track_labels",
                ),
                annotation_processing=policy,
                beat_subsampling=data_config.get("beat_subsampling"),
                track_filter=data_config.get("track_filter"),
                max_plots=args.diagnostic_max_plots,
                audio_audio_max_frames=args.audio_audio_max_frames,
            )

        policy_records.append(
            {
                "policy": policy_name,
                "annotation_processing": policy,
                "infer_dir": str(infer_dir),
                "diagnostics_dir": str(diagnostics_dir),
                "evaluation_path": str(evaluation_path),
            }
        )

    _save_policy_metric_table(output_dir / "policy_metrics.csv", policy_records)
    _save_summary_json(
        output_dir / "summary.json",
        checkpoint=checkpoint,
        manifest=manifest,
        run_manifest=run_manifest,
        policies=policy_records,
    )
    _save_comparison_plots(
        output_dir=output_dir / "comparisons",
        policies=policy_records,
        comparison_limit=args.comparison_limit,
    )
    print(f"\nSaved policy comparison outputs to {output_dir}")


def _annotation_policy_choices(annotation_processing: Any) -> list[str | dict[str, Any]]:
    if isinstance(annotation_processing, dict) and is_random_annotation_processing(annotation_processing):
        return concrete_annotation_processing_choices(annotation_processing)
    if annotation_processing is None:
        return [{"policy": "keep"}]
    return [annotation_processing]


def _default_checkpoint(config: dict[str, Any]) -> Path:
    output_dir = Path(config.get("output_dir", "outputs/train/baseline"))
    for name in ("best_segmentation_checkpoint.pt", "best_checkpoint.pt", "checkpoint.pt"):
        path = output_dir / name
        if path.exists():
            return path
    return output_dir / "best_segmentation_checkpoint.pt"


def _default_output_dir(checkpoint: Path) -> Path:
    return Path("outputs/annotation_policy_comparison") / checkpoint.parent.name


def _limited_manifest(manifest: Path, output_dir: Path, limit: int | None) -> Path:
    if limit is None:
        return manifest
    tracks = load_manifest(manifest)[:limit]
    limited_manifest = output_dir / f"manifest.first_{limit}.jsonl"
    save_manifest(limited_manifest, tracks)
    return limited_manifest


def _policy_name(policy: str | dict[str, Any]) -> str:
    if isinstance(policy, str):
        return _slug(policy)
    label = str(policy.get("policy", "policy"))
    extras = []
    for key in (
        "separator",
        "start_index",
        "annotation_selection",
        "projected_function_policy",
    ):
        if key in policy:
            extras.append(f"{key}-{policy[key]}")
    return _slug("_".join([label, *extras]))


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value)


def _save_policy_metric_table(path: Path, policies: list[dict[str, Any]]) -> None:
    metrics = [
        "F-measure@0.5",
        "F-measure@3.0",
        "Acc",
        "Balanced Acc",
        "Pairwise F-measure",
        "NCE F-measure",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["policy", *metrics])
        writer.writeheader()
        for record in policies:
            evaluation_path = Path(record["evaluation_path"])
            if not evaluation_path.exists():
                writer.writerow({"policy": record["policy"]})
                continue
            evaluation = _load_json(evaluation_path)
            row = {"policy": record["policy"]}
            for metric in metrics:
                if metric in evaluation["summary"]:
                    row[metric] = f"{evaluation['summary'][metric]['mean']:.6f}"
            writer.writerow(row)


def _save_summary_json(
    path: Path,
    checkpoint: Path,
    manifest: Path,
    run_manifest: Path,
    policies: list[dict[str, Any]],
) -> None:
    payload = {
        "checkpoint": str(checkpoint),
        "manifest": str(manifest),
        "run_manifest": str(run_manifest),
        "policies": policies,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _save_comparison_plots(
    output_dir: Path,
    policies: list[dict[str, Any]],
    comparison_limit: int | None,
) -> None:
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional plotting dependency
        print("matplotlib is not installed; skipping side-by-side comparison plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    tracks = _common_tracks(policies)
    if comparison_limit is not None:
        tracks = tracks[:comparison_limit]
    for dataset, track_id in tracks:
        policy_payloads = []
        for record in policies:
            diagnostics_path = (
                Path(record["diagnostics_dir"])
                / dataset
                / track_id
                / "diagnostics.npz"
            )
            prediction_path = (
                Path(record["infer_dir"])
                / dataset
                / f"{track_id}.json"
            )
            if not diagnostics_path.exists() or not prediction_path.exists():
                continue
            policy_payloads.append(
                {
                    "policy": record["policy"],
                    "diagnostics": _load_npz(diagnostics_path),
                    "prediction": _load_json(prediction_path),
                }
            )
        if not policy_payloads:
            continue
        track_dir = output_dir / dataset / track_id
        track_dir.mkdir(parents=True, exist_ok=True)
        _plot_track_policy_comparison(
            plt=plt,
            path=track_dir / "policy_comparison.png",
            dataset=dataset,
            track_id=track_id,
            payloads=policy_payloads,
        )
        print(f"{track_id}: saved {track_dir / 'policy_comparison.png'}")


def _common_tracks(policies: list[dict[str, Any]]) -> list[tuple[str, str]]:
    track_sets = []
    ordered_tracks: list[tuple[str, str]] = []
    for record in policies:
        manifest_path = Path(record["infer_dir"]) / "manifest.json"
        if not manifest_path.exists():
            continue
        entries = _load_json(manifest_path)
        tracks = [(entry["dataset"], entry["track_id"]) for entry in entries]
        track_sets.append(set(tracks))
        if not ordered_tracks:
            ordered_tracks = tracks
    if not track_sets:
        return []
    common = set.intersection(*track_sets)
    return [track for track in ordered_tracks if track in common]


def _plot_track_policy_comparison(
    plt,
    path: Path,
    dataset: str,
    track_id: str,
    payloads: list[dict[str, Any]],
) -> None:
    num_policies = len(payloads)
    fig = plt.figure(figsize=(7.0 * num_policies, 15.0), constrained_layout=True)
    grid = fig.add_gridspec(
        nrows=4,
        ncols=num_policies,
        height_ratios=[0.45, 0.45, 4.0, 4.0],
    )
    fig.suptitle(f"{dataset}/{track_id}: annotation-policy comparison", fontsize=16)

    for col, payload in enumerate(payloads):
        diagnostics = payload["diagnostics"]
        prediction = payload["prediction"]
        labels = [str(label) for label in diagnostics["labels"].tolist()]
        targets = diagnostics["targets"].astype(int)
        beat_intervals = diagnostics["beat_intervals"].astype(float)
        beat_times = _beat_start_times(beat_intervals)
        predicted_indices = np.asarray(
            prediction.get("frame_label_indices", diagnostics["predictions"]),
            dtype=int,
        )
        if len(predicted_indices) != len(targets):
            predicted_indices = diagnostics["predictions"].astype(int)
        audio_indices = diagnostics["audio_audio_indices"].astype(int)
        sampled_times = beat_times[audio_indices] if len(beat_times) else np.arange(len(audio_indices))
        boundaries = np.flatnonzero(_boundary_mask(targets)) + 1
        sampled_boundaries = np.searchsorted(audio_indices, boundaries)

        cmap = _label_colormap(plt, len(labels))
        ref_ax = fig.add_subplot(grid[0, col])
        pred_ax = fig.add_subplot(grid[1, col], sharex=ref_ax)
        prob_ax = fig.add_subplot(grid[2, col])
        audio_ax = fig.add_subplot(grid[3, col])

        ref_values = np.where(targets == -100, len(labels), targets)[None, :]
        pred_values = np.where(
            (predicted_indices < 0) | (predicted_indices >= len(labels)),
            len(labels),
            predicted_indices,
        )[None, :]
        ref_ax.imshow(
            ref_values,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=-0.5,
            vmax=len(labels) + 0.5,
        )
        pred_ax.imshow(
            pred_values,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=-0.5,
            vmax=len(labels) + 0.5,
        )
        for ax in (ref_ax, pred_ax):
            for boundary in boundaries:
                ax.axvline(boundary - 0.5, color="black", linewidth=0.25, alpha=0.3)
            ax.set_yticks([])
            ax.tick_params(axis="x", labelbottom=False)
        ref_ax.set_title(payload["policy"])
        ref_ax.set_ylabel("ref", rotation=0, ha="right", va="center")
        pred_ax.set_ylabel("pred", rotation=0, ha="right", va="center")
        _annotate_label_strip(ref_ax, targets, labels)
        _annotate_label_strip(pred_ax, predicted_indices, labels)

        prob_image = prob_ax.imshow(
            diagnostics["probability_self_similarity"],
            vmin=0.0,
            vmax=1.0,
            cmap="magma",
            origin="lower",
            interpolation="nearest",
        )
        _decorate_similarity_axis(
            ax=prob_ax,
            title="Predicted same-label matrix",
            times=sampled_times,
            boundaries=sampled_boundaries,
        )
        fig.colorbar(prob_image, ax=prob_ax, fraction=0.046, pad=0.04)

        audio_image = audio_ax.imshow(
            diagnostics["audio_audio_similarity"],
            vmin=-1.0,
            vmax=1.0,
            cmap="coolwarm",
            origin="lower",
            interpolation="nearest",
        )
        _decorate_similarity_axis(
            ax=audio_ax,
            title="Audio-token self-similarity",
            times=sampled_times,
            boundaries=sampled_boundaries,
        )
        fig.colorbar(audio_image, ax=audio_ax, fraction=0.046, pad=0.04)

    fig.savefig(path, dpi=170)
    plt.close(fig)


def _decorate_similarity_axis(ax, title: str, times: np.ndarray, boundaries: np.ndarray) -> None:
    for boundary in boundaries:
        if 0 <= boundary < len(times):
            ax.axvline(boundary - 0.5, color="white", linewidth=0.35, alpha=0.55)
            ax.axhline(boundary - 0.5, color="white", linewidth=0.35, alpha=0.55)
    ax.set_title(title)
    _set_time_axis(ax, times, axis="x")
    _set_time_axis(ax, times, axis="y")
    ax.set_xlabel("time (min:ss)")
    ax.set_ylabel("time (min:ss)")


def _label_colormap(plt, num_labels: int):
    from matplotlib.colors import ListedColormap

    base = plt.get_cmap("tab20", max(1, min(20, num_labels)))
    colors = [base(index % 20) for index in range(num_labels)]
    colors.append((0.86, 0.86, 0.86, 1.0))
    return ListedColormap(colors)


def _annotate_label_strip(ax, values: np.ndarray, labels: list[str], min_width: int = 24) -> None:
    start = 0
    while start < len(values):
        label_index = int(values[start])
        end = start + 1
        while end < len(values) and int(values[end]) == label_index:
            end += 1
        if 0 <= label_index < len(labels) and end - start >= min_width:
            ax.text(
                (start + end - 1) / 2,
                0,
                labels[label_index],
                ha="center",
                va="center",
                fontsize=7,
                color="black",
                clip_on=True,
            )
        start = end


def _boundary_mask(targets: np.ndarray) -> np.ndarray:
    if len(targets) < 2:
        return np.asarray([], dtype=bool)
    current = targets[:-1]
    following = targets[1:]
    valid = (current != -100) & (following != -100)
    return valid & (current != following)


def _beat_start_times(beat_intervals: np.ndarray) -> np.ndarray:
    if beat_intervals.size == 0:
        return np.asarray([], dtype=np.float32)
    if beat_intervals.ndim != 2 or beat_intervals.shape[1] < 1:
        return np.arange(len(beat_intervals), dtype=np.float32)
    return beat_intervals[:, 0].astype(np.float32)


def _set_time_axis(ax, times: np.ndarray, axis: str, max_ticks: int = 7) -> None:
    if len(times) == 0:
        return
    tick_count = min(max_ticks, len(times))
    positions = np.unique(np.linspace(0, len(times) - 1, tick_count).round().astype(int))
    labels = [_format_timestamp(float(times[position])) for position in positions]
    if axis == "x":
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=8)
    elif axis == "y":
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=8)
    else:
        raise ValueError("axis must be 'x' or 'y'")


def _format_timestamp(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as payload:
        return {key: payload[key] for key in payload.files}


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    main()
