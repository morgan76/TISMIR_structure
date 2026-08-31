#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from tismir.data.manifest import load_manifest, save_manifest
from tismir.evaluation import evaluate_prediction_manifest, save_evaluation
from tismir.inference import run_baseline_inference
from tismir.io import load_yaml, save_json
from tismir.training import train_projection_baseline


BASE_CONFIG = "configs/benchmarks/harmonix_mert_text_conditioned_ce.yaml"

SETUPS: dict[str, dict[str, Any]] = {
    "e5_base_v2": {
        "text_embedding_root": "data/embeddings/text_harmonix_base_labels",
        "text_encoder": "sentence_transformers",
        "annotation_processing": {"policy": "base_labels"},
    },
    "mpnet_base_v2": {
        "text_embedding_root": "data/embeddings/text_harmonix_base_labels_mpnet",
        "text_encoder": "sentence_transformers",
        "annotation_processing": {"policy": "base_labels"},
    },
    "clap_music_speech": {
        "text_embedding_root": "data/embeddings/text_harmonix_base_labels_clap_music_speech",
        "text_encoder": "clap",
        "annotation_processing": {"policy": "base_labels"},
    },
    "muq_mulan_large": {
        "text_embedding_root": "data/embeddings/text_harmonix_base_labels_muq_mulan",
        "text_encoder": "muq_mulan",
        "annotation_processing": {"policy": "base_labels"},
    },
}

METRICS = (
    "F-measure@0.5",
    "F-measure@3.0",
    "Acc",
    "Balanced Acc",
    "Pairwise F-measure",
    "NCE F-measure",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train/evaluate the simple Harmonix CE model with different text encoders."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/benchmarks/harmonix_text_encoder_ablation",
    )
    parser.add_argument(
        "--setups",
        nargs="+",
        choices=sorted(SETUPS),
        default=list(SETUPS),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-existing-training", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for setup_name in args.setups:
        print(f"\n=== {setup_name} ===")
        setup = SETUPS[setup_name]
        config = _training_config(
            setup_name=setup_name,
            setup=setup,
            output_dir=output_dir,
            device=args.device,
            max_epochs=args.max_epochs,
            progress=not args.no_progress,
            num_workers=args.num_workers,
        )
        metrics, train_seconds = _train_or_load(
            config=config,
            skip_training=args.skip_training,
            skip_existing=args.skip_existing_training,
        )
        checkpoint = _checkpoint_from_metrics_or_config(metrics, config)
        row = {
            "condition": setup_name,
            "checkpoint": str(checkpoint),
            **_training_columns(metrics, train_seconds),
        }
        split_row = _run_eval(
            setup_name=setup_name,
            checkpoint=checkpoint,
            limit=args.eval_limit,
            output_dir=output_dir,
            device=args.device,
            config=config,
            setup=setup,
        )
        for key, value in split_row.items():
            row[f"eval_{key}"] = value
        rows.append(row)
        _write_results(output_dir, rows)

    print(f"\nSaved benchmark table to {output_dir / 'results.md'}")


def _training_config(
    setup_name: str,
    setup: dict[str, Any],
    output_dir: Path,
    device: str,
    max_epochs: int | None,
    progress: bool,
    num_workers: int | None,
) -> dict[str, Any]:
    config = load_yaml(BASE_CONFIG)
    config["device"] = device
    config["output_dir"] = str(output_dir / "train" / setup_name)
    config.setdefault("data", {})["text_embedding_root"] = setup["text_embedding_root"]
    config["data"]["text_encoder"] = setup["text_encoder"]
    config["data"]["annotation_processing"] = setup["annotation_processing"]
    config.setdefault("optimization", {})["progress"] = bool(progress)
    if max_epochs is not None:
        config["optimization"]["max_epochs"] = int(max_epochs)
    if num_workers is not None:
        config["optimization"]["num_workers"] = int(num_workers)
        if int(num_workers) <= 0:
            config["optimization"]["persistent_workers"] = False
            config["optimization"]["prefetch_factor"] = None
    return config


def _train_or_load(
    config: dict[str, Any],
    skip_training: bool,
    skip_existing: bool,
) -> tuple[dict[str, Any], float | None]:
    metrics_path = Path(config["output_dir"]) / "metrics.json"
    if skip_training or (skip_existing and metrics_path.exists()):
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics file for skipped training: {metrics_path}")
        return json.loads(metrics_path.read_text(encoding="utf-8")), None

    start = time.perf_counter()
    metrics = train_projection_baseline(config)
    return metrics, time.perf_counter() - start


def _checkpoint_from_metrics_or_config(metrics: dict[str, Any], config: dict[str, Any]) -> Path:
    output_dir = Path(config["output_dir"])
    candidates = [
        metrics.get("best_segmentation_checkpoint"),
        metrics.get("best_checkpoint"),
        metrics.get("checkpoint"),
        output_dir / "best_segmentation_checkpoint.pt",
        output_dir / "best_checkpoint.pt",
        output_dir / "checkpoint.pt",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    raise FileNotFoundError(f"No checkpoint found for {output_dir}")


def _run_eval(
    setup_name: str,
    checkpoint: Path,
    limit: int | None,
    output_dir: Path,
    device: str,
    config: dict[str, Any],
    setup: dict[str, Any],
) -> dict[str, Any]:
    data_config = config["data"]
    validation = dict(config.get("validation", {}))
    segmentation = dict(validation.get("segmentation", {}))
    manifest = Path(validation.get("manifest", data_config["manifest"]))
    run_manifest = _limited_manifest(manifest, output_dir / "manifests", limit)
    run_dir = output_dir / setup_name / "eval"
    predictions_dir = run_dir / "predictions"
    evaluation_path = run_dir / "evaluation.json"

    run_baseline_inference(
        checkpoint_path=checkpoint,
        manifest=run_manifest,
        audio_embedding_root=data_config["audio_embedding_root"],
        audio_encoder=data_config["audio_encoder"],
        text_embedding_root=setup["text_embedding_root"],
        text_encoder=setup["text_encoder"],
        audio_embedding_key=data_config.get("audio_embedding_key", "beat_sync"),
        namespace=data_config.get("namespace", "segment_open"),
        prediction_namespace=data_config.get("prediction_namespace", "segment_open"),
        output_dir=predictions_dir,
        device=device,
        limit=None,
        candidate_label_strategy=data_config.get("candidate_label_strategy", "track_labels"),
        annotation_processing=setup["annotation_processing"],
        beat_subsampling=data_config.get("beat_subsampling"),
        track_filter=data_config.get("track_filter"),
        smoothing_window=int(segmentation.get("smoothing_window", 7)),
        smoothing_mode=str(segmentation.get("smoothing_mode", "mean")),
        decoder=str(segmentation.get("decoder", "viterbi")),
        transition_penalty=float(segmentation.get("transition_penalty", 4.0)),
        min_segment_duration=float(segmentation.get("min_segment_duration", 3.0)),
        boundary_decoding=False,
        boundary_peak=None,
        use_checkpoint_annotation_processing=False,
    )

    evaluation = evaluate_prediction_manifest(
        reference_manifest=run_manifest,
        predictions_root=predictions_dir,
        namespace=data_config.get("namespace", "segment_open"),
        prediction_namespace=data_config.get("prediction_namespace", "segment_open"),
        trim=bool(segmentation.get("trim", True)),
        reference_annotation_processing=setup["annotation_processing"],
        audio_embedding_root=data_config["audio_embedding_root"],
        audio_encoder=data_config["audio_encoder"],
        text_embedding_root=setup["text_embedding_root"],
        text_encoder=setup["text_encoder"],
        audio_embedding_key=data_config.get("audio_embedding_key", "beat_sync"),
        candidate_label_strategy=data_config.get("candidate_label_strategy", "track_labels"),
        beat_subsampling=data_config.get("beat_subsampling"),
        reference_track_filter=data_config.get("track_filter"),
        ignore_index=int(data_config.get("ignore_index", -100)),
    )
    save_evaluation(evaluation_path, evaluation)

    row = {
        "num_tracks": evaluation["num_tracks"],
        "predictions_dir": str(predictions_dir),
        "evaluation_json": str(evaluation_path),
    }
    for metric in METRICS:
        values = evaluation["summary"].get(metric, {})
        row[metric] = values.get("mean")
        row[f"{metric} std"] = values.get("std")
    return row


def _limited_manifest(manifest: Path, output_dir: Path, limit: int | None) -> Path:
    if limit is None:
        return manifest
    output_dir.mkdir(parents=True, exist_ok=True)
    tracks = load_manifest(manifest)[:limit]
    limited = output_dir / f"{manifest.stem}.first_{limit}.jsonl"
    save_manifest(limited, tracks)
    return limited


def _training_columns(metrics: dict[str, Any], train_seconds: float | None) -> dict[str, Any]:
    return {
        "best_val_loss": metrics.get("best_val_loss"),
        "best_epoch": metrics.get("best_epoch"),
        "best_segmentation_score": metrics.get("best_segmentation_score"),
        "best_segmentation_epoch": metrics.get("best_segmentation_epoch"),
        "epochs_trained": metrics.get("epochs_trained"),
        "train_seconds": train_seconds,
    }


def _write_results(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    columns = _result_columns()
    with (output_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _format_cell(row.get(column)) for column in columns})
    (output_dir / "results.md").write_text(_markdown_table(rows, columns), encoding="utf-8")
    save_json(output_dir / "results.json", {"rows": rows})
    _plot_results(output_dir, rows)


def _result_columns() -> list[str]:
    columns = ["condition", "eval_num_tracks"]
    columns.extend(f"eval_{metric}" for metric in METRICS)
    columns.extend(
        [
            "best_val_loss",
            "best_epoch",
            "best_segmentation_score",
            "best_segmentation_epoch",
            "epochs_trained",
            "train_seconds",
            "checkpoint",
            "eval_evaluation_json",
        ]
    )
    return columns


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(_pretty_column(column) for column in columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row.get(column)) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def _pretty_column(column: str) -> str:
    exact = {
        "condition": "Condition",
        "best_val_loss": "Best Val Loss",
        "best_epoch": "Best Epoch",
        "best_segmentation_score": "Best Seg",
        "best_segmentation_epoch": "Best Seg Epoch",
        "epochs_trained": "Epochs",
        "train_seconds": "Train s",
        "checkpoint": "Checkpoint",
        "eval_evaluation_json": "Eval JSON",
    }
    if column in exact:
        return exact[column]
    metric_names = {
        "num_tracks": "Tracks",
        "F-measure@0.5": "F@0.5",
        "F-measure@3.0": "F@3.0",
        "Acc": "Acc",
        "Balanced Acc": "Bal Acc",
        "Pairwise F-measure": "PFC",
        "NCE F-measure": "NCE",
    }
    prefix = "eval_"
    if column.startswith(prefix):
        metric = column.removeprefix(prefix)
        return f"Eval {metric_names.get(metric, metric)}"
    return column


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _plot_results(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    plt = _matplotlib()
    conditions = [str(row["condition"]) for row in rows]
    metric_labels = {
        "F-measure@0.5": "F@0.5",
        "F-measure@3.0": "F@3.0",
        "Acc": "Acc",
        "Balanced Acc": "Bal Acc",
        "Pairwise F-measure": "PFC",
        "NCE F-measure": "NCE",
    }

    fig, ax = plt.subplots(figsize=(12.5, 5.4), constrained_layout=True)
    x = np.arange(len(conditions), dtype=float)
    width = 0.12
    offsets = (np.arange(len(METRICS), dtype=float) - (len(METRICS) - 1) / 2.0) * width
    for offset, metric in zip(offsets, METRICS, strict=True):
        values = [_float_or_nan(row.get(f"eval_{metric}")) for row in rows]
        ax.bar(x + offset, values, width=width, label=metric_labels[metric])
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Mean score")
    ax.set_title("Harmonix eval metrics by text encoder")
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.legend(ncols=3, frameon=False)
    fig.savefig(output_dir / "eval_metrics_barplot.png", dpi=180)
    plt.close(fig)

    values = np.array(
        [[_float_or_nan(row.get(f"eval_{metric}")) for metric in METRICS] for row in rows],
        dtype=float,
    )
    scaled = np.zeros_like(values)
    for col in range(values.shape[1]):
        column = values[:, col]
        finite = np.isfinite(column)
        if not np.any(finite):
            scaled[:, col] = np.nan
            continue
        lo = float(np.nanmin(column))
        hi = float(np.nanmax(column))
        if hi > lo:
            scaled[:, col] = (column - lo) / (hi - lo)
        else:
            scaled[:, col] = 0.5

    fig_height = max(3.5, 0.55 * len(conditions) + 1.8)
    fig, ax = plt.subplots(figsize=(10.5, fig_height), constrained_layout=True)
    ax.imshow(scaled, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(METRICS)))
    ax.set_xticklabels([metric_labels[metric] for metric in METRICS], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(conditions)))
    ax.set_yticklabels(conditions)
    ax.set_title("Harmonix eval metrics, column-normalized")
    for y, row in enumerate(rows):
        for x_idx, metric in enumerate(METRICS):
            value = _float_or_nan(row.get(f"eval_{metric}"))
            text = "" if not np.isfinite(value) else f"{value:.3f}"
            ax.text(x_idx, y, text, ha="center", va="center", color="white", fontsize=9)
    fig.savefig(output_dir / "metrics_heatmap.png", dpi=180)
    plt.close(fig)


def _float_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _matplotlib():
    cache_root = Path(tempfile.gettempdir()) / "tismir_plot_cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


if __name__ == "__main__":
    main()
