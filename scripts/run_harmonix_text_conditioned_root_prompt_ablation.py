#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from tismir.data.manifest import load_manifest, save_manifest
from tismir.evaluation import evaluate_prediction_manifest, save_evaluation
from tismir.inference import run_baseline_inference
from tismir.io import load_yaml, save_json
from tismir.training import train_projection_baseline


TRAINING_CONFIG = "configs/benchmarks/harmonix_mert_text_conditioned_ce.yaml"

LABEL_SETUPS: dict[str, dict[str, Any]] = {
    "root": {
        "text_embedding_root": "data/embeddings/text_harmonix_base_labels",
        "annotation_processing": {"policy": "base_labels"},
    },
    "corpus_ids": {
        "text_embedding_root": "data/embeddings/text_harmonix_base_section_ids_corpus",
        "annotation_processing": {
            "policy": "section_ids_corpus",
            "source_annotation_processing": {"policy": "base_labels"},
            "section_id_mapping_path": (
                "configs/annotation_mappings/harmonix_section_ids_corpus_base_labels_seed0.json"
            ),
            "section_id_preserve_labels": ["silence"],
        },
    },
    "definition": {
        "text_embedding_root": "data/embeddings/text_harmonix_base_definition",
        "annotation_processing": {"policy": "base_labels"},
    },
    "compact_definition": {
        "text_embedding_root": "data/embeddings/text_harmonix_base_compact_definition",
        "annotation_processing": {"policy": "base_labels"},
    },
}

CONDITIONS = (
    ("root", "root"),
    ("corpus_ids", "corpus_ids"),
    ("definition", "definition"),
    ("compact_definition", "compact_definition"),
)

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
        description=(
            "Run the Harmonix root-label prompt ablation with the text-conditioned "
            "CE-only model."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/benchmarks/harmonix_text_conditioned_ce_root_prompt_ablation",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-existing-training", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override optimization.num_workers in all training configs.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = _load_training_configs(
        output_dir=output_dir,
        device=args.device,
        max_epochs=args.max_epochs,
        progress=not args.no_progress,
        num_workers=args.num_workers,
    )

    checkpoints: dict[str, Path] = {}
    training_metrics: dict[str, dict[str, Any]] = {}
    training_seconds: dict[str, float | None] = {}
    for setup_name, config in configs.items():
        print(f"\n=== train {setup_name} ===")
        metrics, seconds = _train_or_load(
            config=config,
            skip_training=args.skip_training,
            skip_existing=args.skip_existing_training,
        )
        checkpoints[setup_name] = _checkpoint_from_metrics_or_config(metrics, config)
        training_metrics[setup_name] = metrics
        training_seconds[setup_name] = seconds
        print(f"{setup_name}: using checkpoint {checkpoints[setup_name]}")

    rows = []
    for train_setup, inference_setup in CONDITIONS:
        print(f"\n=== {train_setup} -> {inference_setup} ===")
        row = _run_condition(
            train_setup=train_setup,
            inference_setup=inference_setup,
            checkpoint=checkpoints[train_setup],
            training_config=configs[train_setup],
            output_dir=output_dir,
            device=args.device,
            eval_limit=args.eval_limit,
        )
        row.update(_training_columns(training_metrics[train_setup], training_seconds[train_setup]))
        rows.append(row)

    _write_results(output_dir, rows)
    print(f"\nSaved condition table to {output_dir / 'results.md'}")


def _load_training_configs(
    output_dir: Path,
    device: str,
    max_epochs: int | None,
    progress: bool,
    num_workers: int | None,
) -> dict[str, dict[str, Any]]:
    configs = {}
    for setup_name, label_setup in LABEL_SETUPS.items():
        config = load_yaml(TRAINING_CONFIG)
        config["device"] = device
        config["output_dir"] = str(output_dir / "train" / setup_name)
        config["data"]["text_embedding_root"] = label_setup["text_embedding_root"]
        config["data"]["annotation_processing"] = label_setup["annotation_processing"]
        config.setdefault("optimization", {})["progress"] = bool(progress)
        if num_workers is not None:
            config["optimization"]["num_workers"] = int(num_workers)
            if int(num_workers) <= 0:
                config["optimization"]["persistent_workers"] = False
                config["optimization"]["prefetch_factor"] = None
        if max_epochs is not None:
            config["optimization"]["max_epochs"] = int(max_epochs)
        configs[setup_name] = config
    return configs


def _train_or_load(
    config: dict[str, Any],
    skip_training: bool,
    skip_existing: bool,
) -> tuple[dict[str, Any], float | None]:
    metrics_path = Path(config["output_dir"]) / "metrics.json"
    if skip_training or (skip_existing and metrics_path.exists()):
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics file for skipped training: {metrics_path}")
        return json.loads(metrics_path.read_text()), None

    start = time.perf_counter()
    metrics = train_projection_baseline(config)
    return metrics, time.perf_counter() - start


def _checkpoint_from_metrics_or_config(metrics: dict[str, Any], config: dict[str, Any]) -> Path:
    candidates = [
        metrics.get("best_segmentation_checkpoint"),
        metrics.get("best_checkpoint"),
        metrics.get("checkpoint"),
    ]
    output_dir = Path(config["output_dir"])
    candidates.extend(
        str(output_dir / name)
        for name in ("best_segmentation_checkpoint.pt", "best_checkpoint.pt", "checkpoint.pt")
    )
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    raise FileNotFoundError(f"No checkpoint found for {output_dir}")


def _run_condition(
    train_setup: str,
    inference_setup: str,
    checkpoint: Path,
    training_config: dict[str, Any],
    output_dir: Path,
    device: str,
    eval_limit: int | None,
) -> dict[str, Any]:
    validation = dict(training_config.get("validation", {}))
    data_config = dict(training_config["data"])
    segmentation = dict(validation.get("segmentation", {}))
    label_setup = LABEL_SETUPS[inference_setup]
    condition_name = f"{train_setup} -> {inference_setup}"
    eval_manifest = Path(validation.get("manifest", data_config["manifest"]))

    row: dict[str, Any] = {
        "condition": condition_name,
        "train_labels": train_setup,
        "inference_labels": inference_setup,
        "checkpoint": str(checkpoint),
    }
    split_row = _run_split(
        condition_name=condition_name,
        checkpoint=checkpoint,
        manifest=eval_manifest,
        limit=eval_limit,
        output_dir=output_dir,
        device=device,
        data_config=data_config,
        segmentation=segmentation,
        label_setup=label_setup,
    )
    for key, value in split_row.items():
        row[f"eval_{key}"] = value
    return row


def _run_split(
    condition_name: str,
    checkpoint: Path,
    manifest: Path,
    limit: int | None,
    output_dir: Path,
    device: str,
    data_config: dict[str, Any],
    segmentation: dict[str, Any],
    label_setup: dict[str, Any],
) -> dict[str, Any]:
    if not manifest.exists():
        raise FileNotFoundError(f"Missing eval manifest: {manifest}")
    run_manifest = _limited_manifest(manifest, output_dir / "manifests", limit)
    safe_condition = condition_name.replace(" -> ", "_to_").replace(" ", "_")
    run_dir = output_dir / safe_condition / "eval"
    predictions_dir = run_dir / "predictions"
    evaluation_path = run_dir / "evaluation.json"

    run_baseline_inference(
        checkpoint_path=checkpoint,
        manifest=run_manifest,
        audio_embedding_root=data_config["audio_embedding_root"],
        audio_encoder=data_config["audio_encoder"],
        text_embedding_root=label_setup["text_embedding_root"],
        text_encoder=data_config["text_encoder"],
        audio_embedding_key=data_config.get("audio_embedding_key", "beat_sync"),
        namespace=data_config.get("namespace", "segment_open"),
        prediction_namespace=data_config.get("prediction_namespace", "segment_open"),
        output_dir=predictions_dir,
        device=device,
        limit=None,
        candidate_label_strategy=data_config.get("candidate_label_strategy", "track_labels"),
        annotation_processing=label_setup["annotation_processing"],
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
        reference_annotation_processing=label_setup["annotation_processing"],
        audio_embedding_root=data_config["audio_embedding_root"],
        audio_encoder=data_config["audio_encoder"],
        text_embedding_root=label_setup["text_embedding_root"],
        text_encoder=data_config["text_encoder"],
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
    if not rows:
        raise ValueError("No rows were produced")
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = _result_columns()
    csv_path = output_dir / "results.csv"
    md_path = output_dir / "results.md"
    json_path = output_dir / "results.json"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _format_cell(row.get(column)) for column in columns})
    md_path.write_text(_markdown_table(rows, columns), encoding="utf-8")
    save_json(json_path, {"rows": rows})


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


if __name__ == "__main__":
    main()
