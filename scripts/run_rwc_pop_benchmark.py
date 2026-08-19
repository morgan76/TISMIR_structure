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


DEFAULT_CONFIGS = [
    "configs/benchmarks/rwc_pop_mert_projection_ce.yaml",
    "configs/benchmarks/rwc_pop_mert_adapter_rope_ce.yaml",
    "configs/benchmarks/rwc_pop_mert_hybrid_ce.yaml",
    "configs/benchmarks/rwc_pop_mert_hybrid_ce_link.yaml",
    "configs/benchmarks/rwc_pop_mert_hybrid_ce_boundary.yaml",
    "configs/benchmarks/rwc_pop_mert_hybrid_ce_link_boundary.yaml",
]

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
            "Train/evaluate a compact RWC-Pop benchmark suite on existing MERT "
            "beat-synchronous features and write a metrics table."
        )
    )
    parser.add_argument("--config", action="append", default=[], help="Training config; can be repeated.")
    parser.add_argument("--output-dir", default="outputs/benchmarks/rwc_pop_mert")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-existing-training", action="store_true")
    parser.add_argument(
        "--eval-manifest",
        default=None,
        help="Eval manifest. Defaults to each config's validation manifest.",
    )
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument(
        "--test-manifest",
        default="data/manifests/rwc_pop_benchmark_test.local.jsonl",
        help="Held-out test manifest to evaluate after eval.",
    )
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--no-test", action="store_true", help="Only report eval metrics.")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--use-config-output-dir",
        action="store_true",
        help="Write checkpoints to each config's output_dir instead of output-dir/train/{experiment}.",
    )
    parser.add_argument(
        "--decoder",
        action="append",
        choices=[
            "argmax",
            "viterbi",
            "boundary_viterbi",
            "boundary_peak_mean_logits",
            "boundary_peak_majority_vote",
        ],
        default=[],
        help="Decoder variant to evaluate; defaults to all compatible variants.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = [Path(path) for path in (args.config or DEFAULT_CONFIGS)]
    eval_manifest = Path(args.eval_manifest) if args.eval_manifest else None
    test_manifest = None if args.no_test or not args.test_manifest else Path(args.test_manifest)
    decoder_names = args.decoder or [
        "argmax",
        "viterbi",
        "boundary_viterbi",
        "boundary_peak_mean_logits",
        "boundary_peak_majority_vote",
    ]

    rows = []
    for config_path in configs:
        config = load_yaml(config_path)
        experiment = config_path.stem.removeprefix("rwc_pop_mert_")
        config["device"] = args.device
        if not args.use_config_output_dir:
            config["output_dir"] = str(output_dir / "train" / experiment)
        if args.max_epochs is not None:
            config.setdefault("optimization", {})["max_epochs"] = int(args.max_epochs)
        if args.no_progress:
            config.setdefault("optimization", {})["progress"] = False

        print(f"\n=== {experiment}: training ===")
        train_metrics, train_seconds = _train_or_load(
            config=config,
            skip_training=args.skip_training,
            skip_existing=args.skip_existing_training,
        )
        checkpoint = _checkpoint_from_metrics_or_config(train_metrics, config)
        print(f"{experiment}: using checkpoint {checkpoint}")

        for decoder_name in decoder_names:
            if _requires_boundary_head(decoder_name) and not _boundary_head_enabled(config):
                continue
            print(f"\n=== {experiment}: {decoder_name} inference/eval ===")
            row = _run_decoder_eval(
                config=config,
                experiment=experiment,
                decoder_name=decoder_name,
                checkpoint=checkpoint,
                output_dir=output_dir,
                device=args.device,
                eval_manifest=eval_manifest,
                eval_limit=args.eval_limit,
                test_manifest=test_manifest,
                test_limit=args.test_limit,
            )
            row.update(_training_columns(train_metrics, train_seconds))
            rows.append(row)

    _write_results(output_dir, rows)
    print(f"\nSaved benchmark table to {output_dir / 'results.md'}")


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


def _run_decoder_eval(
    config: dict[str, Any],
    experiment: str,
    decoder_name: str,
    checkpoint: Path,
    output_dir: Path,
    device: str,
    eval_manifest: Path | None,
    eval_limit: int | None,
    test_manifest: Path | None,
    test_limit: int | None,
) -> dict[str, Any]:
    data_config = dict(config["data"])
    validation_config = dict(config.get("validation", {}))
    segmentation = dict(validation_config.get("segmentation", {}))
    eval_manifest = eval_manifest or Path(validation_config.get("manifest", data_config["manifest"]))
    decoder_config = _decoder_config(decoder_name, segmentation)
    split_specs: list[tuple[str, Path, int | None]] = [("eval", eval_manifest, eval_limit)]
    if test_manifest is not None:
        split_specs.append(("test", test_manifest, test_limit))

    row: dict[str, Any] = {
        "experiment": experiment,
        "decoder": decoder_name,
        "checkpoint": str(checkpoint),
    }
    for split, manifest, limit in split_specs:
        split_row = _run_single_split_eval(
            config=config,
            data_config=data_config,
            segmentation=segmentation,
            experiment=experiment,
            decoder_name=decoder_name,
            checkpoint=checkpoint,
            output_dir=output_dir,
            device=device,
            manifest=manifest,
            limit=limit,
            decoder_config=decoder_config,
            split=split,
        )
        for key, value in split_row.items():
            row[f"{split}_{key}"] = value
    return row


def _run_single_split_eval(
    config: dict[str, Any],
    data_config: dict[str, Any],
    segmentation: dict[str, Any],
    experiment: str,
    decoder_name: str,
    checkpoint: Path,
    output_dir: Path,
    device: str,
    manifest: Path,
    limit: int | None,
    decoder_config: dict[str, Any],
    split: str,
) -> dict[str, Any]:
    if not manifest.exists():
        raise FileNotFoundError(f"Missing {split} manifest: {manifest}")
    run_manifest = _limited_manifest(manifest, output_dir / "manifests", limit)
    run_dir = output_dir / experiment / decoder_name / split
    predictions_dir = run_dir / "predictions"
    evaluation_path = run_dir / "evaluation.json"

    run_baseline_inference(
        checkpoint_path=checkpoint,
        manifest=run_manifest,
        audio_embedding_root=data_config["audio_embedding_root"],
        audio_encoder=data_config["audio_encoder"],
        text_embedding_root=data_config["text_embedding_root"],
        text_encoder=data_config["text_encoder"],
        audio_embedding_key=data_config.get("audio_embedding_key", "beat_sync"),
        namespace=data_config.get("namespace", "segment_open"),
        prediction_namespace=data_config.get("prediction_namespace", "segment_open"),
        output_dir=predictions_dir,
        device=device,
        limit=None,
        candidate_label_strategy=data_config.get("candidate_label_strategy", "track_labels"),
        annotation_processing=data_config.get("annotation_processing"),
        beat_subsampling=data_config.get("beat_subsampling"),
        track_filter=data_config.get("track_filter"),
        smoothing_window=decoder_config["smoothing_window"],
        smoothing_mode=decoder_config["smoothing_mode"],
        decoder=decoder_config["decoder"],
        transition_penalty=decoder_config["transition_penalty"],
        min_segment_duration=decoder_config["min_segment_duration"],
        boundary_decoding=decoder_config["boundary_decoding"],
        boundary_peak=decoder_config["boundary_peak"],
    )

    evaluation = evaluate_prediction_manifest(
        reference_manifest=run_manifest,
        predictions_root=predictions_dir,
        namespace=data_config.get("namespace", "segment_open"),
        prediction_namespace=data_config.get("prediction_namespace", "segment_open"),
        trim=bool(segmentation.get("trim", True)),
        reference_annotation_processing=data_config.get("annotation_processing"),
        audio_embedding_root=data_config["audio_embedding_root"],
        audio_encoder=data_config["audio_encoder"],
        text_embedding_root=data_config["text_embedding_root"],
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


def _decoder_config(decoder_name: str, segmentation: dict[str, Any]) -> dict[str, Any]:
    base = {
        "smoothing_window": int(segmentation.get("smoothing_window", 7)),
        "smoothing_mode": str(segmentation.get("smoothing_mode", "mean")),
        "transition_penalty": float(segmentation.get("transition_penalty", 4.0)),
        "min_segment_duration": float(segmentation.get("min_segment_duration", 3.0)),
        "boundary_decoding": {"enabled": False},
        "boundary_peak": None,
    }
    if decoder_name == "argmax":
        return {**base, "decoder": "argmax", "transition_penalty": 0.0}
    if decoder_name == "viterbi":
        return {**base, "decoder": "viterbi"}
    if decoder_name == "boundary_viterbi":
        boundary = segmentation.get("boundary_decoding")
        if not isinstance(boundary, dict):
            boundary = {"enabled": True, "weight": 3.0, "eps": 0.0001}
        return {**base, "decoder": "viterbi", "boundary_decoding": {**boundary, "enabled": True}}
    if decoder_name == "boundary_peak_mean_logits":
        return {
            **base,
            "decoder": "boundary_peak",
            "boundary_peak": {
                "threshold": 0.5,
                "min_distance_beats": 4,
                "min_segment_duration": base["min_segment_duration"],
                "label_assignment": "mean_logits",
                "merge_same_label": True,
            },
        }
    if decoder_name == "boundary_peak_majority_vote":
        return {
            **base,
            "decoder": "boundary_peak",
            "boundary_peak": {
                "threshold": 0.5,
                "min_distance_beats": 4,
                "min_segment_duration": base["min_segment_duration"],
                "label_assignment": "majority_vote",
                "merge_same_label": True,
            },
        }
    raise ValueError(f"Unknown decoder variant: {decoder_name}")


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
        "best_segmentation_metric": metrics.get("best_segmentation_metric"),
        "best_segmentation_score": metrics.get("best_segmentation_score"),
        "best_segmentation_epoch": metrics.get("best_segmentation_epoch"),
        "epochs_trained": metrics.get("epochs_trained"),
        "train_seconds": train_seconds,
    }


def _requires_boundary_head(decoder_name: str) -> bool:
    return decoder_name in {
        "boundary_viterbi",
        "boundary_peak_mean_logits",
        "boundary_peak_majority_vote",
    }


def _boundary_head_enabled(config: dict[str, Any]) -> bool:
    update_blocks = config.get("model", {}).get("update_blocks", {})
    boundary_head = update_blocks.get("boundary_head", False)
    if isinstance(boundary_head, dict):
        return bool(boundary_head.get("enabled", True))
    return bool(boundary_head)


def _write_results(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No benchmark rows were produced")
    csv_path = output_dir / "results.csv"
    md_path = output_dir / "results.md"
    json_path = output_dir / "results.json"
    columns = _result_columns()
    output_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _format_cell(row.get(column)) for column in columns})
    md_path.write_text(_markdown_table(rows, columns), encoding="utf-8")
    save_json(json_path, {"rows": rows})


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    headers = [_pretty_column(column) for column in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row.get(column)) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def _result_columns() -> list[str]:
    columns = ["experiment", "decoder"]
    for split in ("eval", "test"):
        columns.append(f"{split}_num_tracks")
        columns.extend(f"{split}_{metric}" for metric in METRICS)
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
            "test_evaluation_json",
        ]
    )
    return columns


def _pretty_column(column: str) -> str:
    exact = {
        "experiment": "Experiment",
        "decoder": "Decoder",
        "best_val_loss": "Best Val Loss",
        "best_epoch": "Best Epoch",
        "best_segmentation_score": "Best Seg",
        "best_segmentation_epoch": "Best Seg Epoch",
        "epochs_trained": "Epochs",
        "train_seconds": "Train s",
        "checkpoint": "Checkpoint",
        "eval_evaluation_json": "Eval JSON",
        "test_evaluation_json": "Test JSON",
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
    for split in ("eval", "test"):
        prefix = f"{split}_"
        if column.startswith(prefix):
            metric = column.removeprefix(prefix)
            return f"{split.title()} {metric_names.get(metric, metric)}"
    return column


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


if __name__ == "__main__":
    main()
