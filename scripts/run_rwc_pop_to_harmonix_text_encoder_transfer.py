#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tismir.data.jams import load_processed_structure_sections, unique_labels
from tismir.data.manifest import load_manifest, save_manifest
from tismir.evaluation import evaluate_prediction_manifest, save_evaluation
from tismir.inference import run_baseline_inference
from tismir.io import save_json
from tismir.training.data import StructureEmbeddingDataset


SOURCE_TRAIN_MANIFEST = "data/manifests/rwc_pop_benchmark_train.local.jsonl"
TARGET_EVAL_MANIFEST = "data/manifests/harmonix_val.local.jsonl"
ANNOTATION_PROCESSING = {"policy": "base_labels"}

SOURCE_AUDIO_ROOT = "data/embeddings/audio"
TARGET_AUDIO_ROOT = "data/embeddings/audio_madmom"
AUDIO_ENCODER = "mert"
AUDIO_EMBEDDING_KEY = "beat_sync"
NAMESPACE = "segment_open"
CANDIDATE_LABEL_STRATEGY = "track_labels"
IGNORE_INDEX = -100

BEAT_SUBSAMPLING = {
    "enabled": True,
    "bpm_threshold": 140.0,
    "factor": 2,
    "pooling": "mean",
    "target_assignment": "max_overlap",
}

SEGMENTATION = {
    "smoothing_window": 7,
    "smoothing_mode": "mean",
    "decoder": "viterbi",
    "transition_penalty": 4.0,
    "min_segment_duration": 3.0,
    "trim": True,
}

BARE_SETUPS: dict[str, dict[str, Any]] = {
    "e5_base_v2": {
        "checkpoint": (
            "outputs/benchmarks/rwc_pop_text_encoder_ablation/train/e5_base_v2/"
            "best_segmentation_checkpoint.pt"
        ),
        "text_embedding_root": "data/embeddings/text_harmonix_base_labels",
        "text_encoder": "sentence_transformers",
    },
    "mpnet_base_v2": {
        "checkpoint": (
            "outputs/benchmarks/rwc_pop_text_encoder_ablation/train/mpnet_base_v2/"
            "best_segmentation_checkpoint.pt"
        ),
        "text_embedding_root": "data/embeddings/text_harmonix_base_labels_mpnet",
        "text_encoder": "sentence_transformers",
    },
    "clap_music_speech": {
        "checkpoint": (
            "outputs/benchmarks/rwc_pop_text_encoder_ablation/train/clap_music_speech/"
            "best_segmentation_checkpoint.pt"
        ),
        "text_embedding_root": "data/embeddings/text_harmonix_base_labels_clap_music_speech",
        "text_encoder": "clap",
    },
    "muq_mulan_large": {
        "checkpoint": (
            "outputs/benchmarks/rwc_pop_text_encoder_ablation/train/muq_mulan_large/"
            "best_segmentation_checkpoint.pt"
        ),
        "text_embedding_root": "data/embeddings/text_harmonix_base_labels_muq_mulan",
        "text_encoder": "muq_mulan",
    },
}

COMPACT_DEFINITION_SETUPS: dict[str, dict[str, Any]] = {
    "e5_base_v2": {
        "checkpoint": (
            "outputs/benchmarks/rwc_pop_compact_definition_text_encoder_ablation/train/e5_base_v2/"
            "best_segmentation_checkpoint.pt"
        ),
        "text_embedding_root": "data/embeddings/text_harmonix_base_compact_definition",
        "text_encoder": "sentence_transformers",
    },
    "mpnet_base_v2": {
        "checkpoint": (
            "outputs/benchmarks/rwc_pop_compact_definition_text_encoder_ablation/train/mpnet_base_v2/"
            "best_segmentation_checkpoint.pt"
        ),
        "text_embedding_root": "data/embeddings/text_harmonix_base_compact_definition_mpnet",
        "text_encoder": "sentence_transformers",
    },
    "clap_music_speech": {
        "checkpoint": (
            "outputs/benchmarks/rwc_pop_compact_definition_text_encoder_ablation/train/clap_music_speech/"
            "best_segmentation_checkpoint.pt"
        ),
        "text_embedding_root": "data/embeddings/text_harmonix_base_compact_definition_clap_music_speech",
        "text_encoder": "clap",
    },
    "muq_mulan_large": {
        "checkpoint": (
            "outputs/benchmarks/rwc_pop_compact_definition_text_encoder_ablation/train/muq_mulan_large/"
            "best_segmentation_checkpoint.pt"
        ),
        "text_embedding_root": "data/embeddings/text_harmonix_base_compact_definition_muq_mulan",
        "text_encoder": "muq_mulan",
    },
}

SETUP_GROUPS: dict[str, dict[str, dict[str, Any]]] = {
    "bare": BARE_SETUPS,
    "compact_definition": COMPACT_DEFINITION_SETUPS,
}

METRICS = (
    "F-measure@0.5",
    "F-measure@3.0",
    "Acc",
    "Balanced Acc",
    "Pairwise F-measure",
    "NCE F-measure",
)

ZERO_SHOT_METRICS = (
    "seen_frame_count",
    "seen_frame_acc",
    "unseen_frame_count",
    "unseen_frame_acc",
    "unseen_macro_label_acc",
    "unseen_pred_count",
    "unseen_pred_precision",
    "unseen_target_rate",
    "unseen_pred_rate",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate RWC-Pop-trained text-conditioned CE checkpoints on Harmonix "
            "and report metrics for labels unseen in the RWC-Pop training labels."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/benchmarks/rwc_pop_to_harmonix_text_encoder_transfer",
    )
    parser.add_argument(
        "--label-text",
        choices=sorted(SETUP_GROUPS),
        default="bare",
        help="Which label text embedding/checkpoint family to evaluate.",
    )
    parser.add_argument(
        "--setups",
        nargs="+",
        choices=sorted(BARE_SETUPS),
        default=None,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-limit", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setups = SETUP_GROUPS[args.label_text]
    selected_setups = list(setups) if args.setups is None else list(args.setups)
    eval_manifest = _limited_manifest(
        Path(TARGET_EVAL_MANIFEST),
        output_dir / "manifests",
        args.eval_limit,
    )

    source_labels = _manifest_label_set(Path(SOURCE_TRAIN_MANIFEST))
    target_labels = _manifest_label_set(eval_manifest)
    label_overlap = _label_overlap(source_labels=source_labels, target_labels=target_labels)
    _write_label_overlap(output_dir, label_overlap)

    rows = []
    unseen_label_rows = []
    for setup_name in selected_setups:
        print(f"\n=== {setup_name}: RWC-Pop -> Harmonix ===")
        setup = setups[setup_name]
        row, per_label_rows = _run_setup(
            setup_name=setup_name,
            setup=setup,
            output_dir=output_dir,
            eval_manifest=eval_manifest,
            source_labels=source_labels,
            device=args.device,
        )
        rows.append(row)
        unseen_label_rows.extend(per_label_rows)
        _write_results(output_dir, rows)
        _write_unseen_label_results(output_dir, unseen_label_rows)

    print(f"\nSaved transfer table to {output_dir / 'results.md'}")


def _run_setup(
    setup_name: str,
    setup: dict[str, Any],
    output_dir: Path,
    eval_manifest: Path,
    source_labels: set[str],
    device: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    checkpoint = Path(setup["checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing RWC-Pop checkpoint for {setup_name}: {checkpoint}")
    run_dir = output_dir / setup_name / "harmonix_eval"
    predictions_dir = run_dir / "predictions"
    evaluation_path = run_dir / "evaluation.json"

    run_baseline_inference(
        checkpoint_path=checkpoint,
        manifest=eval_manifest,
        audio_embedding_root=TARGET_AUDIO_ROOT,
        audio_encoder=AUDIO_ENCODER,
        text_embedding_root=setup["text_embedding_root"],
        text_encoder=setup["text_encoder"],
        audio_embedding_key=AUDIO_EMBEDDING_KEY,
        namespace=NAMESPACE,
        prediction_namespace=NAMESPACE,
        output_dir=predictions_dir,
        device=device,
        limit=None,
        candidate_label_strategy=CANDIDATE_LABEL_STRATEGY,
        annotation_processing=ANNOTATION_PROCESSING,
        beat_subsampling=BEAT_SUBSAMPLING,
        smoothing_window=int(SEGMENTATION["smoothing_window"]),
        smoothing_mode=str(SEGMENTATION["smoothing_mode"]),
        decoder=str(SEGMENTATION["decoder"]),
        transition_penalty=float(SEGMENTATION["transition_penalty"]),
        min_segment_duration=float(SEGMENTATION["min_segment_duration"]),
        boundary_decoding=False,
        boundary_peak=None,
        use_checkpoint_annotation_processing=False,
    )

    evaluation = evaluate_prediction_manifest(
        reference_manifest=eval_manifest,
        predictions_root=predictions_dir,
        namespace=NAMESPACE,
        prediction_namespace=NAMESPACE,
        trim=bool(SEGMENTATION["trim"]),
        reference_annotation_processing=ANNOTATION_PROCESSING,
        audio_embedding_root=TARGET_AUDIO_ROOT,
        audio_encoder=AUDIO_ENCODER,
        text_embedding_root=setup["text_embedding_root"],
        text_encoder=setup["text_encoder"],
        audio_embedding_key=AUDIO_EMBEDDING_KEY,
        candidate_label_strategy=CANDIDATE_LABEL_STRATEGY,
        beat_subsampling=BEAT_SUBSAMPLING,
        ignore_index=IGNORE_INDEX,
    )
    save_evaluation(evaluation_path, evaluation)

    zero_shot, per_label_rows = _zero_shot_frame_metrics(
        setup_name=setup_name,
        eval_manifest=eval_manifest,
        predictions_root=predictions_dir,
        text_embedding_root=setup["text_embedding_root"],
        text_encoder=setup["text_encoder"],
        source_labels=source_labels,
    )

    row: dict[str, Any] = {
        "condition": setup_name,
        "source_train": "rwc_pop",
        "target_eval": "harmonix",
        "checkpoint": str(checkpoint),
        "eval_num_tracks": evaluation["num_tracks"],
        "evaluation_json": str(evaluation_path),
        "predictions_dir": str(predictions_dir),
    }
    for metric in METRICS:
        values = evaluation["summary"].get(metric, {})
        row[metric] = values.get("mean")
        row[f"{metric} std"] = values.get("std")
    row.update(zero_shot)
    return row, per_label_rows


def _zero_shot_frame_metrics(
    setup_name: str,
    eval_manifest: Path,
    predictions_root: Path,
    text_embedding_root: str | Path,
    text_encoder: str,
    source_labels: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset = StructureEmbeddingDataset(
        manifest=eval_manifest,
        audio_embedding_root=TARGET_AUDIO_ROOT,
        audio_encoder=AUDIO_ENCODER,
        text_embedding_root=text_embedding_root,
        text_encoder=text_encoder,
        audio_embedding_key=AUDIO_EMBEDDING_KEY,
        namespace=NAMESPACE,
        candidate_label_strategy=CANDIDATE_LABEL_STRATEGY,
        annotation_processing=ANNOTATION_PROCESSING,
        beat_subsampling=BEAT_SUBSAMPLING,
        ignore_index=IGNORE_INDEX,
    )

    seen_correct = seen_count = 0
    unseen_correct = unseen_count = 0
    unseen_pred_correct = unseen_pred_count = 0
    valid_count = 0
    per_label_correct: Counter[str] = Counter()
    per_label_count: Counter[str] = Counter()

    for example in dataset:
        prediction_json = predictions_root / example.dataset / f"{example.track_id}.json"
        with prediction_json.open("r", encoding="utf-8") as handle:
            prediction = json.load(handle)
        predicted = _prediction_indices_for_labels(prediction, labels=example.labels)
        if len(predicted) != len(example.targets):
            raise ValueError(
                f"Frame prediction/target length mismatch for {example.dataset}/{example.track_id}"
            )

        labels = np.asarray(example.labels, dtype=object)
        targets = example.targets.astype(np.int64, copy=False)
        valid = targets != IGNORE_INDEX
        valid_count += int(valid.sum())
        if not np.any(valid):
            continue

        target_labels = labels[targets[valid]]
        predicted_valid = predicted[valid]
        predicted_labels = np.asarray(
            [
                example.labels[index] if 0 <= int(index) < len(example.labels) else ""
                for index in predicted_valid
            ],
            dtype=object,
        )
        target_unseen = np.asarray([label not in source_labels for label in target_labels], dtype=bool)
        pred_unseen = np.asarray([label not in source_labels for label in predicted_labels], dtype=bool)
        correct = predicted_valid == targets[valid]

        seen_mask = ~target_unseen
        seen_count += int(seen_mask.sum())
        seen_correct += int(np.sum(correct & seen_mask))
        unseen_count += int(target_unseen.sum())
        unseen_correct += int(np.sum(correct & target_unseen))
        unseen_pred_count += int(pred_unseen.sum())
        unseen_pred_correct += int(np.sum(correct & pred_unseen & target_unseen))
        for label, is_correct in zip(target_labels[target_unseen], correct[target_unseen]):
            per_label_count[str(label)] += 1
            per_label_correct[str(label)] += int(bool(is_correct))

    per_label_rows = [
        {
            "condition": setup_name,
            "label": label,
            "frame_count": count,
            "frame_acc": per_label_correct[label] / count if count else float("nan"),
        }
        for label, count in sorted(per_label_count.items(), key=lambda item: (-item[1], item[0]))
    ]
    per_label_acc = [
        row["frame_acc"]
        for row in per_label_rows
        if row["frame_count"] > 0 and np.isfinite(float(row["frame_acc"]))
    ]
    metrics = {
        "seen_frame_count": seen_count,
        "seen_frame_acc": seen_correct / seen_count if seen_count else float("nan"),
        "unseen_frame_count": unseen_count,
        "unseen_frame_acc": unseen_correct / unseen_count if unseen_count else float("nan"),
        "unseen_macro_label_acc": float(np.mean(per_label_acc)) if per_label_acc else float("nan"),
        "unseen_pred_count": unseen_pred_count,
        "unseen_pred_precision": (
            unseen_pred_correct / unseen_pred_count if unseen_pred_count else float("nan")
        ),
        "unseen_target_rate": unseen_count / valid_count if valid_count else float("nan"),
        "unseen_pred_rate": unseen_pred_count / valid_count if valid_count else float("nan"),
    }
    return metrics, per_label_rows


def _prediction_indices_for_labels(prediction: dict[str, Any], labels: list[str]) -> np.ndarray:
    raw_indices = np.asarray(prediction["frame_label_indices"], dtype=np.int64)
    prediction_labels = list(prediction.get("labels", []))
    if prediction_labels == labels:
        return raw_indices

    label_to_index = {label: index for index, label in enumerate(labels)}
    mapped = np.full(raw_indices.shape, IGNORE_INDEX, dtype=np.int64)
    for output_index, label in enumerate(prediction_labels):
        if label in label_to_index:
            mapped[raw_indices == output_index] = label_to_index[label]
    return mapped


def _manifest_label_set(manifest: Path) -> set[str]:
    labels: set[str] = set()
    for track in load_manifest(manifest):
        sections = load_processed_structure_sections(
            track.jams_path,
            namespace=NAMESPACE,
            annotation_processing=ANNOTATION_PROCESSING,
        )
        labels.update(unique_labels(sections))
    return labels


def _label_overlap(source_labels: set[str], target_labels: set[str]) -> dict[str, Any]:
    return {
        "source_train_manifest": SOURCE_TRAIN_MANIFEST,
        "target_eval_manifest": TARGET_EVAL_MANIFEST,
        "annotation_processing": ANNOTATION_PROCESSING,
        "source_label_count": len(source_labels),
        "target_label_count": len(target_labels),
        "shared_label_count": len(source_labels & target_labels),
        "target_unseen_label_count": len(target_labels - source_labels),
        "source_only_label_count": len(source_labels - target_labels),
        "source_labels": sorted(source_labels),
        "target_labels": sorted(target_labels),
        "shared_labels": sorted(source_labels & target_labels),
        "target_unseen_labels": sorted(target_labels - source_labels),
        "source_only_labels": sorted(source_labels - target_labels),
    }


def _write_label_overlap(output_dir: Path, overlap: dict[str, Any]) -> None:
    save_json(output_dir / "label_overlap.json", overlap)
    lines = [
        "# Label Overlap",
        "",
        f"- Source train labels: {overlap['source_label_count']}",
        f"- Target eval labels: {overlap['target_label_count']}",
        f"- Shared labels: {overlap['shared_label_count']}",
        f"- Target labels unseen in source train: {overlap['target_unseen_label_count']}",
        "",
        "## Target Unseen Labels",
        "",
        ", ".join(overlap["target_unseen_labels"]) or "None",
        "",
        "## Shared Labels",
        "",
        ", ".join(overlap["shared_labels"]) or "None",
        "",
    ]
    (output_dir / "label_overlap.md").write_text("\n".join(lines), encoding="utf-8")


def _limited_manifest(manifest: Path, output_dir: Path, limit: int | None) -> Path:
    if limit is None:
        return manifest
    output_dir.mkdir(parents=True, exist_ok=True)
    tracks = load_manifest(manifest)[:limit]
    limited = output_dir / f"{manifest.stem}.first_{limit}.jsonl"
    save_manifest(limited, tracks)
    return limited


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


def _write_unseen_label_results(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    columns = ["condition", "label", "frame_count", "frame_acc"]
    with (output_dir / "unseen_label_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _format_cell(row.get(column)) for column in columns})
    (output_dir / "unseen_label_metrics.md").write_text(
        _markdown_table(rows, columns),
        encoding="utf-8",
    )


def _result_columns() -> list[str]:
    columns = [
        "condition",
        "source_train",
        "target_eval",
        "eval_num_tracks",
        *METRICS,
        *ZERO_SHOT_METRICS,
        "checkpoint",
        "evaluation_json",
        "predictions_dir",
    ]
    return list(columns)


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
        "source_train": "Source Train",
        "target_eval": "Target Eval",
        "eval_num_tracks": "Tracks",
        "F-measure@0.5": "F@0.5",
        "F-measure@3.0": "F@3.0",
        "Balanced Acc": "Bal Acc",
        "Pairwise F-measure": "PFC",
        "NCE F-measure": "NCE",
        "seen_frame_count": "Seen Frames",
        "seen_frame_acc": "Seen Acc",
        "unseen_frame_count": "Unseen Frames",
        "unseen_frame_acc": "Unseen Acc",
        "unseen_macro_label_acc": "Unseen Macro Acc",
        "unseen_pred_count": "Unseen Pred Frames",
        "unseen_pred_precision": "Unseen Pred Precision",
        "unseen_target_rate": "Unseen Target Rate",
        "unseen_pred_rate": "Unseen Pred Rate",
        "frame_count": "Frames",
        "frame_acc": "Acc",
        "checkpoint": "Checkpoint",
        "evaluation_json": "Eval JSON",
        "predictions_dir": "Predictions",
    }
    return exact.get(column, column)


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        return f"{value:.6f}"
    return str(value)


def _plot_results(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    plt = _matplotlib()
    conditions = [str(row["condition"]) for row in rows]
    metrics = ("F-measure@0.5", "F-measure@3.0", "Acc", "Pairwise F-measure", "NCE F-measure")
    labels = {
        "F-measure@0.5": "F@0.5",
        "F-measure@3.0": "F@3.0",
        "Acc": "Acc",
        "Pairwise F-measure": "PFC",
        "NCE F-measure": "NCE",
    }

    x = np.arange(len(conditions), dtype=float)
    width = 0.15
    offsets = (np.arange(len(metrics), dtype=float) - (len(metrics) - 1) / 2.0) * width
    fig, ax = plt.subplots(figsize=(12.5, 5.4), constrained_layout=True)
    for offset, metric in zip(offsets, metrics, strict=True):
        values = [_float_or_nan(row.get(metric)) for row in rows]
        ax.bar(x + offset, values, width=width, label=labels[metric])
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Mean score")
    ax.set_title("RWC-Pop -> Harmonix transfer metrics")
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.legend(ncols=3, frameon=False)
    fig.savefig(output_dir / "transfer_metrics_barplot.png", dpi=180)
    plt.close(fig)

    zero_metrics = ("seen_frame_acc", "unseen_frame_acc", "unseen_pred_precision")
    zero_labels = {
        "seen_frame_acc": "Seen Acc",
        "unseen_frame_acc": "Unseen Acc",
        "unseen_pred_precision": "Unseen Pred Precision",
    }
    x = np.arange(len(conditions), dtype=float)
    width = 0.22
    offsets = (np.arange(len(zero_metrics), dtype=float) - 1.0) * width
    fig, ax = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    for offset, metric in zip(offsets, zero_metrics, strict=True):
        values = [_float_or_nan(row.get(metric)) for row in rows]
        ax.bar(x + offset, values, width=width, label=zero_labels[metric])
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Frame-level score")
    ax.set_title("Seen vs unseen Harmonix labels after RWC-Pop training")
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.legend(frameon=False)
    fig.savefig(output_dir / "seen_unseen_frame_metrics.png", dpi=180)
    plt.close(fig)


def _float_or_nan(value: Any) -> float:
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
