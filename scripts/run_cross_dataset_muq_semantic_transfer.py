#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from tismir.data.annotations import build_corpus_section_id_mapping
from tismir.data.jams import load_processed_structure_sections, unique_labels
from tismir.data.manifest import load_manifest, save_manifest
from tismir.evaluation import evaluate_prediction_manifest, save_evaluation
from tismir.inference import run_baseline_inference
from tismir.io import load_yaml, save_json
from tismir.preprocessing.label_normalization import normalize_label
from tismir.preprocessing.text import preprocess_dataset_text
from tismir.training import train_projection_baseline
from tismir.training.data import StructureEmbeddingDataset


TEXT_ENCODER = "muq_mulan"
TEXT_ENCODER_CONFIG = {
    "checkpoint": "OpenMuQ/MuQ-MuLan-large",
    "device": None,
    "normalize_embeddings": True,
    "batch_size": 16,
}
NAMESPACE = "segment_open"
ANNOTATION_PROCESSING_REAL = {"policy": "base_labels"}
IGNORE_INDEX = -100
PRESERVE_LABELS = ("silence",)

JOINT_MAPPING_PATH = (
    "configs/annotation_mappings/"
    "rwc_pop_harmonix_section_ids_joint_base_labels_seed0.json"
)
JOINT_TEXT_ROOT = "data/embeddings/text_rwc_pop_harmonix_base_section_ids_joint_muq_mulan"
CANONICAL_LABEL_MAPPING_PATH = (
    "configs/annotation_mappings/"
    "rwc_pop_harmonix_canonical_base_label_mapping.json"
)
CANONICAL_JOINT_MAPPING_PATH = (
    "configs/annotation_mappings/"
    "rwc_pop_harmonix_section_ids_joint_canonical_base_labels_seed0.json"
)
CANONICAL_TEXT_ROOT = "data/embeddings/text_rwc_pop_harmonix_base_labels_canonical_muq_mulan"
CANONICAL_JOINT_TEXT_ROOT = (
    "data/embeddings/text_rwc_pop_harmonix_base_section_ids_joint_canonical_muq_mulan"
)

DATASETS: dict[str, dict[str, Any]] = {
    "rwc_pop": {
        "name": "RWC-Pop",
        "config": "configs/benchmarks/rwc_pop_mert_text_conditioned_ce.yaml",
        "train_manifest": "data/manifests/rwc_pop_benchmark_train.local.jsonl",
        "val_manifest": "data/manifests/rwc_pop_benchmark_val.local.jsonl",
        "test_manifest": "data/manifests/rwc_pop_benchmark_test.local.jsonl",
        "real_text_root": "data/embeddings/text_rwc_pop_base_labels_muq_mulan",
    },
    "harmonix": {
        "name": "Harmonix",
        "config": "configs/benchmarks/harmonix_mert_text_conditioned_ce.yaml",
        "train_manifest": "data/manifests/harmonix_train.local.jsonl",
        "val_manifest": "data/manifests/harmonix_val.local.jsonl",
        "test_manifest": None,
        "real_text_root": "data/embeddings/text_harmonix_base_labels_muq_mulan",
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

ZERO_SHOT_METRICS = (
    "source_label_count",
    "target_label_count",
    "target_unseen_label_count",
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

DIRECTIONS = {
    "rwc_to_harmonix": ("rwc_pop", "harmonix"),
    "harmonix_to_rwc": ("harmonix", "rwc_pop"),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train the simple MuQ text-conditioned CE model on one dataset and "
            "evaluate it on the other, comparing semantic labels against a joint "
            "corpus-level anonymous section-ID mapping."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/benchmarks/cross_dataset_muq_text_conditioned_ce_semantic_transfer",
    )
    parser.add_argument(
        "--directions",
        nargs="+",
        choices=sorted(DIRECTIONS),
        default=sorted(DIRECTIONS),
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=["real", "joint_corpus_ids"],
        default=["real", "joint_corpus_ids"],
    )
    parser.add_argument(
        "--label-canonicalization",
        choices=["raw", "canonical"],
        default="raw",
        help=(
            "Use raw base labels, or first collapse cross-dataset spelling/"
            "near-synonym variants such as prechorus/pre chorus."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-existing-training", action="store_true")
    parser.add_argument("--skip-text-preprocessing", action="store_true")
    parser.add_argument("--force-joint-mapping", action="store_true")
    parser.add_argument("--force-text-preprocessing", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--source-val-limit", type=int, default=None)
    parser.add_argument("--target-eval-limit", type=int, default=None)
    parser.add_argument("--target-test-limit", type=int, default=None)
    parser.add_argument("--no-target-test", action="store_true")
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

    label_context = _label_context(args.label_canonicalization)
    if label_context["canonical_label_mapping_path"] is not None:
        _load_or_build_canonical_label_mapping(
            mapping_path=Path(label_context["canonical_label_mapping_path"]),
            force=args.force_joint_mapping,
        )
    mapping_path = Path(label_context["joint_mapping_path"])
    mapping = _load_or_build_joint_mapping(
        mapping_path=mapping_path,
        force=args.force_joint_mapping,
        source_annotation_processing=label_context["real_annotation_processing"],
    )
    if not args.skip_text_preprocessing:
        _ensure_text_embeddings(
            label_context=label_context,
            force=args.force_text_preprocessing,
        )

    if args.prepare_only:
        print(f"Prepared joint mapping/text embeddings at {mapping_path}")
        return

    rows: list[dict[str, Any]] = []
    unseen_label_rows: list[dict[str, Any]] = []
    training_cache: dict[tuple[str, str], tuple[Path, dict[str, Any], float | None, dict[str, Any]]] = {}

    for direction_name in args.directions:
        source_dataset, target_dataset = DIRECTIONS[direction_name]
        source_labels = _manifest_label_set(
            Path(DATASETS[source_dataset]["train_manifest"]),
            annotation_processing=label_context["real_annotation_processing"],
        )
        for condition in args.conditions:
            print(f"\n=== train {source_dataset} / {condition} ===")
            checkpoint, training_config, training_metrics, training_seconds = _train_or_load_condition(
                source_dataset=source_dataset,
                condition=condition,
                output_dir=output_dir,
                device=args.device,
                max_epochs=args.max_epochs,
                train_limit=args.train_limit,
                source_val_limit=args.source_val_limit,
                progress=not args.no_progress,
                num_workers=args.num_workers,
                skip_training=args.skip_training,
                skip_existing=args.skip_existing_training,
                mapping_path=mapping_path,
                label_context=label_context,
                cache=training_cache,
            )
            print(f"{source_dataset} / {condition}: using checkpoint {checkpoint}")

            label_setup = _label_setup(target_dataset, condition, mapping_path, label_context)
            real_label_setup = _label_setup(target_dataset, "real", mapping_path, label_context)
            target_config = _target_data_config(target_dataset)
            split_specs = [("eval", Path(DATASETS[target_dataset]["val_manifest"]), args.target_eval_limit)]
            target_test_manifest = DATASETS[target_dataset].get("test_manifest")
            if target_test_manifest is not None and not args.no_target_test:
                split_specs.append(("test", Path(target_test_manifest), args.target_test_limit))

            row: dict[str, Any] = {
                "direction": direction_name,
                "condition": condition,
                "source_train": source_dataset,
                "target_eval": target_dataset,
                "checkpoint": str(checkpoint),
            }
            row.update(_training_columns(training_metrics, training_seconds))
            for split, manifest, limit in split_specs:
                print(f"\n=== {source_dataset} {condition} -> {target_dataset} {split} ===")
                split_row, per_label_rows = _run_transfer_split(
                    source_dataset=source_dataset,
                    target_dataset=target_dataset,
                    condition=condition,
                    split=split,
                    checkpoint=checkpoint,
                    manifest=manifest,
                    limit=limit,
                    output_dir=output_dir,
                    device=args.device,
                    target_data_config=target_config,
                    label_setup=label_setup,
                    real_label_setup=real_label_setup,
                    source_labels=source_labels,
                    mapping=mapping,
                )
                for key, value in split_row.items():
                    row[f"{split}_{key}"] = value
                unseen_label_rows.extend(per_label_rows)

            rows.append(row)
            _write_results(output_dir, rows)
            _write_unseen_label_results(output_dir, unseen_label_rows)

    print(f"\nSaved transfer table to {output_dir / 'results.md'}")


def _label_context(label_canonicalization: str) -> dict[str, Any]:
    if label_canonicalization == "raw":
        return {
            "label_canonicalization": "raw",
            "real_annotation_processing": ANNOTATION_PROCESSING_REAL,
            "real_text_roots": {
                dataset_name: dataset["real_text_root"]
                for dataset_name, dataset in DATASETS.items()
            },
            "joint_mapping_path": JOINT_MAPPING_PATH,
            "joint_text_root": JOINT_TEXT_ROOT,
            "canonical_label_mapping_path": None,
        }
    if label_canonicalization == "canonical":
        canonical_mapping_path = Path(CANONICAL_LABEL_MAPPING_PATH)
        return {
            "label_canonicalization": "canonical",
            "real_annotation_processing": _canonical_annotation_processing(canonical_mapping_path),
            "real_text_roots": {
                dataset_name: CANONICAL_TEXT_ROOT
                for dataset_name in DATASETS
            },
            "joint_mapping_path": CANONICAL_JOINT_MAPPING_PATH,
            "joint_text_root": CANONICAL_JOINT_TEXT_ROOT,
            "canonical_label_mapping_path": CANONICAL_LABEL_MAPPING_PATH,
        }
    raise ValueError(f"Unknown label canonicalization: {label_canonicalization}")


def _canonical_annotation_processing(mapping_path: Path) -> dict[str, Any]:
    return {
        "policy": "section_ids_corpus",
        "source_annotation_processing": ANNOTATION_PROCESSING_REAL,
        "section_id_mapping_path": str(mapping_path),
        "section_id_preserve_labels": list(PRESERVE_LABELS),
    }


def _load_or_build_canonical_label_mapping(mapping_path: Path, force: bool) -> dict[str, str]:
    if mapping_path.exists() and not force:
        with mapping_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        mapping = payload.get("mapping", payload)
        if not isinstance(mapping, dict):
            raise TypeError(f"Canonical mapping file has no mapping object: {mapping_path}")
        return {str(source): str(target) for source, target in mapping.items()}

    labels: list[str] = []
    manifests: list[str] = []
    for dataset in DATASETS.values():
        for key in ("train_manifest", "val_manifest", "test_manifest"):
            manifest = dataset.get(key)
            if manifest is None:
                continue
            manifests.append(str(manifest))
            labels.extend(
                _manifest_labels_in_order(
                    Path(manifest),
                    annotation_processing=ANNOTATION_PROCESSING_REAL,
                )
            )
    labels = list(dict.fromkeys(labels))
    mapping = {label: _canonical_section_label(label) for label in labels}
    payload = {
        "mapping": mapping,
        "labels": labels,
        "namespace": NAMESPACE,
        "source_annotation_processing": ANNOTATION_PROCESSING_REAL,
        "manifests": manifests,
        "note": (
            "Raw/base labels mapped to a shared canonical vocabulary before "
            "cross-dataset semantic-transfer experiments."
        ),
    }
    save_json(mapping_path, payload)
    changed = sum(label != mapped for label, mapped in mapping.items())
    print(f"saved {len(mapping)} canonical labels to {mapping_path} ({changed} changed)")
    return mapping


def _canonical_section_label(label: str) -> str:
    text = normalize_label(
        label,
        config={"name": "harmonix", "normalize_whitespace": True},
    ).strip().lower()
    overrides = {
        "ending": "outro",
        "nothing": "silence",
        "silent": "silence",
        "no music": "silence",
        "guitar solo": "solo",
    }
    return overrides.get(text, text)


def _load_or_build_joint_mapping(
    mapping_path: Path,
    force: bool,
    source_annotation_processing: str | dict[str, Any] | None,
) -> dict[str, str]:
    if mapping_path.exists() and not force:
        with mapping_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        mapping = payload.get("mapping", payload)
        if not isinstance(mapping, dict):
            raise TypeError(f"Mapping file has no mapping object: {mapping_path}")
        return {str(source): str(target) for source, target in mapping.items()}

    labels: list[str] = []
    manifests: list[str] = []
    for dataset in DATASETS.values():
        for key in ("train_manifest", "val_manifest", "test_manifest"):
            manifest = dataset.get(key)
            if manifest is None:
                continue
            manifests.append(str(manifest))
            labels.extend(
                _manifest_labels_in_order(
                    Path(manifest),
                    annotation_processing=source_annotation_processing,
                )
            )
    labels = list(dict.fromkeys(labels))
    mapping = build_corpus_section_id_mapping(
        labels,
        prefix="section",
        seed=0,
        preserve_labels=PRESERVE_LABELS,
    )
    payload = {
        "mapping": mapping,
        "labels": labels,
        "prefix": "section",
        "seed": 0,
        "preserve_labels": list(PRESERVE_LABELS),
        "namespace": NAMESPACE,
        "source_annotation_processing": source_annotation_processing,
        "manifests": manifests,
        "note": (
            "Joint anonymous section-ID mapping over RWC-Pop and Harmonix "
            "labels. IDs are arbitrary but stable across both datasets."
        ),
    }
    save_json(mapping_path, payload)
    print(f"saved {len(mapping)} mapped labels to {mapping_path}")
    return mapping


def _ensure_text_embeddings(label_context: dict[str, Any], force: bool) -> None:
    if label_context["label_canonicalization"] == "canonical":
        _ensure_dataset_text_embeddings(
            output_root=label_context["real_text_roots"]["rwc_pop"],
            annotation_processing=label_context["real_annotation_processing"],
            force=force,
        )
    _ensure_dataset_text_embeddings(
        output_root=label_context["joint_text_root"],
        annotation_processing=_joint_corpus_annotation_processing(
            Path(label_context["joint_mapping_path"]),
            label_context=label_context,
        ),
        force=force,
    )


def _ensure_dataset_text_embeddings(
    output_root: str | Path,
    annotation_processing: str | dict[str, Any] | None,
    force: bool,
) -> None:
    missing = []
    for dataset_name in DATASETS:
        labels_path = (
            Path(output_root)
            / TEXT_ENCODER
            / dataset_name
            / "labels.json"
        )
        embeddings_path = labels_path.with_name("embeddings.npy")
        if force or not labels_path.exists() or not embeddings_path.exists():
            missing.append(dataset_name)
    if not missing:
        return

    tracks = []
    seen_tracks: set[tuple[str, str]] = set()
    for dataset in DATASETS.values():
        for key in ("train_manifest", "val_manifest", "test_manifest"):
            manifest = dataset.get(key)
            if manifest is None:
                continue
            for track in load_manifest(manifest):
                track_key = (track.dataset, track.track_id)
                if track_key in seen_tracks:
                    continue
                seen_tracks.add(track_key)
                tracks.append(track)
    preprocess_dataset_text(
        tracks=tracks,
        output_root=output_root,
        text_encoder_name=TEXT_ENCODER,
        text_encoder_params=TEXT_ENCODER_CONFIG,
        prompt={"mode": "bare", "normalize_whitespace": True},
        label_normalization={},
        annotation_processing=annotation_processing,
        namespace=NAMESPACE,
        scope="dataset",
    )


def _train_or_load_condition(
    source_dataset: str,
    condition: str,
    output_dir: Path,
    device: str,
    max_epochs: int | None,
    train_limit: int | None,
    source_val_limit: int | None,
    progress: bool,
    num_workers: int | None,
    skip_training: bool,
    skip_existing: bool,
    mapping_path: Path,
    label_context: dict[str, Any],
    cache: dict[tuple[str, str], tuple[Path, dict[str, Any], float | None, dict[str, Any]]],
) -> tuple[Path, dict[str, Any], dict[str, Any], float | None]:
    cache_key = (source_dataset, condition)
    if cache_key in cache:
        checkpoint, config, seconds, metrics = cache[cache_key]
        return checkpoint, config, metrics, seconds

    config = _training_config(
        source_dataset=source_dataset,
        condition=condition,
        output_dir=output_dir,
        device=device,
        max_epochs=max_epochs,
        train_limit=train_limit,
        source_val_limit=source_val_limit,
        progress=progress,
        num_workers=num_workers,
        mapping_path=mapping_path,
        label_context=label_context,
    )
    metrics, seconds = _train_or_load(
        config=config,
        skip_training=skip_training,
        skip_existing=skip_existing,
    )
    checkpoint = _checkpoint_from_metrics_or_config(metrics, config)
    cache[cache_key] = (checkpoint, config, seconds, metrics)
    return checkpoint, config, metrics, seconds


def _training_config(
    source_dataset: str,
    condition: str,
    output_dir: Path,
    device: str,
    max_epochs: int | None,
    train_limit: int | None,
    source_val_limit: int | None,
    progress: bool,
    num_workers: int | None,
    mapping_path: Path,
    label_context: dict[str, Any],
) -> dict[str, Any]:
    dataset = DATASETS[source_dataset]
    config = load_yaml(dataset["config"])
    config["device"] = device
    config["output_dir"] = str(output_dir / "train" / source_dataset / condition)
    config["data"]["manifest"] = str(
        _limited_manifest(
            Path(dataset["train_manifest"]),
            output_dir / "manifests",
            train_limit,
            suffix=f"{source_dataset}_train",
        )
    )
    config["data"]["text_encoder"] = TEXT_ENCODER
    label_setup = _label_setup(source_dataset, condition, mapping_path, label_context)
    config["data"]["text_embedding_root"] = label_setup["text_embedding_root"]
    config["data"]["annotation_processing"] = label_setup["annotation_processing"]
    config.setdefault("validation", {})["manifest"] = str(
        _limited_manifest(
            Path(dataset["val_manifest"]),
            output_dir / "manifests",
            source_val_limit,
            suffix=f"{source_dataset}_val",
        )
    )
    config.setdefault("optimization", {})["progress"] = bool(progress)
    if num_workers is not None:
        config["optimization"]["num_workers"] = int(num_workers)
        if int(num_workers) <= 0:
            config["optimization"]["persistent_workers"] = False
            config["optimization"]["prefetch_factor"] = None
    if max_epochs is not None:
        config["optimization"]["max_epochs"] = int(max_epochs)
    return config


def _target_data_config(target_dataset: str) -> dict[str, Any]:
    config = load_yaml(DATASETS[target_dataset]["config"])
    data = dict(config["data"])
    data["text_encoder"] = TEXT_ENCODER
    return data


def _label_setup(
    dataset_name: str,
    condition: str,
    mapping_path: Path,
    label_context: dict[str, Any],
) -> dict[str, Any]:
    if condition == "real":
        return {
            "text_embedding_root": label_context["real_text_roots"][dataset_name],
            "annotation_processing": label_context["real_annotation_processing"],
        }
    if condition == "joint_corpus_ids":
        return {
            "text_embedding_root": label_context["joint_text_root"],
            "annotation_processing": _joint_corpus_annotation_processing(
                mapping_path,
                label_context=label_context,
            ),
        }
    raise ValueError(f"Unknown condition: {condition}")


def _joint_corpus_annotation_processing(
    mapping_path: Path,
    label_context: dict[str, Any],
) -> dict[str, Any]:
    return {
        "policy": "section_ids_corpus",
        "source_annotation_processing": label_context["real_annotation_processing"],
        "section_id_mapping_path": str(mapping_path),
        "section_id_preserve_labels": list(PRESERVE_LABELS),
    }


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


def _run_transfer_split(
    source_dataset: str,
    target_dataset: str,
    condition: str,
    split: str,
    checkpoint: Path,
    manifest: Path,
    limit: int | None,
    output_dir: Path,
    device: str,
    target_data_config: dict[str, Any],
    label_setup: dict[str, Any],
    real_label_setup: dict[str, Any],
    source_labels: set[str],
    mapping: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_manifest = _limited_manifest(
        manifest,
        output_dir / "manifests",
        limit,
        suffix=f"{target_dataset}_{split}",
    )
    segmentation = dict(load_yaml(DATASETS[target_dataset]["config"]).get("validation", {}).get("segmentation", {}))
    beat_subsampling = target_data_config.get("beat_subsampling", False)
    track_filter = target_data_config.get("track_filter", False)
    run_dir = output_dir / f"{source_dataset}_to_{target_dataset}" / condition / split
    predictions_dir = run_dir / "predictions"
    evaluation_path = run_dir / "evaluation.json"

    run_baseline_inference(
        checkpoint_path=checkpoint,
        manifest=run_manifest,
        audio_embedding_root=target_data_config["audio_embedding_root"],
        audio_encoder=target_data_config["audio_encoder"],
        text_embedding_root=label_setup["text_embedding_root"],
        text_encoder=TEXT_ENCODER,
        audio_embedding_key=target_data_config.get("audio_embedding_key", "beat_sync"),
        namespace=target_data_config.get("namespace", NAMESPACE),
        prediction_namespace=target_data_config.get("prediction_namespace", NAMESPACE),
        output_dir=predictions_dir,
        device=device,
        limit=None,
        candidate_label_strategy=target_data_config.get("candidate_label_strategy", "track_labels"),
        annotation_processing=label_setup["annotation_processing"],
        beat_subsampling=beat_subsampling,
        track_filter=track_filter,
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
        namespace=target_data_config.get("namespace", NAMESPACE),
        prediction_namespace=target_data_config.get("prediction_namespace", NAMESPACE),
        trim=bool(segmentation.get("trim", True)),
        reference_annotation_processing=label_setup["annotation_processing"],
        audio_embedding_root=target_data_config["audio_embedding_root"],
        audio_encoder=target_data_config["audio_encoder"],
        text_embedding_root=label_setup["text_embedding_root"],
        text_encoder=TEXT_ENCODER,
        audio_embedding_key=target_data_config.get("audio_embedding_key", "beat_sync"),
        candidate_label_strategy=target_data_config.get("candidate_label_strategy", "track_labels"),
        beat_subsampling=beat_subsampling,
        reference_track_filter=track_filter,
        ignore_index=int(target_data_config.get("ignore_index", IGNORE_INDEX)),
    )
    save_evaluation(evaluation_path, evaluation)

    zero_shot, per_label_rows = _zero_shot_frame_metrics(
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        condition=condition,
        split=split,
        manifest=run_manifest,
        predictions_root=predictions_dir,
        target_data_config=target_data_config,
        label_setup=label_setup,
        real_label_setup=real_label_setup,
        source_labels=source_labels,
        mapping=mapping,
    )
    overlap = _label_overlap(
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        split=split,
        source_labels=source_labels,
        target_labels=set(zero_shot.pop("_target_labels")),
    )
    _write_label_overlap(output_dir, overlap)

    row: dict[str, Any] = {
        "num_tracks": evaluation["num_tracks"],
        "predictions_dir": str(predictions_dir),
        "evaluation_json": str(evaluation_path),
    }
    for metric in METRICS:
        values = evaluation["summary"].get(metric, {})
        row[metric] = values.get("mean")
        row[f"{metric} std"] = values.get("std")
    row.update(zero_shot)
    return row, per_label_rows


def _zero_shot_frame_metrics(
    source_dataset: str,
    target_dataset: str,
    condition: str,
    split: str,
    manifest: Path,
    predictions_root: Path,
    target_data_config: dict[str, Any],
    label_setup: dict[str, Any],
    real_label_setup: dict[str, Any],
    source_labels: set[str],
    mapping: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    transformed_dataset = StructureEmbeddingDataset(
        manifest=manifest,
        audio_embedding_root=target_data_config["audio_embedding_root"],
        audio_encoder=target_data_config["audio_encoder"],
        text_embedding_root=label_setup["text_embedding_root"],
        text_encoder=TEXT_ENCODER,
        audio_embedding_key=target_data_config.get("audio_embedding_key", "beat_sync"),
        namespace=target_data_config.get("namespace", NAMESPACE),
        candidate_label_strategy=target_data_config.get("candidate_label_strategy", "track_labels"),
        annotation_processing=label_setup["annotation_processing"],
        beat_subsampling=target_data_config.get("beat_subsampling", False),
        track_filter=target_data_config.get("track_filter", False),
        ignore_index=int(target_data_config.get("ignore_index", IGNORE_INDEX)),
    )
    real_dataset = StructureEmbeddingDataset(
        manifest=manifest,
        audio_embedding_root=target_data_config["audio_embedding_root"],
        audio_encoder=target_data_config["audio_encoder"],
        text_embedding_root=real_label_setup["text_embedding_root"],
        text_encoder=TEXT_ENCODER,
        audio_embedding_key=target_data_config.get("audio_embedding_key", "beat_sync"),
        namespace=target_data_config.get("namespace", NAMESPACE),
        candidate_label_strategy=target_data_config.get("candidate_label_strategy", "track_labels"),
        annotation_processing=real_label_setup["annotation_processing"],
        beat_subsampling=target_data_config.get("beat_subsampling", False),
        track_filter=target_data_config.get("track_filter", False),
        ignore_index=int(target_data_config.get("ignore_index", IGNORE_INDEX)),
    )
    inverse_mapping = {target: source for source, target in mapping.items()}

    seen_correct = seen_count = 0
    unseen_correct = unseen_count = 0
    unseen_pred_correct = unseen_pred_count = 0
    valid_count = 0
    target_labels_seen: set[str] = set()
    per_label_correct: Counter[str] = Counter()
    per_label_count: Counter[str] = Counter()

    for transformed_example, real_example in zip(transformed_dataset, real_dataset, strict=True):
        if (
            transformed_example.track_id != real_example.track_id
            or transformed_example.dataset != real_example.dataset
        ):
            raise ValueError(
                "Transformed and real datasets are not aligned: "
                f"{transformed_example.dataset}/{transformed_example.track_id} vs "
                f"{real_example.dataset}/{real_example.track_id}"
            )
        prediction_json = predictions_root / transformed_example.dataset / f"{transformed_example.track_id}.json"
        with prediction_json.open("r", encoding="utf-8") as handle:
            prediction = json.load(handle)

        predicted = _prediction_indices_for_labels(
            prediction=prediction,
            labels=transformed_example.labels,
        )
        transformed_targets = transformed_example.targets.astype(np.int64, copy=False)
        real_targets = real_example.targets.astype(np.int64, copy=False)
        if len(predicted) != len(transformed_targets) or len(real_targets) != len(transformed_targets):
            raise ValueError(
                f"Frame length mismatch for {transformed_example.dataset}/{transformed_example.track_id}"
            )

        valid = (transformed_targets != IGNORE_INDEX) & (real_targets != IGNORE_INDEX)
        valid_count += int(valid.sum())
        if not np.any(valid):
            continue

        real_labels = np.asarray(real_example.labels, dtype=object)
        target_real_labels = real_labels[real_targets[valid]]
        target_labels_seen.update(str(label) for label in target_real_labels)
        predicted_labels = np.asarray(
            [
                transformed_example.labels[index] if 0 <= int(index) < len(transformed_example.labels) else ""
                for index in predicted[valid]
            ],
            dtype=object,
        )
        predicted_real_labels = np.asarray(
            [
                _underlying_label(str(label), condition=condition, inverse_mapping=inverse_mapping)
                for label in predicted_labels
            ],
            dtype=object,
        )

        target_unseen = np.asarray(
            [str(label) not in source_labels for label in target_real_labels],
            dtype=bool,
        )
        pred_unseen = np.asarray(
            [str(label) not in source_labels for label in predicted_real_labels],
            dtype=bool,
        )
        correct = predicted[valid] == transformed_targets[valid]

        seen_mask = ~target_unseen
        seen_count += int(seen_mask.sum())
        seen_correct += int(np.sum(correct & seen_mask))
        unseen_count += int(target_unseen.sum())
        unseen_correct += int(np.sum(correct & target_unseen))
        unseen_pred_count += int(pred_unseen.sum())
        unseen_pred_correct += int(np.sum(correct & pred_unseen & target_unseen))
        for label, is_correct in zip(target_real_labels[target_unseen], correct[target_unseen]):
            per_label_count[str(label)] += 1
            per_label_correct[str(label)] += int(bool(is_correct))

    per_label_rows = [
        {
            "source_train": source_dataset,
            "target_eval": target_dataset,
            "condition": condition,
            "split": split,
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
        "source_label_count": len(source_labels),
        "target_label_count": len(target_labels_seen),
        "target_unseen_label_count": len(target_labels_seen - source_labels),
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
        "_target_labels": sorted(target_labels_seen),
    }
    return metrics, per_label_rows


def _underlying_label(
    label: str,
    condition: str,
    inverse_mapping: dict[str, str],
) -> str:
    if condition == "joint_corpus_ids":
        return inverse_mapping.get(label, label)
    return label


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


def _manifest_labels_in_order(
    manifest: Path,
    annotation_processing: str | dict[str, Any] | None,
) -> list[str]:
    labels: list[str] = []
    for track in load_manifest(manifest):
        sections = load_processed_structure_sections(
            track.jams_path,
            namespace=NAMESPACE,
            annotation_processing=annotation_processing,
        )
        labels.extend(unique_labels(sections))
    return list(dict.fromkeys(labels))


def _manifest_label_set(
    manifest: Path,
    annotation_processing: str | dict[str, Any] | None,
) -> set[str]:
    return set(
        _manifest_labels_in_order(
            manifest,
            annotation_processing=annotation_processing,
        )
    )


def _label_overlap(
    source_dataset: str,
    target_dataset: str,
    split: str,
    source_labels: set[str],
    target_labels: set[str],
) -> dict[str, Any]:
    return {
        "source_train": source_dataset,
        "target_eval": target_dataset,
        "split": split,
        "annotation_processing": ANNOTATION_PROCESSING_REAL,
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
    base = output_dir / "label_overlap" / (
        f"{overlap['source_train']}_to_{overlap['target_eval']}_{overlap['split']}"
    )
    save_json(base.with_suffix(".json"), overlap)
    lines = [
        "# Label Overlap",
        "",
        f"- Source train: {overlap['source_train']}",
        f"- Target: {overlap['target_eval']} ({overlap['split']})",
        f"- Source labels: {overlap['source_label_count']}",
        f"- Target labels: {overlap['target_label_count']}",
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
    base.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def _limited_manifest(
    manifest: Path,
    output_dir: Path,
    limit: int | None,
    suffix: str,
) -> Path:
    if limit is None:
        return manifest
    output_dir.mkdir(parents=True, exist_ok=True)
    tracks = load_manifest(manifest)[:limit]
    limited = output_dir / f"{suffix}.first_{limit}.jsonl"
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
        return
    columns = _result_columns(rows)
    with (output_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _format_cell(row.get(column)) for column in columns})
    (output_dir / "results.md").write_text(_markdown_table(rows, columns), encoding="utf-8")
    save_json(output_dir / "results.json", {"rows": rows})
    _plot_results(output_dir, rows)


def _write_unseen_label_results(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = ["source_train", "target_eval", "condition", "split", "label", "frame_count", "frame_acc"]
    with (output_dir / "unseen_label_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _format_cell(row.get(column)) for column in columns})
    (output_dir / "unseen_label_metrics.md").write_text(
        _markdown_table(rows, columns),
        encoding="utf-8",
    )


def _result_columns(rows: list[dict[str, Any]]) -> list[str]:
    columns = [
        "direction",
        "condition",
        "source_train",
        "target_eval",
        "eval_num_tracks",
        *(f"eval_{metric}" for metric in METRICS),
        *(f"eval_{metric}" for metric in ZERO_SHOT_METRICS),
    ]
    if any("test_num_tracks" in row for row in rows):
        columns.extend(
            [
                "test_num_tracks",
                *(f"test_{metric}" for metric in METRICS),
                *(f"test_{metric}" for metric in ZERO_SHOT_METRICS),
            ]
        )
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
        "direction": "Direction",
        "condition": "Condition",
        "source_train": "Source Train",
        "target_eval": "Target Eval",
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
        "source_label_count": "Source Labels",
        "target_label_count": "Target Labels",
        "target_unseen_label_count": "Unseen Labels",
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
        if np.isnan(value):
            return "nan"
        return f"{value:.6f}"
    return str(value)


def _plot_results(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    plt = _matplotlib()
    plot_specs = (
        ("eval", "Cross-Dataset Eval Metrics", output_dir / "eval_metrics_barplot.png"),
        ("test", "Cross-Dataset Test Metrics", output_dir / "test_metrics_barplot.png"),
    )
    metrics = ("F-measure@0.5", "F-measure@3.0", "Acc", "Pairwise F-measure", "NCE F-measure")
    labels = {
        "F-measure@0.5": "F@0.5",
        "F-measure@3.0": "F@3.0",
        "Acc": "Acc",
        "Pairwise F-measure": "PFC",
        "NCE F-measure": "NCE",
    }
    for split, title, path in plot_specs:
        split_rows = [row for row in rows if row.get(f"{split}_num_tracks") is not None]
        if not split_rows:
            continue
        conditions = [
            f"{row['source_train']} -> {row['target_eval']}\n{row['condition']}"
            for row in split_rows
        ]
        x = np.arange(len(conditions), dtype=float)
        width = 0.15
        offsets = (np.arange(len(metrics), dtype=float) - (len(metrics) - 1) / 2.0) * width
        fig, ax = plt.subplots(figsize=(12.5, 5.6), constrained_layout=True)
        for offset, metric in zip(offsets, metrics, strict=True):
            values = [_float_or_nan(row.get(f"{split}_{metric}")) for row in split_rows]
            ax.bar(x + offset, values, width=width, label=labels[metric])
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=20, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Mean score")
        ax.set_title(title)
        ax.grid(axis="y", color="#dddddd", linewidth=0.8)
        ax.legend(ncols=3, frameon=False)
        fig.savefig(path, dpi=180)
        plt.close(fig)

    zero_metrics = ("seen_frame_acc", "unseen_frame_acc", "unseen_pred_precision")
    zero_labels = {
        "seen_frame_acc": "Seen Acc",
        "unseen_frame_acc": "Unseen Acc",
        "unseen_pred_precision": "Unseen Pred Precision",
    }
    split_rows = [row for row in rows if row.get("eval_num_tracks") is not None]
    if split_rows:
        conditions = [
            f"{row['source_train']} -> {row['target_eval']}\n{row['condition']}"
            for row in split_rows
        ]
        x = np.arange(len(conditions), dtype=float)
        width = 0.22
        offsets = (np.arange(len(zero_metrics), dtype=float) - 1.0) * width
        fig, ax = plt.subplots(figsize=(11.5, 5.4), constrained_layout=True)
        for offset, metric in zip(offsets, zero_metrics, strict=True):
            values = [_float_or_nan(row.get(f"eval_{metric}")) for row in split_rows]
            ax.bar(x + offset, values, width=width, label=zero_labels[metric])
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=20, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Frame-level score")
        ax.set_title("Eval seen/unseen target-label behavior")
        ax.grid(axis="y", color="#dddddd", linewidth=0.8)
        ax.legend(frameon=False)
        fig.savefig(output_dir / "eval_seen_unseen_frame_metrics.png", dpi=180)
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
