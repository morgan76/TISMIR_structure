#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


EXPERIMENTS = [
    {
        "name": "e5_base_query_prefix",
        "text_encoder": "sentence_transformers",
        "checkpoint": "intfloat/e5-base-v2",
        "prompt_prefix": "query: ",
    },
    {
        "name": "all_mpnet_base_v2",
        "text_encoder": "sentence_transformers",
        "checkpoint": "sentence-transformers/all-mpnet-base-v2",
        "prompt_prefix": "",
    },
    {
        "name": "bge_base_en_v1_5",
        "text_encoder": "sentence_transformers",
        "checkpoint": "BAAI/bge-base-en-v1.5",
        "prompt_prefix": "",
    },
    {
        "name": "clap_music_speech",
        "text_encoder": "clap",
        "checkpoint": "laion/larger_clap_music_and_speech",
        "prompt_prefix": "",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Harmonix text-only geometry diagnostics for several text encoders."
    )
    parser.add_argument("--manifest", default="data/manifests/harmonix.local.jsonl")
    parser.add_argument("--namespace", default="segment_open")
    parser.add_argument("--annotation-policy", default="base_labels")
    parser.add_argument(
        "--section-id-mapping",
        default="configs/annotation_mappings/harmonix_section_ids_corpus_base_labels_seed0.json",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/text_geometry/harmonix_text_encoder_sweep",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip an experiment when its summary_metrics.csv already exists.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for experiment in EXPERIMENTS:
        run_dir = output_dir / experiment["name"]
        summary_path = run_dir / "summary_metrics.csv"
        if args.skip_existing and summary_path.exists():
            print(f"Skipping {experiment['name']} because {summary_path} exists")
            continue

        command = [
            sys.executable,
            "scripts/diagnose_text_geometry.py",
            "--manifest",
            args.manifest,
            "--namespace",
            args.namespace,
            "--annotation-policy",
            args.annotation_policy,
            "--condition-set",
            "root_real_vs_corpus_ids",
            "--section-id-mapping",
            args.section_id_mapping,
            "--output-dir",
            str(run_dir),
            "--text-encoder",
            experiment["text_encoder"],
            "--checkpoint",
            experiment["checkpoint"],
            "--batch-size",
            str(args.batch_size),
        ]
        if args.device:
            command.extend(["--device", args.device])
        if experiment["prompt_prefix"]:
            command.extend(["--prompt-prefix", experiment["prompt_prefix"]])

        print(f"=== {experiment['name']} ===")
        subprocess.run(command, check=True)

    _write_combined_summary(output_dir)
    print(f"Saved Harmonix text-encoder geometry sweep to {output_dir}")


def _write_combined_summary(output_dir: Path) -> None:
    rows = []
    for experiment in EXPERIMENTS:
        summary_path = output_dir / experiment["name"] / "summary_metrics.csv"
        if not summary_path.exists():
            continue
        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                row = dict(row)
                row["experiment"] = experiment["name"]
                row["text_encoder"] = experiment["text_encoder"]
                row["checkpoint"] = experiment["checkpoint"]
                row["prompt_prefix"] = experiment["prompt_prefix"]
                rows.append(row)

    if not rows:
        return

    fields = [
        "experiment",
        "condition",
        "text_encoder",
        "checkpoint",
        "prompt_prefix",
        "num_labels",
        "embedding_dim",
        "mean_offdiag_cosine",
        "std_offdiag_cosine",
        "mean_same_base_cosine",
        "mean_different_base_cosine",
        "same_minus_different",
        "top1_same_base_accuracy",
        "mean_nearest_same_base_rank",
        "effective_rank",
        "participation_ratio",
        "pc1_explained",
        "pc2_explained",
        "centroid_norm",
        "uniformity_t2",
        "base_silhouette_cosine",
        "nearest_same_base_margin",
        "top1_neighbor_indegree_gini",
    ]
    csv_path = output_dir / "combined_summary_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    md_path = output_dir / "summary.md"
    metrics = [
        "mean_offdiag_cosine",
        "same_minus_different",
        "effective_rank",
        "participation_ratio",
        "pc1_explained",
        "centroid_norm",
        "uniformity_t2",
        "base_silhouette_cosine",
        "nearest_same_base_margin",
        "top1_neighbor_indegree_gini",
    ]
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Harmonix Text Encoder Geometry Sweep\n\n")
        handle.write(
            "| Experiment | Condition | "
            + " | ".join(_short_metric(metric) for metric in metrics)
            + " |\n"
        )
        handle.write(
            "| --- | --- | " + " | ".join("---:" for _ in metrics) + " |\n"
        )
        for row in rows:
            values = [_format_float(row.get(metric, "")) for metric in metrics]
            handle.write(
                f"| {row['experiment']} | {row['condition']} | "
                + " | ".join(values)
                + " |\n"
            )


def _format_float(value: str) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return ""


def _short_metric(name: str) -> str:
    return {
        "mean_offdiag_cosine": "offdiag",
        "same_minus_different": "same-diff",
        "effective_rank": "eff rank",
        "participation_ratio": "part ratio",
        "pc1_explained": "PC1",
        "centroid_norm": "centroid",
        "uniformity_t2": "uniformity",
        "base_silhouette_cosine": "silhouette",
        "nearest_same_base_margin": "same margin",
        "top1_neighbor_indegree_gini": "top1 gini",
    }.get(name, name)


if __name__ == "__main__":
    main()
