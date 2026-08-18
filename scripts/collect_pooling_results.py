#!/usr/bin/env python3
"""Collect beat-pooling experiment metrics into a Markdown results table.

Reads ``metrics.json`` from each experiment's ``output_dir`` and reports the
segmentation metrics at the best (monitored) epoch, alongside the best
validation loss and run status. Runs that have not finished (or not started)
are shown as such, so this is safe to run while the sweep is in progress.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# (display label, output_dir) for the beat-pooling sweep.
EXPERIMENTS = [
    ("mean", "outputs/train/harmonix_bigvgan_self_attention_pool_mean"),
    ("max", "outputs/train/harmonix_bigvgan_self_attention_pool_max"),
    ("energy_weighted", "outputs/train/harmonix_bigvgan_self_attention_pool_energy"),
    ("multi_stat", "outputs/train/harmonix_bigvgan_self_attention_pool_multistat"),
    ("first", "outputs/train/harmonix_bigvgan_self_attention_pool_first"),
    ("last", "outputs/train/harmonix_bigvgan_self_attention_pool_last"),
    ("attention (A2)", "outputs/train/harmonix_bigvgan_self_attention_beat_pool_attention"),
]

METRIC_COLUMNS = [
    ("F@0.5", "val_segmentation_F-measure@0.5"),
    ("F@3.0", "val_segmentation_F-measure@3.0"),
    ("PairwiseF", "val_segmentation_Pairwise F-measure"),
    ("NCE-F", "val_segmentation_NCE F-measure"),
    ("Acc", "val_segmentation_Acc"),
]


def _best_epoch_record(metrics: dict) -> dict | None:
    history = metrics.get("history") or []
    if not history:
        return None
    best_epoch = metrics.get("best_segmentation_epoch")  # 1-indexed
    if best_epoch is not None and 1 <= best_epoch <= len(history):
        return history[best_epoch - 1]
    # Fall back to the row maximizing the monitored metric.
    monitor = "val_segmentation_" + str(metrics.get("best_segmentation_metric", "F-measure@3.0"))
    scored = [r for r in history if monitor in r]
    if scored:
        return max(scored, key=lambda r: r[monitor])
    return history[-1]


def _row(label: str, output_dir: str) -> tuple[str, str]:
    path = Path(output_dir) / "metrics.json"
    if not path.exists():
        cells = ["—"] * len(METRIC_COLUMNS) + ["—", "not started"]
        return label, "| " + " | ".join([label, *cells]) + " |"
    metrics = json.loads(path.read_text())
    record = _best_epoch_record(metrics)
    values = []
    for _, key in METRIC_COLUMNS:
        value = None if record is None else record.get(key)
        values.append("—" if value is None else f"{value:.3f}")
    best_val = metrics.get("best_val_loss")
    values.append("—" if best_val is None else f"{best_val:.3f}")
    epochs = metrics.get("epochs_trained")
    best_epoch = metrics.get("best_segmentation_epoch")
    status = f"{epochs} ep"
    if best_epoch is not None:
        status += f" (best @{best_epoch})"
    if not metrics.get("stopped_early"):
        status += "*"  # did not early-stop -> may be truncated / still capped
    values.append(status)
    return label, "| " + " | ".join([label, *values]) + " |"


def build_table() -> str:
    header_cells = ["pooling", *[name for name, _ in METRIC_COLUMNS], "best val loss", "epochs"]
    lines = [
        "| " + " | ".join(header_cells) + " |",
        "|" + "|".join(["---"] * len(header_cells)) + "|",
    ]
    for label, output_dir in EXPERIMENTS:
        _, row = _row(label, output_dir)
        lines.append(row)
    note = (
        "\nMetrics are on the Harmonix val split at the best F-measure@3.0 epoch. "
        "F@0.5 / F@3.0 = boundary hit-rate F; PairwiseF / NCE-F = structure "
        "labelling; Acc = frame accuracy. `*` = run did not early-stop.\n"
    )
    return "\n".join(lines) + "\n" + note


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="docs/beat_pooling_results.md",
        help="Write the Markdown table here (also printed to stdout).",
    )
    args = parser.parse_args()
    table = "# Beat-Pooling Results (Harmonix)\n\n" + build_table()
    print(table)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(table + "\n", encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
