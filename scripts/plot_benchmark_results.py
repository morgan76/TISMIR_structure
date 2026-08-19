#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from pathlib import Path
from typing import Any


METRICS = [
    "F-measure@0.5",
    "F-measure@3.0",
    "Acc",
    "Balanced Acc",
    "Pairwise F-measure",
    "NCE F-measure",
]

METRIC_LABELS = {
    "F-measure@0.5": "F@0.5",
    "F-measure@3.0": "F@3.0",
    "Acc": "Acc",
    "Balanced Acc": "Bal Acc",
    "Pairwise F-measure": "PFC",
    "NCE F-measure": "NCE-F",
}

EXPERIMENT_ORDER = [
    "projection_ce",
    "adapter_rope_ce",
    "hybrid_ce",
    "hybrid_ce_link",
    "hybrid_ce_boundary",
    "hybrid_ce_link_boundary",
    "hybrid_ce_link_boundary_cosine",
    "hybrid_ce_link_boundary_section_ids_corpus",
    "hybrid_ce_link_boundary_section_ids_ordered",
    "hybrid_ce_link_boundary_section_ids_shuffled",
]

DECODER_ORDER = [
    "argmax",
    "viterbi",
    "boundary_viterbi",
    "boundary_peak_mean_logits",
    "boundary_peak_majority_vote",
]

EXPERIMENT_LABELS = {
    "projection_ce": "Projection\nCE",
    "adapter_rope_ce": "Adapter\nRoPE CE",
    "hybrid_ce": "Hybrid\nCE",
    "hybrid_ce_link": "Hybrid\nCE+Link",
    "hybrid_ce_boundary": "Hybrid\nCE+Boundary",
    "hybrid_ce_link_boundary": "Hybrid\nCE+Link+Boundary",
    "hybrid_ce_link_boundary_cosine": "Hybrid\nCosine Only",
    "hybrid_ce_link_boundary_section_ids_corpus": "Hybrid\nCorpus IDs",
    "hybrid_ce_link_boundary_section_ids_ordered": "Hybrid\nOrdered IDs",
    "hybrid_ce_link_boundary_section_ids_shuffled": "Hybrid\nShuffled IDs",
}

DECODER_LABELS = {
    "argmax": "Argmax",
    "viterbi": "Viterbi",
    "boundary_viterbi": "Boundary\nViterbi",
    "boundary_peak_mean_logits": "Peak\nMean Logits",
    "boundary_peak_majority_vote": "Peak\nMajority",
}

COLORS = {
    "eval": "#4E79A7",
    "test": "#F28E2B",
    "gap": "#E15759",
    "neutral": "#76B7B2",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark tables produced by run_rwc_pop_benchmark.py.")
    parser.add_argument("--results", default="outputs/benchmarks/rwc_pop_mert/results.csv")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--selection-metric", default="F-measure@3.0", choices=METRICS)
    args = parser.parse_args()

    results_path = Path(args.results)
    output_dir = Path(args.output_dir) if args.output_dir else results_path.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_rows(results_path)
    selected = _select_best_by_eval(rows, args.selection_metric)

    _plot_best_eval_test_bars(selected, output_dir, args.selection_metric)
    _plot_test_metric_heatmap(rows, output_dir)
    _plot_decoder_effects(rows, output_dir)
    _plot_generalization_gaps(selected, output_dir)
    _plot_training_time(selected, output_dir)
    _write_summary(selected, rows, output_dir, args.selection_metric)
    print(f"Saved benchmark plots to {output_dir}")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    for row in rows:
        for key, value in list(row.items()):
            if value == "":
                row[key] = None
            elif key not in {"experiment", "decoder", "checkpoint", "eval_evaluation_json", "test_evaluation_json"}:
                row[key] = _to_float(value)
    return rows


def _to_float(value: str) -> float | str:
    try:
        return float(value)
    except ValueError:
        return value


def _select_best_by_eval(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    selected = []
    for experiment in _ordered_unique(row["experiment"] for row in rows):
        candidates = [row for row in rows if row["experiment"] == experiment]
        key = f"eval_{metric}"
        selected.append(max(candidates, key=lambda row: _number(row.get(key))))
    return selected


def _plot_best_eval_test_bars(
    rows: list[dict[str, Any]],
    output_dir: Path,
    selection_metric: str,
) -> None:
    plt = _matplotlib()
    metrics = ["F-measure@0.5", "F-measure@3.0", "Pairwise F-measure", "NCE F-measure"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    labels = [_experiment_label(row["experiment"]) + f"\n{_decoder_label(row['decoder'])}" for row in rows]
    x = list(range(len(rows)))
    width = 0.38
    for ax, metric in zip(axes.flat, metrics):
        eval_values = [_number(row.get(f"eval_{metric}")) for row in rows]
        test_values = [_number(row.get(f"test_{metric}")) for row in rows]
        ax.bar([pos - width / 2 for pos in x], eval_values, width, label="Eval", color=COLORS["eval"])
        ax.bar([pos + width / 2 for pos in x], test_values, width, label="Test", color=COLORS["test"])
        ax.set_title(METRIC_LABELS[metric])
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
        _annotate_bars(ax)
    axes.flat[0].legend(loc="upper left", frameon=False)
    fig.suptitle(f"Best Decoder Per Model, Selected By Eval {METRIC_LABELS[selection_metric]}", fontsize=14)
    fig.savefig(output_dir / "best_decoder_eval_test_metrics.png", dpi=180)
    plt.close(fig)


def _plot_test_metric_heatmap(rows: list[dict[str, Any]], output_dir: Path) -> None:
    plt = _matplotlib()
    labels = [f"{_experiment_label(row['experiment'])}\n{_decoder_label(row['decoder'])}" for row in rows]
    data = [[_number(row.get(f"test_{metric}")) for metric in METRICS] for row in rows]
    color_data = _column_normalized_values(data)
    fig_height = max(6, 0.42 * len(rows) + 2.5)
    fig, ax = plt.subplots(figsize=(10.5, fig_height), constrained_layout=True)
    ax.imshow(color_data, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_title("Test Metrics For Every Model/Decoder (colors normalized per metric)")
    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels([METRIC_LABELS[metric] for metric in METRICS], rotation=25, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for row_index, values in enumerate(data):
        for col_index, value in enumerate(values):
            color = "white" if value < 0.55 else "black"
            ax.text(col_index, row_index, f"{value:.3f}", ha="center", va="center", fontsize=8, color=color)
    fig.savefig(output_dir / "test_metrics_heatmap.png", dpi=180)
    plt.close(fig)


def _plot_decoder_effects(rows: list[dict[str, Any]], output_dir: Path) -> None:
    plt = _matplotlib()
    boundary_rows = [
        row
        for row in rows
        if row["experiment"] in {"hybrid_ce_boundary", "hybrid_ce_link_boundary"}
    ]
    if not boundary_rows:
        return
    experiments = _ordered_unique(row["experiment"] for row in boundary_rows)
    metrics = ["F-measure@0.5", "F-measure@3.0"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4.8), constrained_layout=True)
    for ax, metric in zip(axes, metrics):
        labels = []
        values = []
        colors = []
        for experiment in experiments:
            for decoder in DECODER_ORDER:
                row = _find_row(boundary_rows, experiment, decoder)
                if row is None:
                    continue
                labels.append(f"{_experiment_label(experiment)}\n{_decoder_label(decoder)}")
                values.append(_number(row.get(f"test_{metric}")))
                colors.append(COLORS["test"] if "boundary" in decoder else COLORS["neutral"])
        ax.bar(range(len(values)), values, color=colors)
        ax.set_title(f"Boundary Decoder Effect: Test {METRIC_LABELS[metric]}")
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=40, ha="right")
        ax.grid(axis="y", alpha=0.25)
        _annotate_bars(ax)
    fig.savefig(output_dir / "boundary_decoder_effects.png", dpi=180)
    plt.close(fig)


def _plot_generalization_gaps(rows: list[dict[str, Any]], output_dir: Path) -> None:
    plt = _matplotlib()
    metrics = ["F-measure@0.5", "F-measure@3.0", "Pairwise F-measure", "NCE F-measure"]
    labels = [_experiment_label(row["experiment"]) + f"\n{_decoder_label(row['decoder'])}" for row in rows]
    x = list(range(len(rows)))
    width = 0.18
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    fig, ax = plt.subplots(figsize=(14, 5.2), constrained_layout=True)
    for offset, metric in zip(offsets, metrics):
        gaps = [
            _number(row.get(f"test_{metric}")) - _number(row.get(f"eval_{metric}"))
            for row in rows
        ]
        ax.bar([pos + offset for pos in x], gaps, width, label=METRIC_LABELS[metric])
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("Generalization Gap For Selected Decoders: Test Minus Eval")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Test - Eval")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=4, loc="lower left")
    fig.savefig(output_dir / "generalization_gaps.png", dpi=180)
    plt.close(fig)


def _plot_training_time(rows: list[dict[str, Any]], output_dir: Path) -> None:
    plt = _matplotlib()
    labels = [_experiment_label(row["experiment"]) for row in rows]
    times_minutes = [_number(row.get("train_seconds")) / 60.0 for row in rows]
    test_f3 = [_number(row.get("test_F-measure@3.0")) for row in rows]
    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    ax.scatter(times_minutes, test_f3, s=70, color=COLORS["neutral"])
    for label, x_value, y_value in zip(labels, times_minutes, test_f3):
        ax.annotate(label.replace("\n", " "), (x_value, y_value), xytext=(5, 4), textcoords="offset points")
    ax.set_title("Training Cost Vs Test F@3.0")
    ax.set_xlabel("Training time (minutes)")
    ax.set_ylabel("Test F@3.0")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    fig.savefig(output_dir / "training_time_vs_test_f3.png", dpi=180)
    plt.close(fig)


def _write_summary(
    selected: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    output_dir: Path,
    selection_metric: str,
) -> None:
    best_test_f3 = max(rows, key=lambda row: _number(row.get("test_F-measure@3.0")))
    best_test_f05 = max(rows, key=lambda row: _number(row.get("test_F-measure@0.5")))
    best_eval_selection = max(selected, key=lambda row: _number(row.get(f"eval_{selection_metric}")))
    lines = [
        "# RWC-Pop Benchmark Plot Summary",
        "",
        f"Best decoder per model is selected by eval {METRIC_LABELS[selection_metric]}.",
        "",
        "## Best Overall Rows",
        "",
        (
            f"- Best test F@3.0: `{best_test_f3['experiment']}` / `{best_test_f3['decoder']}` "
            f"= {_number(best_test_f3.get('test_F-measure@3.0')):.3f}"
        ),
        (
            f"- Best test F@0.5: `{best_test_f05['experiment']}` / `{best_test_f05['decoder']}` "
            f"= {_number(best_test_f05.get('test_F-measure@0.5')):.3f}"
        ),
        (
            f"- Best selected eval row: `{best_eval_selection['experiment']}` / "
            f"`{best_eval_selection['decoder']}` "
            f"eval {METRIC_LABELS[selection_metric]} = "
            f"{_number(best_eval_selection.get(f'eval_{selection_metric}')):.3f}"
        ),
        "",
        "## Files",
        "",
        "- `best_decoder_eval_test_metrics.png`",
        "- `test_metrics_heatmap.png`",
        "- `boundary_decoder_effects.png`",
        "- `generalization_gaps.png`",
        "- `training_time_vs_test_f3.png`",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _matplotlib():
    cache_root = Path(tempfile.gettempdir()) / "tismir_plot_cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )
    return plt


def _ordered_unique(values) -> list[str]:
    unique = list(dict.fromkeys(values))
    return sorted(
        unique,
        key=lambda value: (
            EXPERIMENT_ORDER.index(value) if value in EXPERIMENT_ORDER else math.inf,
            value,
        ),
    )


def _column_normalized_values(data: list[list[float]]) -> list[list[float]]:
    if not data:
        return data
    columns = list(zip(*data))
    normalized_columns = []
    for column in columns:
        finite_values = [value for value in column if not math.isnan(value)]
        if not finite_values:
            normalized_columns.append([float("nan") for _ in column])
            continue
        vmin = min(finite_values)
        vmax = max(finite_values)
        if math.isclose(vmin, vmax):
            normalized_columns.append([0.5 if not math.isnan(value) else float("nan") for value in column])
            continue
        normalized_columns.append(
            [
                float("nan") if math.isnan(value) else (value - vmin) / (vmax - vmin)
                for value in column
            ]
        )
    return [list(row) for row in zip(*normalized_columns)]


def _find_row(rows: list[dict[str, Any]], experiment: str, decoder: str) -> dict[str, Any] | None:
    for row in rows:
        if row["experiment"] == experiment and row["decoder"] == decoder:
            return row
    return None


def _experiment_label(experiment: str) -> str:
    return EXPERIMENT_LABELS.get(experiment, experiment.replace("_", "\n"))


def _decoder_label(decoder: str) -> str:
    return DECODER_LABELS.get(decoder, decoder.replace("_", "\n"))


def _number(value: Any) -> float:
    if value is None:
        return float("nan")
    return float(value)


def _annotate_bars(ax) -> None:
    for patch in ax.patches:
        height = patch.get_height()
        if math.isnan(height):
            continue
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height + 0.012,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
        )


if __name__ == "__main__":
    main()
