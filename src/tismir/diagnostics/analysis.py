from __future__ import annotations

import json
import os
from html import escape
from pathlib import Path
from typing import Any

import numpy as np

from tismir.models import build_model
from tismir.training.data import StructureEmbeddingDataset


def run_diagnostics(
    checkpoint_path: str | Path,
    manifest: str | Path,
    audio_embedding_root: str | Path,
    audio_encoder: str,
    text_embedding_root: str | Path,
    text_encoder: str,
    output_dir: str | Path,
    audio_embedding_key: str = "beat_sync",
    namespace: str = "segment_open",
    device: str = "auto",
    limit: int | None = None,
    candidate_label_strategy: str = "track_labels",
    annotation_processing: str | dict[str, Any] | None = None,
    max_plots: int = 20,
    audio_audio_max_frames: int = 512,
) -> list[dict[str, Any]]:
    """Run token-similarity diagnostics over a manifest."""

    torch = _require_torch()
    device_obj = _resolve_device(device, torch)
    checkpoint = torch.load(checkpoint_path, map_location=device_obj)
    config = checkpoint["config"]
    if annotation_processing is None:
        annotation_processing = config.get("data", {}).get("annotation_processing")

    dataset = StructureEmbeddingDataset(
        manifest=manifest,
        audio_embedding_root=audio_embedding_root,
        audio_encoder=audio_encoder,
        text_embedding_root=text_embedding_root,
        text_encoder=text_encoder,
        audio_embedding_key=audio_embedding_key,
        namespace=namespace,
        candidate_label_strategy=candidate_label_strategy,
        annotation_processing=annotation_processing,
        ignore_index=int(config.get("data", {}).get("ignore_index", -100)),
    )

    model = build_model(config.get("model", {}), checkpoint["audio_dim"], checkpoint["text_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device_obj)
    model.eval()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    count = len(dataset) if limit is None else min(limit, len(dataset))
    for index in range(count):
        example = dataset[index]
        with torch.inference_mode():
            audio = torch.from_numpy(example.audio).unsqueeze(0).to(device_obj)
            text = torch.from_numpy(example.text).to(device_obj)
            features = model.extract_features(audio, text)

        audio_tokens = features["audio_tokens"][0].detach().cpu().numpy().astype(np.float32)
        text_tokens = features["text_tokens"].detach().cpu().numpy().astype(np.float32)
        if text_tokens.ndim == 3:
            text_tokens = text_tokens[0]
        audio_text = features["similarity"][0].detach().cpu().numpy().astype(np.float32)
        logits = features["logits"][0].detach().cpu().numpy().astype(np.float32)
        predictions = logits.argmax(axis=-1).astype(np.int64)

        text_text = _cosine_matrix(text_tokens, text_tokens)
        audio_indices = _sample_indices(len(audio_tokens), max_count=audio_audio_max_frames)
        audio_audio = _cosine_matrix(audio_tokens[audio_indices], audio_tokens[audio_indices])
        summary = _summarize_track(
            text_text=text_text,
            audio_audio=audio_audio,
            audio_audio_indices=audio_indices,
            audio_text=audio_text,
            targets=example.targets,
            predictions=predictions,
            ignore_index=int(config.get("data", {}).get("ignore_index", -100)),
        )

        track_dir = output_dir / example.dataset / example.track_id
        track_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            track_dir / "diagnostics.npz",
            text_text_similarity=text_text,
            audio_text_similarity=audio_text,
            audio_audio_similarity=audio_audio,
            audio_audio_indices=audio_indices.astype(np.int64),
            targets=example.targets.astype(np.int64),
            predictions=predictions.astype(np.int64),
            beat_intervals=np.asarray(example.beat_intervals, dtype=np.float32),
            labels=np.asarray(example.labels, dtype=object),
        )
        _save_track_json(track_dir / "summary.json", example, summary)
        if index < max_plots:
            _save_plots(
                track_dir=track_dir,
                labels=example.labels,
                targets=example.targets,
                predictions=predictions,
                text_text=text_text,
                audio_text=audio_text,
                audio_audio=audio_audio,
                audio_audio_indices=audio_indices,
            )

        result = {
            "track_id": example.track_id,
            "dataset": example.dataset,
            "num_frames": int(len(example.audio)),
            "num_labels": int(len(example.labels)),
            "track_dir": str(track_dir),
            **summary,
        }
        results.append(result)
        print(
            f"{example.track_id}: frames={result['num_frames']} labels={result['num_labels']} "
            f"acc={summary['frame_accuracy']:.3f} gt_margin={summary['audio_text_gt_margin_mean']:.3f}"
        )

    corpus_summary = _summarize_corpus(results)
    payload = {
        "checkpoint": str(checkpoint_path),
        "manifest": str(manifest),
        "num_tracks": len(results),
        "summary": corpus_summary,
        "tracks": results,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return results


def _summarize_track(
    text_text: np.ndarray,
    audio_audio: np.ndarray,
    audio_audio_indices: np.ndarray,
    audio_text: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
    ignore_index: int,
) -> dict[str, float]:
    valid = targets != ignore_index
    valid_targets = targets[valid].astype(int)
    valid_predictions = predictions[valid].astype(int)
    offdiag = text_text[~np.eye(len(text_text), dtype=bool)] if len(text_text) > 1 else np.asarray([])

    gt_scores = audio_text[valid, valid_targets] if valid.any() else np.asarray([])
    masked_scores = audio_text[valid].copy() if valid.any() else np.empty((0, audio_text.shape[1]))
    if len(masked_scores):
        masked_scores[np.arange(len(masked_scores)), valid_targets] = -np.inf
        wrong_scores = masked_scores.max(axis=1)
        margins = gt_scores - wrong_scores
    else:
        margins = np.asarray([])

    sampled_targets = targets[audio_audio_indices]
    sampled_valid = sampled_targets != ignore_index
    same_mask = sampled_targets[:, None] == sampled_targets[None, :]
    valid_pair_mask = sampled_valid[:, None] & sampled_valid[None, :]
    offdiag_pair_mask = ~np.eye(len(audio_audio), dtype=bool)
    same_values = audio_audio[valid_pair_mask & offdiag_pair_mask & same_mask]
    different_values = audio_audio[valid_pair_mask & offdiag_pair_mask & ~same_mask]

    audio_change = 1.0 - np.sum(_normalize_rows(audio_text[:-1]) * _normalize_rows(audio_text[1:]), axis=1)
    boundary_mask = _boundary_mask(targets, ignore_index=ignore_index)
    non_boundary_mask = _non_boundary_mask(targets, ignore_index=ignore_index)

    return {
        "frame_accuracy": _safe_mean((valid_predictions == valid_targets).astype(float)),
        "text_text_offdiag_mean": _safe_mean(offdiag),
        "text_text_offdiag_max": _safe_max(offdiag),
        "audio_text_gt_score_mean": _safe_mean(gt_scores),
        "audio_text_gt_margin_mean": _safe_mean(margins),
        "audio_audio_same_label_mean": _safe_mean(same_values),
        "audio_audio_different_label_mean": _safe_mean(different_values),
        "audio_text_change_at_boundary_mean": _safe_mean(audio_change[boundary_mask]),
        "audio_text_change_non_boundary_mean": _safe_mean(audio_change[non_boundary_mask]),
    }


def _save_track_json(path: Path, example, summary: dict[str, float]) -> None:
    payload = {
        "track_id": example.track_id,
        "dataset": example.dataset,
        "labels": example.labels,
        "num_frames": len(example.audio),
        "summary": summary,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _save_plots(
    track_dir: Path,
    labels: list[str],
    targets: np.ndarray,
    predictions: np.ndarray,
    text_text: np.ndarray,
    audio_text: np.ndarray,
    audio_audio: np.ndarray,
    audio_audio_indices: np.ndarray,
) -> None:
    _save_svg_plots(
        track_dir=track_dir,
        labels=labels,
        targets=targets,
        predictions=predictions,
        text_text=text_text,
        audio_text=audio_text,
        audio_audio=audio_audio,
        audio_audio_indices=audio_audio_indices,
    )
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional plotting dependency
        return

    boundaries = np.flatnonzero(_boundary_mask(targets, ignore_index=-100)) + 1

    fig, ax = plt.subplots(figsize=(max(7.0, len(labels) * 0.7), max(5.0, len(labels) * 0.55)))
    image = ax.imshow(text_text, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title("Text-token cosine similarity")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(track_dir / "text_text_similarity.png", dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(17, max(5.5, len(labels) * 0.42 + 1.8)))
    grid = fig.add_gridspec(
        nrows=3,
        ncols=2,
        width_ratios=[40, 1],
        height_ratios=[max(3.0, len(labels) * 0.38), 0.28, 0.28],
        hspace=0.08,
        wspace=0.02,
    )
    ax = fig.add_subplot(grid[0, 0])
    reference_ax = fig.add_subplot(grid[1, 0], sharex=ax)
    prediction_ax = fig.add_subplot(grid[2, 0], sharex=ax)
    colorbar_ax = fig.add_subplot(grid[0, 1])

    image = ax.imshow(
        audio_text.T,
        aspect="auto",
        vmin=-1.0,
        vmax=1.0,
        cmap="coolwarm",
        origin="lower",
        interpolation="nearest",
    )
    for boundary in boundaries:
        ax.axvline(boundary - 0.5, color="black", linewidth=0.35, alpha=0.3)
        reference_ax.axvline(boundary - 0.5, color="black", linewidth=0.25, alpha=0.25)
        prediction_ax.axvline(boundary - 0.5, color="black", linewidth=0.25, alpha=0.25)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title("Audio-token to text-token similarity")
    ax.tick_params(axis="x", labelbottom=False)
    fig.colorbar(image, cax=colorbar_ax)

    label_cmap = _label_colormap(plt, len(labels))
    reference_values = np.where(targets == -100, len(labels), targets)[None, :]
    prediction_values = predictions[None, :]
    reference_ax.imshow(reference_values, aspect="auto", interpolation="nearest", cmap=label_cmap, vmin=-0.5, vmax=len(labels) + 0.5)
    prediction_ax.imshow(prediction_values, aspect="auto", interpolation="nearest", cmap=label_cmap, vmin=-0.5, vmax=len(labels) + 0.5)
    _annotate_label_strip(reference_ax, targets, labels)
    _annotate_label_strip(prediction_ax, predictions, labels)
    reference_ax.set_yticks([])
    prediction_ax.set_yticks([])
    reference_ax.set_ylabel("ref", rotation=0, ha="right", va="center", fontsize=9)
    prediction_ax.set_ylabel("pred", rotation=0, ha="right", va="center", fontsize=9)
    prediction_ax.set_xlabel("beat-synchronous frame")
    reference_ax.tick_params(axis="x", labelbottom=False)
    fig.savefig(track_dir / "audio_text_similarity.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(audio_audio, vmin=-1.0, vmax=1.0, cmap="coolwarm", origin="lower")
    sampled_boundary_positions = np.searchsorted(audio_audio_indices, boundaries)
    for boundary in sampled_boundary_positions:
        if 0 <= boundary < len(audio_audio_indices):
            ax.axvline(boundary - 0.5, color="white", linewidth=0.35, alpha=0.55)
            ax.axhline(boundary - 0.5, color="white", linewidth=0.35, alpha=0.55)
    ax.set_title("Audio-token self-similarity")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(track_dir / "audio_audio_similarity.png", dpi=160)
    plt.close(fig)


def _save_svg_plots(
    track_dir: Path,
    labels: list[str],
    targets: np.ndarray,
    predictions: np.ndarray,
    text_text: np.ndarray,
    audio_text: np.ndarray,
    audio_audio: np.ndarray,
    audio_audio_indices: np.ndarray,
) -> None:
    _save_text_text_svg(track_dir / "text_text_similarity.svg", labels, text_text)
    _save_audio_text_svg(
        track_dir / "audio_text_similarity.svg",
        labels=labels,
        targets=targets,
        predictions=predictions,
        matrix=audio_text,
    )
    _save_audio_audio_svg(
        track_dir / "audio_audio_similarity.svg",
        matrix=audio_audio,
        source_indices=audio_audio_indices,
        targets=targets,
    )


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


def _save_text_text_svg(path: Path, labels: list[str], matrix: np.ndarray) -> None:
    cell = 26
    left = 140
    top = 30
    width = left + cell * len(labels) + 20
    height = top + cell * len(labels) + 120
    parts = [_svg_header(width, height)]
    parts.append(f'<text x="{left}" y="18" font-size="13">Text-token cosine similarity</text>')
    for row, label in enumerate(labels):
        y = top + row * cell
        parts.append(f'<text x="{left - 8}" y="{y + 17}" text-anchor="end" font-size="10">{escape(label)}</text>')
        for col, value in enumerate(matrix[row]):
            x = left + col * cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{_heat_color(float(value))}"/>')
    for col, label in enumerate(labels):
        x = left + col * cell + 17
        y = top + cell * len(labels) + 8
        parts.append(
            f'<text x="{x}" y="{y}" font-size="10" transform="rotate(90 {x} {y})">{escape(label)}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _save_audio_text_svg(
    path: Path,
    labels: list[str],
    targets: np.ndarray,
    predictions: np.ndarray,
    matrix: np.ndarray,
) -> None:
    frame_indices = _sample_indices(len(matrix), max_count=900)
    sampled = matrix[frame_indices].T
    cell_w = 1.4
    cell_h = 24
    left = 140
    top = 32
    plot_width = cell_w * len(frame_indices)
    plot_height = cell_h * len(labels)
    width = int(left + plot_width + 20)
    height = int(top + plot_height + 36)
    parts = [_svg_header(width, height)]
    parts.append(f'<text x="{left}" y="18" font-size="13">Audio-token to text-token similarity</text>')
    for row, label in enumerate(labels):
        y = top + (len(labels) - row - 1) * cell_h
        parts.append(f'<text x="{left - 8}" y="{y + 16}" text-anchor="end" font-size="10">{escape(label)}</text>')
        for col, value in enumerate(sampled[row]):
            x = left + col * cell_w
            parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_w + 0.05:.2f}" height="{cell_h:.2f}" '
                f'fill="{_heat_color(float(value))}"/>'
            )

    valid = targets[frame_indices] != -100
    if valid.any():
        parts.append(_polyline_for_labels(frame_indices[valid], targets[frame_indices][valid], left, top, cell_w, cell_h, len(labels), "black", 1.2))
    parts.append(_polyline_for_labels(frame_indices, predictions[frame_indices], left, top, cell_w, cell_h, len(labels), "#f5d000", 0.9))
    for boundary in np.flatnonzero(_boundary_mask(targets, ignore_index=-100)) + 1:
        col = int(np.searchsorted(frame_indices, boundary))
        x = left + col * cell_w
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="white" stroke-width="0.45" opacity="0.65"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _save_audio_audio_svg(
    path: Path,
    matrix: np.ndarray,
    source_indices: np.ndarray,
    targets: np.ndarray,
) -> None:
    indices = _sample_indices(len(matrix), max_count=180)
    sampled = matrix[np.ix_(indices, indices)]
    sampled_source_indices = source_indices[indices]
    cell = 3
    left = 36
    top = 30
    plot_size = cell * len(indices)
    width = left + plot_size + 20
    height = top + plot_size + 20
    parts = [_svg_header(width, height)]
    parts.append(f'<text x="{left}" y="18" font-size="13">Audio-token self-similarity</text>')
    for row in range(len(indices)):
        y = top + (len(indices) - row - 1) * cell
        for col, value in enumerate(sampled[row]):
            x = left + col * cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{_heat_color(float(value))}"/>')
    for boundary in np.flatnonzero(_boundary_mask(targets, ignore_index=-100)) + 1:
        pos = int(np.searchsorted(sampled_source_indices, boundary))
        if 0 <= pos < len(indices):
            x = left + pos * cell
            y = top + (len(indices) - pos - 1) * cell
            parts.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="{top + plot_size}" stroke="white" stroke-width="0.5" opacity="0.55"/>')
            parts.append(f'<line x1="{left}" y1="{y}" x2="{left + plot_size}" y2="{y}" stroke="white" stroke-width="0.5" opacity="0.55"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _svg_header(width: int | float, height: int | float) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(width)}" height="{int(height)}" '
        'font-family="Arial, sans-serif" shape-rendering="crispEdges">'
        '<rect width="100%" height="100%" fill="white"/>'
    )


def _polyline_for_labels(
    frame_indices: np.ndarray,
    label_indices: np.ndarray,
    left: float,
    top: float,
    cell_w: float,
    cell_h: float,
    num_labels: int,
    color: str,
    stroke_width: float,
) -> str:
    points = []
    for frame_index, label_index in zip(frame_indices, label_indices):
        x = left + np.searchsorted(frame_indices, frame_index) * cell_w + cell_w / 2
        y = top + (num_labels - int(label_index) - 0.5) * cell_h
        points.append(f"{x:.2f},{y:.2f}")
    return (
        f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" '
        f'stroke-width="{stroke_width}" opacity="0.9"/>'
    )


def _heat_color(value: float) -> str:
    value = max(-1.0, min(1.0, value))
    if value < 0:
        amount = value + 1.0
        red = int(55 + 200 * amount)
        green = int(100 + 155 * amount)
        blue = 255
    else:
        amount = 1.0 - value
        red = 255
        green = int(70 + 185 * amount)
        blue = int(70 + 185 * amount)
    return f"#{red:02x}{green:02x}{blue:02x}"


def _summarize_corpus(results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    keys = [
        "frame_accuracy",
        "text_text_offdiag_mean",
        "text_text_offdiag_max",
        "audio_text_gt_score_mean",
        "audio_text_gt_margin_mean",
        "audio_audio_same_label_mean",
        "audio_audio_different_label_mean",
        "audio_text_change_at_boundary_mean",
        "audio_text_change_non_boundary_mean",
    ]
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        values = np.asarray([result[key] for result in results], dtype=float)
        values = values[np.isfinite(values)]
        summary[key] = {
            "mean": _safe_mean(values),
            "std": float(values.std()) if len(values) else float("nan"),
        }
    return summary


def _cosine_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = _normalize_rows(left)
    right = _normalize_rows(right)
    return (left @ right.T).astype(np.float32)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def _sample_indices(length: int, max_count: int) -> np.ndarray:
    if length <= max_count:
        return np.arange(length, dtype=np.int64)
    return np.linspace(0, length - 1, num=max_count).round().astype(np.int64)


def _boundary_mask(targets: np.ndarray, ignore_index: int) -> np.ndarray:
    valid_pairs = (targets[:-1] != ignore_index) & (targets[1:] != ignore_index)
    return valid_pairs & (targets[:-1] != targets[1:])


def _non_boundary_mask(targets: np.ndarray, ignore_index: int) -> np.ndarray:
    valid_pairs = (targets[:-1] != ignore_index) & (targets[1:] != ignore_index)
    return valid_pairs & (targets[:-1] == targets[1:])


def _safe_mean(values: np.ndarray) -> float:
    return float(values.mean()) if len(values) else float("nan")


def _safe_max(values: np.ndarray) -> float:
    return float(values.max()) if len(values) else float("nan")


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
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install torch to run diagnostics.") from exc
    return torch
