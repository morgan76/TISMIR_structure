from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tismir.decoding.jams import save_segments_jams
from tismir.decoding.segments import (
    decode_label_indices,
    merge_frame_labels,
    remove_short_segments,
    smooth_logits,
)
from tismir.models import build_model
from tismir.training.data import StructureEmbeddingDataset


def run_baseline_inference(
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
    smoothing_window: int = 1,
    smoothing_mode: str = "mean",
    decoder: str = "argmax",
    transition_penalty: float = 0.0,
    min_segment_duration: float = 0.0,
) -> list[dict[str, Any]]:
    """Run baseline inference over a manifest and save predictions."""

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
            logits = model(audio, text)[0].detach().cpu().numpy()

        decoded_logits = smooth_logits(logits, window=smoothing_window, mode=smoothing_mode)
        label_indices = decode_label_indices(
            decoded_logits,
            strategy=decoder,
            transition_penalty=transition_penalty,
        )
        frame_labels = [example.labels[int(label_index)] for label_index in label_indices]
        segments = merge_frame_labels(example.beat_intervals, frame_labels)
        segments = remove_short_segments(segments, min_duration=min_segment_duration)
        duration = example.beat_intervals[-1][1] if example.beat_intervals else None

        track_dir = output_dir / example.dataset
        track_dir.mkdir(parents=True, exist_ok=True)
        jams_path = track_dir / f"{example.track_id}.jams"
        json_path = track_dir / f"{example.track_id}.json"
        save_segments_jams(jams_path, segments, duration=duration, namespace=namespace)
        _save_prediction_json(
            json_path,
            example,
            segments,
            logits,
            decoded_logits,
            {
                "candidate_label_strategy": candidate_label_strategy,
                "annotation_processing": annotation_processing,
                "smoothing_window": smoothing_window,
                "smoothing_mode": smoothing_mode,
                "decoder": decoder,
                "transition_penalty": transition_penalty,
                "min_segment_duration": min_segment_duration,
            },
            label_indices,
        )

        result = {
            "track_id": example.track_id,
            "dataset": example.dataset,
            "num_frames": len(example.audio),
            "num_segments": len(segments),
            "jams_path": str(jams_path),
            "json_path": str(json_path),
        }
        results.append(result)
        print(f"{example.track_id}: frames={result['num_frames']} segments={result['num_segments']}")

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
        handle.write("\n")
    return results


def _save_prediction_json(
    path: Path,
    example,
    segments,
    raw_logits: np.ndarray,
    decoded_logits: np.ndarray,
    decoding: dict[str, Any],
    label_indices: np.ndarray,
) -> None:
    probabilities = _softmax(decoded_logits, axis=-1)
    payload = {
        "track_id": example.track_id,
        "dataset": example.dataset,
        "labels": example.labels,
        "decoding": decoding,
        "segments": [
            {"start": start, "end": end, "label": label}
            for start, end, label in segments
        ],
        "frame_label_indices": label_indices.astype(int).tolist(),
        "frame_confidence": probabilities.max(axis=-1).astype(float).tolist(),
        "raw_frame_label_indices": raw_logits.argmax(axis=-1).astype(int).tolist(),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _softmax(values: np.ndarray, axis: int) -> np.ndarray:
    values = values - values.max(axis=axis, keepdims=True)
    exp = np.exp(values)
    return exp / exp.sum(axis=axis, keepdims=True)


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
        raise ImportError("Install torch to run inference.") from exc
    return torch
