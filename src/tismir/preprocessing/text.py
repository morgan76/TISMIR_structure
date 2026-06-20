from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from collections import Counter
from typing import Any
import re

import numpy as np

from tismir.data.jams import load_structure_sections, unique_labels
from tismir.data.schemas import Track
from tismir.encoders.text import text_encoders
from tismir.io import save_array, save_json
from tismir.preprocessing.label_normalization import normalize_labels


@dataclass(frozen=True)
class TextPreprocessingResult:
    dataset: str
    output_dir: str
    num_labels: int
    embedding_shape: tuple[int, int]


def preprocess_dataset_text(
    tracks: list[Track],
    output_root: str | Path,
    text_encoder_name: str,
    text_encoder_params: dict[str, Any] | None,
    prompt: dict[str, Any] | None = None,
    label_normalization: dict[str, Any] | None = None,
    namespace: str = "segment_open",
    scope: str = "dataset",
) -> list[TextPreprocessingResult]:
    """Encode labels from a manifest at dataset or track scope."""

    if scope not in {"dataset", "track"}:
        raise ValueError("scope must be one of: dataset, track")

    encoder = text_encoders.build(text_encoder_name, **(text_encoder_params or {}))
    prompt = {} if prompt is None else dict(prompt)
    label_normalization = {} if label_normalization is None else dict(label_normalization)

    if scope == "track":
        return [
            _encode_label_set(
                labels=_track_labels(track, namespace),
                dataset=track.dataset,
                output_dir=Path(output_root) / text_encoder_name / track.dataset / track.track_id,
                encoder=encoder,
                encoder_name=text_encoder_name,
                prompt=prompt,
                label_normalization=label_normalization,
                metadata={"track_id": track.track_id, "scope": scope},
            )
            for track in tracks
        ]

    labels_by_dataset: dict[str, list[str]] = {}
    for track in tracks:
        labels_by_dataset.setdefault(track.dataset, [])
        labels_by_dataset[track.dataset].extend(_track_labels(track, namespace))

    results = []
    for dataset, labels in sorted(labels_by_dataset.items()):
        labels = _unique_in_order(labels)
        results.append(
            _encode_label_set(
                labels=labels,
                dataset=dataset,
                output_dir=Path(output_root) / text_encoder_name / dataset,
                encoder=encoder,
                encoder_name=text_encoder_name,
                prompt=prompt,
                label_normalization=label_normalization,
                metadata={"scope": scope},
            )
        )
    return results


def result_to_dict(result: TextPreprocessingResult) -> dict[str, Any]:
    return asdict(result)


def _track_labels(track: Track, namespace: str) -> list[str]:
    return unique_labels(load_structure_sections(track.jams_path, namespace=namespace))


def _encode_label_set(
    labels: list[str],
    dataset: str,
    output_dir: Path,
    encoder,
    encoder_name: str,
    prompt: dict[str, Any],
    label_normalization: dict[str, Any],
    metadata: dict[str, Any],
) -> TextPreprocessingResult:
    text_labels = normalize_labels(labels, config=label_normalization)
    prompt_labels = _prompt_labels(labels, text_labels, label_normalization)
    prompts = [
        _format_prompt(raw_label=raw_label, text_label=prompt_label, prompt=prompt)
        for raw_label, prompt_label in zip(labels, prompt_labels)
    ]
    embeddings = encoder.encode(prompts)

    save_json(
        output_dir / "labels.json",
        {
            "labels": labels,
            "text_labels": text_labels,
            "prompt_labels": prompt_labels,
            "prompts": prompts,
        },
    )
    save_array(output_dir / "embeddings.npy", embeddings)
    save_json(
        output_dir / "metadata.json",
        {
            "dataset": dataset,
            "text_encoder": {
                "name": encoder_name,
                "checkpoint": getattr(encoder, "checkpoint", None),
                "output_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 and embeddings.shape[0] else getattr(encoder, "output_dim", None),
                "normalize_embeddings": getattr(encoder, "normalize_embeddings", None),
            },
            "prompt": prompt,
            "label_normalization": label_normalization,
            "num_labels": len(labels),
            "embedding_shape": tuple(embeddings.shape),
            **metadata,
        },
    )

    return TextPreprocessingResult(
        dataset=dataset,
        output_dir=str(output_dir),
        num_labels=len(labels),
        embedding_shape=tuple(embeddings.shape),
    )


def _format_prompt(raw_label: str, text_label: str, prompt: dict[str, Any]) -> str:
    template = prompt.get("template", "{label}")
    text = template.format(label=text_label, raw_label=raw_label, text_label=text_label)
    if prompt.get("normalize_whitespace", True):
        text = " ".join(text.split())
    return text


def _prompt_labels(
    raw_labels: list[str],
    text_labels: list[str],
    label_normalization: dict[str, Any],
) -> list[str]:
    if not label_normalization.get("disambiguate_duplicates", False):
        return text_labels

    counts = Counter(text_labels)
    prompt_labels = []
    for raw_label, text_label in zip(raw_labels, text_labels):
        if counts[text_label] <= 1:
            prompt_labels.append(text_label)
            continue
        prompt_labels.append(f"{text_label} ({_readable_raw_label(raw_label)})")
    return prompt_labels


def _readable_raw_label(label: str) -> str:
    text = re.sub(r"[_\-]+", " ", label.strip().lower())
    text = re.sub(r"(?<=[a-z])(?=\d)", " ", text)
    text = re.sub(r"(?<=\d)(?=[a-z])", " ", text)
    return " ".join(text.split())


def _unique_in_order(labels: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for label in labels:
        if label not in seen:
            unique.append(label)
            seen.add(label)
    return unique
