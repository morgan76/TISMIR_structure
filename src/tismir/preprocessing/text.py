from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from collections import Counter
from typing import Any
import re

import numpy as np

from tismir.data.annotations import (
    concrete_annotation_processing_choices,
    is_random_annotation_processing,
    process_sections,
)
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
    annotation_processing: str | dict[str, Any] | None = None,
    namespace: str = "segment_open",
    scope: str = "dataset",
) -> list[TextPreprocessingResult]:
    """Encode labels from a manifest at dataset or track scope."""

    if scope not in {"dataset", "track"}:
        raise ValueError("scope must be one of: dataset, track")

    encoder = text_encoders.build(text_encoder_name, **(text_encoder_params or {}))
    prompt = _resolve_prompt_config(prompt)
    label_normalization = {} if label_normalization is None else dict(label_normalization)

    if scope == "track":
        return [
            _encode_label_set(
                labels=_track_labels(track, namespace, annotation_processing),
                dataset=track.dataset,
                output_dir=Path(output_root) / text_encoder_name / track.dataset / track.track_id,
                encoder=encoder,
                encoder_name=text_encoder_name,
                prompt=prompt,
                label_normalization=label_normalization,
                annotation_processing=annotation_processing,
                metadata={"track_id": track.track_id, "scope": scope},
            )
            for track in tracks
        ]

    labels_by_dataset: dict[str, list[str]] = {}
    for track in tracks:
        labels_by_dataset.setdefault(track.dataset, [])
        labels_by_dataset[track.dataset].extend(_track_labels(track, namespace, annotation_processing))

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
                annotation_processing=annotation_processing,
                metadata={"scope": scope},
            )
        )
    return results


def result_to_dict(result: TextPreprocessingResult) -> dict[str, Any]:
    return asdict(result)


def _track_labels(
    track: Track,
    namespace: str,
    annotation_processing: str | dict[str, Any] | None,
) -> list[str]:
    if is_random_annotation_processing(annotation_processing):
        assert isinstance(annotation_processing, dict)
        labels: list[str] = []
        for choice in concrete_annotation_processing_choices(annotation_processing):
            labels.extend(_track_labels(track, namespace, choice))
        return _unique_in_order(labels)

    sections = load_structure_sections(track.jams_path, namespace=namespace)
    sections = process_sections(sections, annotation_processing=annotation_processing)
    return unique_labels(sections)


def _encode_label_set(
    labels: list[str],
    dataset: str,
    output_dir: Path,
    encoder,
    encoder_name: str,
    prompt: dict[str, Any],
    label_normalization: dict[str, Any],
    annotation_processing: str | dict[str, Any] | None,
    metadata: dict[str, Any],
) -> TextPreprocessingResult:
    text_labels = normalize_labels(labels, config=label_normalization)
    prompt_labels = _prompt_labels(labels, text_labels, label_normalization)
    prompts = [
        _format_prompt(
            raw_label=raw_label,
            normalized_label=text_label,
            prompt_label=prompt_label,
            prompt=prompt,
        )
        for raw_label, text_label, prompt_label in zip(labels, text_labels, prompt_labels)
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
            "annotation_processing": annotation_processing,
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


def _format_prompt(
    raw_label: str,
    normalized_label: str,
    prompt_label: str,
    prompt: dict[str, Any],
) -> str:
    template = prompt.get("template", "{label}")
    base_label, occurrence_index = _split_occurrence(normalized_label)
    values = {
        "label": prompt_label,
        "raw_label": raw_label,
        "text_label": normalized_label,
        "base_label": base_label,
        "occurrence": "" if occurrence_index is None else str(occurrence_index),
        "occurrence_index": "" if occurrence_index is None else str(occurrence_index),
        "occurrence_description": _occurrence_description(base_label, occurrence_index),
        "occurrence_prompt": _occurrence_prompt(base_label, occurrence_index),
        "description": _label_description(base_label, prompt),
    }
    text = template.format(**values)
    if prompt.get("normalize_whitespace", True):
        text = " ".join(text.split())
    return text


def _resolve_prompt_config(prompt: dict[str, Any] | None) -> dict[str, Any]:
    config = {} if prompt is None else dict(prompt)
    mode = str(config.get("mode", "bare")).lower()
    if mode not in PROMPT_MODE_TEMPLATES:
        modes = ", ".join(sorted(PROMPT_MODE_TEMPLATES))
        raise ValueError(f"Unknown prompt mode: {mode}. Expected one of: {modes}")

    config["mode"] = mode
    config.setdefault("template", PROMPT_MODE_TEMPLATES[mode])
    config.setdefault("normalize_whitespace", True)
    if mode == "descriptive":
        config.setdefault("description_preset", "music_structure")
    return config


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


def _split_occurrence(label: str) -> tuple[str, int | None]:
    text = _strip_parenthetical(label)
    match = re.fullmatch(r"(.+?)\s+([0-9]+)", text)
    if match is None:
        return text, None
    return match.group(1), int(match.group(2))


def _strip_parenthetical(label: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*$", "", label.strip().lower())


def _occurrence_description(base_label: str, occurrence_index: int | None) -> str:
    if occurrence_index is None:
        return f"{_indefinite_article(base_label)} {base_label} section"
    return f"the {ordinal(occurrence_index)} {base_label} section in chronological order"


def _occurrence_prompt(base_label: str, occurrence_index: int | None) -> str:
    if occurrence_index is None:
        return f"the {base_label} section"
    return (
        f"the {_ordinal_word(occurrence_index)} occurrence of "
        f"the {base_label} section in this song"
    )


def _indefinite_article(text: str) -> str:
    return "an" if text[:1].lower() in {"a", "e", "i", "o", "u"} else "a"


def ordinal(value: int) -> str:
    if 10 <= value % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"{value}{suffix}"


def _ordinal_word(value: int) -> str:
    words = {
        1: "first",
        2: "second",
        3: "third",
        4: "fourth",
        5: "fifth",
        6: "sixth",
        7: "seventh",
        8: "eighth",
        9: "ninth",
        10: "tenth",
        11: "eleventh",
        12: "twelfth",
        13: "thirteenth",
        14: "fourteenth",
        15: "fifteenth",
        16: "sixteenth",
        17: "seventeenth",
        18: "eighteenth",
        19: "nineteenth",
        20: "twentieth",
    }
    return words.get(value, ordinal(value))


def _label_description(base_label: str, prompt: dict[str, Any]) -> str:
    descriptions = _description_map(prompt)
    description = descriptions.get(
        base_label,
        f"a music structure section annotated as {base_label}",
    )
    return description.rstrip(".")


def _description_map(prompt: dict[str, Any]) -> dict[str, str]:
    preset = str(prompt.get("description_preset", "none")).lower()
    descriptions = dict(MUSIC_STRUCTURE_DESCRIPTIONS if preset == "music_structure" else {})
    descriptions.update({str(key): str(value) for key, value in prompt.get("descriptions", {}).items()})
    return descriptions


PROMPT_MODE_TEMPLATES = {
    "bare": "{label}",
    "compact": "Music structure label: {label}",
    "descriptive": (
        "Music structure label: {label}. Base type: {base_label}. "
        "Occurrence: {occurrence_description}. Meaning: {description}. "
        "Use this label for frames belonging to this section."
    ),
    "occurrence_descriptive": (
        "Music structure label: {label}. Meaning: {occurrence_prompt}. "
        "Use this label for frames belonging to {occurrence_prompt}."
    ),
}


MUSIC_STRUCTURE_DESCRIPTIONS = {
    "alternate chorus": "a chorus variant with a changed arrangement or musical role.",
    "big outro": "an expanded closing section at the end of the song.",
    "break": "a short contrasting or reduced-texture section within the song.",
    "bridge": "a contrasting transitional section between other sections.",
    "chorus": "the main repeated hook or refrain section of the song.",
    "chorus half": "a shortened chorus section containing only part of the main refrain.",
    "drum roll": "a short drum-led transition or build-up passage.",
    "fade in": "an opening section where the song gradually increases from silence.",
    "fade out": "a closing section where the song gradually decreases toward silence.",
    "guitar": "a section featuring guitar material.",
    "guitar break": "a short contrasting section led by guitar material.",
    "guitar solo": "a featured instrumental passage led by guitar.",
    "instrumental": "a section without lead vocals or with instrumental focus.",
    "instrumental bridge": "an instrumental contrasting transitional section between other sections.",
    "instrumental chorus": "an instrumental version of the chorus or refrain section.",
    "instrumental intro": "an instrumental opening section before the main body of the song.",
    "instrumental verse": "an instrumental version of a verse-like section.",
    "interlude": "a short connecting passage between larger song sections.",
    "intro": "the opening section before the main body of the song.",
    "intro verse": "an opening section with verse-like material.",
    "main riff": "a section centered on the song's primary repeated instrumental riff.",
    "outro": "the closing section at the end of the song.",
    "post chorus": "a section following the chorus, often extending or resolving the hook.",
    "post post chorus": "a section following a post-chorus, often extending the chorus area further.",
    "pre chorus": "a build-up section before the chorus.",
    "pre verse": "a short section leading into a verse.",
    "quiet chorus": "a softer or reduced version of the chorus section.",
    "rhythmless intro": "an opening section before the main rhythmic groove begins.",
    "silence": "a silent or near-silent region.",
    "slow verse": "a slower or reduced-energy verse-like section.",
    "solo": "a featured instrumental passage, often highlighting one instrument.",
    "transition": "a connecting section that moves between other song sections.",
    "verse": "a recurring lyrical section, usually distinct from the chorus.",
    "vocal intro": "an opening section featuring vocals before the main body of the song.",
    "vocal outro": "a closing section featuring vocals at the end of the song.",
}


def _unique_in_order(labels: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for label in labels:
        if label not in seen:
            unique.append(label)
            seen.add(label)
    return unique
