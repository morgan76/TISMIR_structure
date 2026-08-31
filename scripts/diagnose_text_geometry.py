#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from tismir.data.annotations import label_base
from tismir.data.jams import load_processed_structure_sections, unique_labels
from tismir.data.manifest import load_manifest
from tismir.encoders.text import text_encoders
from tismir.preprocessing.text import MUSIC_STRUCTURE_DESCRIPTIONS


EXTRA_DESCRIPTIONS = {
    "ending": "the closing section at the end of the song",
    "nothing": "a silent, empty, or non-musical region",
    "post chorus": "a section following the chorus, often extending or resolving the hook",
    "pre chorus": "a build-up section before the chorus",
}


@dataclass(frozen=True)
class LabelEntry:
    key: str
    label: str
    base: str
    prompt: str
    source_labels: tuple[str, ...]


@dataclass(frozen=True)
class Condition:
    name: str
    description: str
    builder: Callable[[list[str]], list[LabelEntry]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose pretrained sentence-encoder geometry for music-structure labels."
    )
    parser.add_argument("--manifest", default="data/manifests/rwc_pop.local.jsonl")
    parser.add_argument("--namespace", default="segment_open")
    parser.add_argument("--output-dir", default="outputs/text_geometry/rwc_pop_prompting")
    parser.add_argument("--text-encoder", default="sentence_transformers")
    parser.add_argument("--checkpoint", default="intfloat/e5-base-v2")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--prompt-prefix",
        default="",
        help="Optional string prepended to every text prompt before encoding.",
    )
    parser.add_argument("--annotation-policy", default=None)
    parser.add_argument(
        "--condition-set",
        choices=["prompting", "root_real_vs_corpus_ids"],
        default="prompting",
    )
    parser.add_argument(
        "--section-id-mapping",
        default="configs/annotation_mappings/rwc_pop_section_ids_corpus_base_labels_seed0.json",
        help="Corpus section-ID mapping JSON used by --condition-set root_real_vs_corpus_ids.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotation_processing = (
        None if args.annotation_policy is None else {"policy": args.annotation_policy}
    )

    raw_labels, label_counts = _collect_labels(
        manifest=args.manifest,
        namespace=args.namespace,
        annotation_processing=annotation_processing,
    )
    conditions = _conditions(
        condition_set=args.condition_set,
        section_id_mapping_path=Path(args.section_id_mapping),
    )
    encoder = text_encoders.build(
        args.text_encoder,
        checkpoint=args.checkpoint,
        device=args.device,
        normalize_embeddings=True,
        batch_size=args.batch_size,
    )

    results = []
    embeddings_by_condition: dict[str, np.ndarray] = {}
    entries_by_condition: dict[str, list[LabelEntry]] = {}
    for condition in conditions:
        entries = condition.builder(raw_labels)
        if args.prompt_prefix:
            entries = _with_prompt_prefix(entries, args.prompt_prefix)
        prompts = [entry.prompt for entry in entries]
        embeddings = encoder.encode(prompts)
        embeddings = _l2_normalize(embeddings)
        entries_by_condition[condition.name] = entries
        embeddings_by_condition[condition.name] = embeddings

        condition_dir = output_dir / condition.name
        condition_dir.mkdir(parents=True, exist_ok=True)
        _write_condition_labels(condition_dir / "labels.csv", entries, label_counts)
        _write_similarity_matrix(condition_dir / "cosine_similarity.csv", entries, embeddings)
        _plot_similarity_matrix(condition_dir / "cosine_similarity.png", entries, embeddings)
        _plot_pca(condition_dir / "pca.png", entries, embeddings)
        _write_nearest_neighbors(condition_dir / "nearest_neighbors.csv", entries, embeddings)
        results.append(_condition_summary(condition, entries, embeddings))

    _write_summary_csv(output_dir / "summary_metrics.csv", results)
    _plot_summary_metric_heatmap(output_dir / "summary_metric_heatmap.png", results)
    _write_alignment_csv(
        output_dir / "cross_condition_alignment.csv",
        entries_by_condition=entries_by_condition,
        embeddings_by_condition=embeddings_by_condition,
    )
    _plot_pairwise_histograms(
        output_dir / "pairwise_similarity_histograms.png",
        entries_by_condition=entries_by_condition,
        embeddings_by_condition=embeddings_by_condition,
    )
    _write_report(
        output_dir / "report.md",
        manifest=args.manifest,
        namespace=args.namespace,
        raw_labels=raw_labels,
        label_counts=label_counts,
        conditions=conditions,
        summaries=results,
        entries_by_condition=entries_by_condition,
        embeddings_by_condition=embeddings_by_condition,
    )
    print(f"Saved text-geometry diagnostics to {output_dir}")


def _collect_labels(
    manifest: str,
    namespace: str,
    annotation_processing,
) -> tuple[list[str], Counter[str]]:
    counts: Counter[str] = Counter()
    labels: list[str] = []
    seen: set[str] = set()
    for track in load_manifest(manifest):
        sections = load_processed_structure_sections(
            track.jams_path,
            namespace=namespace,
            annotation_processing=annotation_processing,
        )
        for section in sections:
            counts[section.label] += 1
        for label in unique_labels(sections):
            if label not in seen:
                labels.append(label)
                seen.add(label)
    return labels, counts


def _conditions(condition_set: str, section_id_mapping_path: Path) -> list[Condition]:
    if condition_set == "prompting":
        return _prompting_conditions()
    if condition_set == "root_real_vs_corpus_ids":
        mapping = _load_section_id_mapping(section_id_mapping_path)
        return [
            Condition(
                name="root_real",
                description="Root section labels encoded directly.",
                builder=lambda labels: _entries_from_labels(
                    labels,
                    label_transform=label_base,
                    key_transform=label_base,
                    prompt_builder=_bare_prompt,
                ),
            ),
            Condition(
                name="root_real_definition",
                description="Root section labels phrased with an explicit musical definition.",
                builder=lambda labels: _entries_from_labels(
                    labels,
                    label_transform=label_base,
                    key_transform=label_base,
                    prompt_builder=_base_definition_prompt,
                ),
            ),
            Condition(
                name="root_real_compact_definition",
                description="Root section labels phrased as compact section type and musical role.",
                builder=lambda labels: _entries_from_labels(
                    labels,
                    label_transform=label_base,
                    key_transform=label_base,
                    prompt_builder=_compact_definition_prompt,
                ),
            ),
            Condition(
                name="root_real_music_caption",
                description=(
                    "Root section labels phrased as short music-caption-style "
                    "descriptions."
                ),
                builder=lambda labels: _entries_from_labels(
                    labels,
                    label_transform=label_base,
                    key_transform=label_base,
                    prompt_builder=_music_caption_prompt,
                ),
            ),
            Condition(
                name="root_corpus_ids",
                description=(
                    "Root section labels replaced by a fixed corpus-level anonymous "
                    "section-ID codebook."
                ),
                builder=lambda labels: _entries_from_corpus_section_ids(labels, mapping),
            ),
        ]
    raise ValueError(f"Unknown condition set: {condition_set}")


def _prompting_conditions() -> list[Condition]:
    return [
        Condition(
            name="original_bare",
            description="Raw corpus labels encoded directly.",
            builder=lambda labels: _entries_from_labels(labels, prompt_builder=_bare_prompt),
        ),
        Condition(
            name="base_bare",
            description="Occurrence/variant markers collapsed to base section types.",
            builder=lambda labels: _entries_from_labels(
                labels,
                label_transform=lambda label: _split_base_marker(label)[0],
                key_transform=lambda label: _split_base_marker(label)[0],
                prompt_builder=_bare_prompt,
            ),
        ),
        Condition(
            name="original_definition",
            description="Raw labels with explicit base type and musical definition.",
            builder=lambda labels: _entries_from_labels(labels, prompt_builder=_definition_prompt),
        ),
        Condition(
            name="base_definition",
            description="Base section types with explicit musical definition.",
            builder=lambda labels: _entries_from_labels(
                labels,
                label_transform=lambda label: _split_base_marker(label)[0],
                key_transform=lambda label: _split_base_marker(label)[0],
                prompt_builder=_base_definition_prompt,
            ),
        ),
        Condition(
            name="base_compact_definition",
            description="Base section types with concise structured musical definition.",
            builder=lambda labels: _entries_from_labels(
                labels,
                label_transform=lambda label: _split_base_marker(label)[0],
                key_transform=lambda label: _split_base_marker(label)[0],
                prompt_builder=_compact_definition_prompt,
            ),
        ),
        Condition(
            name="original_occurrence_definition",
            description="Raw labels with base definition plus occurrence/variant marker wording.",
            builder=lambda labels: _entries_from_labels(
                labels,
                prompt_builder=_occurrence_definition_prompt,
            ),
        ),
        Condition(
            name="original_compact_occurrence_definition",
            description="Raw labels with concise structured base type, marker, and musical definition.",
            builder=lambda labels: _entries_from_labels(
                labels,
                prompt_builder=_compact_occurrence_definition_prompt,
            ),
        ),
    ]


def _load_section_id_mapping(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping = payload.get("mapping", payload)
    if not isinstance(mapping, dict):
        raise ValueError(f"Invalid section-ID mapping payload: {path}")
    return {str(key): str(value) for key, value in mapping.items()}


def _entries_from_labels(
    labels: list[str],
    label_transform: Callable[[str], str] | None = None,
    key_transform: Callable[[str], str] | None = None,
    prompt_builder: Callable[[str], str] = lambda label: label,
) -> list[LabelEntry]:
    groups: dict[str, list[str]] = {}
    order: list[str] = []
    for raw_label in labels:
        label = label_transform(raw_label) if label_transform is not None else raw_label
        key = key_transform(raw_label) if key_transform is not None else label
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(raw_label)

    entries = []
    for key in order:
        source_labels = tuple(groups[key])
        label = label_transform(source_labels[0]) if label_transform is not None else source_labels[0]
        base, _ = _split_base_marker(label)
        entries.append(
            LabelEntry(
                key=key,
                label=label,
                base=base,
                prompt=prompt_builder(label),
                source_labels=source_labels,
            )
        )
    return entries


def _with_prompt_prefix(entries: list[LabelEntry], prefix: str) -> list[LabelEntry]:
    return [
        LabelEntry(
            key=entry.key,
            label=entry.label,
            base=entry.base,
            prompt=f"{prefix}{entry.prompt}",
            source_labels=entry.source_labels,
        )
        for entry in entries
    ]


def _entries_from_corpus_section_ids(
    labels: list[str],
    mapping: dict[str, str],
) -> list[LabelEntry]:
    groups: dict[str, list[str]] = {}
    order: list[str] = []
    for raw_label in labels:
        root = label_base(raw_label)
        if root not in groups:
            groups[root] = []
            order.append(root)
        groups[root].append(raw_label)

    entries = []
    for root in order:
        label = mapping.get(root, root)
        entries.append(
            LabelEntry(
                key=root,
                label=label,
                base=root,
                prompt=_bare_prompt(label),
                source_labels=tuple(groups[root]),
            )
        )
    return entries


def _bare_prompt(label: str) -> str:
    return label


def _definition_prompt(label: str) -> str:
    base, marker = _split_base_marker(label)
    marker_sentence = "" if marker is None else f" Variant or occurrence marker: {marker}."
    return (
        f"Music structure label: {_readable_label(label)}. "
        f"Base section type: {base}.{marker_sentence} "
        f"Meaning: {_definition(base)}."
    )


def _base_definition_prompt(label: str) -> str:
    base, _ = _split_base_marker(label)
    return f"Music structure label: {base}. Meaning: {_definition(base)}."


def _compact_definition_prompt(label: str) -> str:
    base, _ = _split_base_marker(label)
    return f"section type: {base}; musical role: {_definition(base)}"


def _music_caption_prompt(label: str) -> str:
    base, _ = _split_base_marker(label)
    return f"a {base} section in a pop song, {_definition(base)}"


def _occurrence_definition_prompt(label: str) -> str:
    base, marker = _split_base_marker(label)
    if marker is None:
        occurrence = f"the {base} section"
    elif marker.isalpha() and len(marker) == 1:
        occurrence = f"a distinct {base} variant marked {marker.upper()}"
    else:
        occurrence = f"a distinct occurrence of the {base} section marked {marker}"
    return (
        f"Music structure label: {_readable_label(label)}. "
        f"Meaning: {occurrence}. Base type meaning: {_definition(base)}. "
        "Use this label for frames belonging to this section."
    )


def _compact_occurrence_definition_prompt(label: str) -> str:
    base, marker = _split_base_marker(label)
    if marker is None:
        return f"section type: {base}; musical role: {_definition(base)}"
    return (
        f"section type: {base}; occurrence marker: {marker.upper()}; "
        f"musical role: {_definition(base)}"
    )


def _definition(base: str) -> str:
    return EXTRA_DESCRIPTIONS.get(
        base,
        MUSIC_STRUCTURE_DESCRIPTIONS.get(
            base,
            f"a music structure section annotated as {base}",
        ),
    ).rstrip(".")


def _readable_label(label: str) -> str:
    text = re.sub(r"[_\-]+", " ", label.strip())
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = re.sub(r"(?<=[a-zA-Z])(?=\d)", " ", text)
    text = re.sub(r"(?<=\d)(?=[a-zA-Z])", " ", text)
    return " ".join(text.split())


def _split_base_marker(label: str) -> tuple[str, str | None]:
    readable = _readable_label(label).lower()
    if readable in {"silence", "nothing"}:
        return readable, None
    match = re.fullmatch(r"(.+?)\s+([a-z]|[0-9]+[a-z]?)", readable)
    if match is None:
        return readable, None
    base, marker = match.groups()
    if base in {"verse", "chorus", "bridge", "section", "part", "theme"}:
        return base, marker
    return readable, None


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def _similarity(embeddings: np.ndarray) -> np.ndarray:
    return embeddings @ embeddings.T


def _pair_masks(entries: list[LabelEntry]) -> tuple[np.ndarray, np.ndarray]:
    bases = np.asarray([entry.base for entry in entries])
    same = bases[:, None] == bases[None, :]
    off_diagonal = ~np.eye(len(entries), dtype=bool)
    return same & off_diagonal, (~same) & off_diagonal


def _condition_summary(
    condition: Condition,
    entries: list[LabelEntry],
    embeddings: np.ndarray,
) -> dict[str, float | int | str]:
    sim = _similarity(embeddings)
    off_diagonal = sim[~np.eye(len(entries), dtype=bool)]
    same_mask, different_mask = _pair_masks(entries)
    same_values = sim[same_mask]
    different_values = sim[different_mask]
    top1_same = _top1_same_base_accuracy(entries, sim)
    rank = _mean_nearest_same_base_rank(entries, sim)
    _, explained = _pca2(embeddings)
    return {
        "condition": condition.name,
        "description": condition.description,
        "num_labels": len(entries),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "mean_offdiag_cosine": _mean(off_diagonal),
        "std_offdiag_cosine": _std(off_diagonal),
        "same_base_pairs": int(same_mask.sum()),
        "different_base_pairs": int(different_mask.sum()),
        "mean_same_base_cosine": _mean(same_values),
        "mean_different_base_cosine": _mean(different_values),
        "same_minus_different": _mean(same_values) - _mean(different_values),
        "top1_same_base_accuracy": top1_same,
        "mean_nearest_same_base_rank": rank,
        "effective_rank": _effective_rank(embeddings),
        "participation_ratio": _participation_ratio(embeddings),
        "pc1_explained": float(explained[0]) if explained.size > 0 else float("nan"),
        "pc2_explained": float(explained[1]) if explained.size > 1 else float("nan"),
        "centroid_norm": _centroid_norm(embeddings),
        "uniformity_t2": _uniformity(embeddings, t=2.0),
        "base_silhouette_cosine": _base_silhouette(entries, sim),
        "nearest_same_base_margin": _nearest_same_base_margin(entries, sim),
        "top1_neighbor_indegree_gini": _top1_neighbor_indegree_gini(sim),
    }


def _top1_same_base_accuracy(entries: list[LabelEntry], sim: np.ndarray) -> float:
    correct = 0
    total = 0
    for index, entry in enumerate(entries):
        candidates = [
            other
            for other, other_entry in enumerate(entries)
            if other != index and other_entry.base == entry.base
        ]
        if not candidates:
            continue
        nearest = int(np.argsort(sim[index])[::-1][1])
        correct += int(entries[nearest].base == entry.base)
        total += 1
    return float("nan") if total == 0 else correct / total


def _mean_nearest_same_base_rank(entries: list[LabelEntry], sim: np.ndarray) -> float:
    ranks = []
    for index, entry in enumerate(entries):
        order = [int(value) for value in np.argsort(sim[index])[::-1] if int(value) != index]
        same = {other for other, other_entry in enumerate(entries) if other_entry.base == entry.base}
        same.discard(index)
        if not same:
            continue
        for rank, other in enumerate(order, start=1):
            if other in same:
                ranks.append(rank)
                break
    return _mean(np.asarray(ranks, dtype=np.float64))


def _effective_rank(embeddings: np.ndarray) -> float:
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    total = float(singular_values.sum())
    if total <= 0:
        return 0.0
    probabilities = singular_values / total
    entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12))))
    return math.exp(entropy)


def _participation_ratio(embeddings: np.ndarray) -> float:
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    variances = singular_values**2
    total = float(variances.sum())
    squared_total = float(np.square(variances).sum())
    if total <= 0 or squared_total <= 0:
        return 0.0
    return total**2 / squared_total


def _centroid_norm(embeddings: np.ndarray) -> float:
    return float(np.linalg.norm(embeddings.mean(axis=0)))


def _uniformity(embeddings: np.ndarray, t: float) -> float:
    if len(embeddings) < 2:
        return float("nan")
    sim = _similarity(embeddings)
    distances_squared = np.maximum(2.0 - 2.0 * sim, 0.0)
    values = np.exp(-t * distances_squared[~np.eye(len(embeddings), dtype=bool)])
    return float(np.log(np.maximum(values.mean(), 1e-12)))


def _base_silhouette(entries: list[LabelEntry], sim: np.ndarray) -> float:
    bases = np.asarray([entry.base for entry in entries])
    counts = Counter(entry.base for entry in entries)
    if sum(count > 1 for count in counts.values()) == 0 or len(counts) < 2:
        return float("nan")

    distances = 1.0 - sim
    scores = []
    for index, base in enumerate(bases):
        same_mask = bases == base
        same_mask[index] = False
        if not same_mask.any():
            continue
        other_distances = [
            float(distances[index, bases == other_base].mean())
            for other_base in counts
            if other_base != base
        ]
        if not other_distances:
            continue
        a_value = float(distances[index, same_mask].mean())
        b_value = min(other_distances)
        scores.append((b_value - a_value) / max(a_value, b_value, 1e-12))
    return _mean(np.asarray(scores, dtype=np.float64))


def _nearest_same_base_margin(entries: list[LabelEntry], sim: np.ndarray) -> float:
    margins = []
    bases = np.asarray([entry.base for entry in entries])
    for index, base in enumerate(bases):
        same_mask = bases == base
        same_mask[index] = False
        different_mask = bases != base
        if not same_mask.any() or not different_mask.any():
            continue
        best_same = float(sim[index, same_mask].max())
        best_different = float(sim[index, different_mask].max())
        margins.append(best_same - best_different)
    return _mean(np.asarray(margins, dtype=np.float64))


def _top1_neighbor_indegree_gini(sim: np.ndarray) -> float:
    if len(sim) < 2:
        return float("nan")
    top1 = []
    for index in range(len(sim)):
        order = [int(value) for value in np.argsort(sim[index])[::-1] if int(value) != index]
        top1.append(order[0])
    counts = np.bincount(np.asarray(top1), minlength=len(sim)).astype(np.float64)
    return _gini(counts)


def _gini(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    sorted_values = np.sort(np.maximum(values, 0.0))
    total = float(sorted_values.sum())
    if total <= 0:
        return 0.0
    indices = np.arange(1, len(sorted_values) + 1, dtype=np.float64)
    return float((2.0 * np.sum(indices * sorted_values)) / (len(sorted_values) * total) - (len(sorted_values) + 1.0) / len(sorted_values))


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else float("nan")


def _std(values: np.ndarray) -> float:
    return float(np.std(values)) if values.size else float("nan")


def _write_condition_labels(path: Path, entries: list[LabelEntry], counts: Counter[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["label", "base", "prompt", "source_labels", "source_segment_count"],
        )
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "label": entry.label,
                    "base": entry.base,
                    "prompt": entry.prompt,
                    "source_labels": " | ".join(entry.source_labels),
                    "source_segment_count": sum(counts[label] for label in entry.source_labels),
                }
            )


def _write_similarity_matrix(path: Path, entries: list[LabelEntry], embeddings: np.ndarray) -> None:
    sim = _similarity(embeddings)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", *[entry.label for entry in entries]])
        for entry, row in zip(entries, sim):
            writer.writerow([entry.label, *[f"{value:.8f}" for value in row]])


def _write_nearest_neighbors(path: Path, entries: list[LabelEntry], embeddings: np.ndarray) -> None:
    sim = _similarity(embeddings)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "label",
                "base",
                "rank",
                "neighbor",
                "neighbor_base",
                "cosine",
                "same_base",
            ],
        )
        writer.writeheader()
        for index, entry in enumerate(entries):
            order = [int(value) for value in np.argsort(sim[index])[::-1] if int(value) != index]
            for rank, neighbor in enumerate(order[:5], start=1):
                neighbor_entry = entries[neighbor]
                writer.writerow(
                    {
                        "label": entry.label,
                        "base": entry.base,
                        "rank": rank,
                        "neighbor": neighbor_entry.label,
                        "neighbor_base": neighbor_entry.base,
                        "cosine": f"{sim[index, neighbor]:.8f}",
                        "same_base": entry.base == neighbor_entry.base,
                    }
                )


def _write_summary_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    fields = [
        "condition",
        "num_labels",
        "embedding_dim",
        "mean_offdiag_cosine",
        "std_offdiag_cosine",
        "same_base_pairs",
        "different_base_pairs",
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
        "description",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_alignment_csv(
    path: Path,
    entries_by_condition: dict[str, list[LabelEntry]],
    embeddings_by_condition: dict[str, np.ndarray],
) -> None:
    rows = []
    names = list(entries_by_condition)
    for left_index, left_name in enumerate(names):
        left_entries = entries_by_condition[left_name]
        left_lookup = {entry.key: index for index, entry in enumerate(left_entries)}
        for right_name in names[left_index + 1 :]:
            right_entries = entries_by_condition[right_name]
            right_lookup = {entry.key: index for index, entry in enumerate(right_entries)}
            keys = [key for key in left_lookup if key in right_lookup]
            if not keys:
                continue
            left = embeddings_by_condition[left_name]
            right = embeddings_by_condition[right_name]
            values = np.asarray(
                [left[left_lookup[key]] @ right[right_lookup[key]] for key in keys],
                dtype=np.float64,
            )
            left_common = left[[left_lookup[key] for key in keys]]
            right_common = right[[right_lookup[key] for key in keys]]
            left_sim = _similarity(left_common)
            right_sim = _similarity(right_common)
            rows.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "common_labels": len(keys),
                    "mean_same_label_cosine": _mean(values),
                    "std_same_label_cosine": _std(values),
                    "min_same_label_cosine": float(values.min()),
                    "max_same_label_cosine": float(values.max()),
                    "pairwise_similarity_pearson": _upper_triangle_pearson(left_sim, right_sim),
                    "top3_neighbor_jaccard": _mean_topk_neighbor_jaccard(left_sim, right_sim, k=3),
                }
            )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "left",
                "right",
                "common_labels",
                "mean_same_label_cosine",
                "std_same_label_cosine",
                "min_same_label_cosine",
                "max_same_label_cosine",
                "pairwise_similarity_pearson",
                "top3_neighbor_jaccard",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _upper_triangle_pearson(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3:
        return float("nan")
    mask = np.triu(np.ones_like(left, dtype=bool), k=1)
    left_values = left[mask]
    right_values = right[mask]
    if _std(left_values) <= 0 or _std(right_values) <= 0:
        return float("nan")
    return float(np.corrcoef(left_values, right_values)[0, 1])


def _mean_topk_neighbor_jaccard(left: np.ndarray, right: np.ndarray, k: int) -> float:
    if len(left) < 2:
        return float("nan")
    k = min(k, len(left) - 1)
    scores = []
    for index in range(len(left)):
        left_neighbors = [
            int(value)
            for value in np.argsort(left[index])[::-1]
            if int(value) != index
        ]
        right_neighbors = [
            int(value)
            for value in np.argsort(right[index])[::-1]
            if int(value) != index
        ]
        left_top = set(left_neighbors[:k])
        right_top = set(right_neighbors[:k])
        scores.append(len(left_top & right_top) / max(len(left_top | right_top), 1))
    return _mean(np.asarray(scores, dtype=np.float64))


def _plot_similarity_matrix(path: Path, entries: list[LabelEntry], embeddings: np.ndarray) -> None:
    plt = _matplotlib()
    sim = _similarity(embeddings)
    fig, ax = plt.subplots(figsize=(max(7.0, len(entries) * 0.45), max(6.0, len(entries) * 0.4)))
    image = ax.imshow(sim, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    labels = [entry.label for entry in entries]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Sentence-Encoder Cosine Similarity")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_pca(path: Path, entries: list[LabelEntry], embeddings: np.ndarray) -> None:
    plt = _matplotlib()
    coords, explained = _pca2(embeddings)
    bases = list(dict.fromkeys(entry.base for entry in entries))
    colors = {base: plt.cm.tab20(index % 20) for index, base in enumerate(bases)}
    fig, ax = plt.subplots(figsize=(9, 7))
    for entry, xy in zip(entries, coords):
        ax.scatter(xy[0], xy[1], color=colors[entry.base], s=48)
        ax.text(xy[0], xy[1], entry.label, fontsize=8, ha="left", va="bottom")
    ax.axhline(0, color="#999999", linewidth=0.8, alpha=0.4)
    ax.axvline(0, color="#999999", linewidth=0.8, alpha=0.4)
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
    ax.set_title("Text Embedding PCA")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_pairwise_histograms(
    path: Path,
    entries_by_condition: dict[str, list[LabelEntry]],
    embeddings_by_condition: dict[str, np.ndarray],
) -> None:
    plt = _matplotlib()
    names = list(entries_by_condition)
    fig, axes = plt.subplots(len(names), 1, figsize=(9, max(3, 2.2 * len(names))), sharex=True)
    if len(names) == 1:
        axes = [axes]
    bins = np.linspace(-0.1, 1.0, 35)
    for ax, name in zip(axes, names):
        entries = entries_by_condition[name]
        sim = _similarity(embeddings_by_condition[name])
        same_mask, different_mask = _pair_masks(entries)
        same_values = sim[same_mask]
        different_values = sim[different_mask]
        ax.hist(different_values, bins=bins, alpha=0.65, label="different base", color="#4E79A7")
        if same_values.size:
            ax.hist(same_values, bins=bins, alpha=0.65, label="same base", color="#F28E2B")
        ax.set_ylabel(name)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    axes[-1].set_xlabel("cosine similarity")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_summary_metric_heatmap(
    path: Path,
    rows: list[dict[str, float | int | str]],
) -> None:
    plt = _matplotlib()
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
    conditions = [str(row["condition"]) for row in rows]
    raw = np.asarray(
        [
            [float(row[metric]) if not isinstance(row[metric], str) else float("nan") for metric in metrics]
            for row in rows
        ],
        dtype=np.float64,
    )
    normalized = np.zeros_like(raw)
    for col in range(raw.shape[1]):
        values = raw[:, col]
        valid = np.isfinite(values)
        if not valid.any():
            normalized[:, col] = np.nan
            continue
        minimum = float(values[valid].min())
        maximum = float(values[valid].max())
        if maximum <= minimum:
            normalized[valid, col] = 0.5
        else:
            normalized[valid, col] = (values[valid] - minimum) / (maximum - minimum)
        normalized[~valid, col] = np.nan

    fig, ax = plt.subplots(figsize=(14, max(4.5, 0.55 * len(conditions))))
    image = ax.imshow(normalized, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    image.cmap.set_bad("#f2f2f2")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([_short_metric_name(metric) for metric in metrics], rotation=35, ha="right")
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels(conditions)
    ax.set_title("Text Embedding Geometry Metrics")
    for row_index in range(raw.shape[0]):
        for col_index in range(raw.shape[1]):
            value = raw[row_index, col_index]
            label = "n/a" if not np.isfinite(value) else f"{value:.3f}"
            ax.text(col_index, row_index, label, ha="center", va="center", fontsize=8, color="white")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _short_metric_name(metric: str) -> str:
    return {
        "mean_offdiag_cosine": "mean cos",
        "same_minus_different": "same-diff",
        "effective_rank": "eff rank",
        "participation_ratio": "part. ratio",
        "pc1_explained": "PC1",
        "centroid_norm": "centroid",
        "uniformity_t2": "uniformity",
        "base_silhouette_cosine": "silhouette",
        "nearest_same_base_margin": "NN margin",
        "top1_neighbor_indegree_gini": "hub gini",
    }[metric]


def _pca2(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    variances = singular_values**2
    total = variances.sum()
    explained = variances[:2] / max(total, 1e-12)
    return coords, explained


def _write_report(
    path: Path,
    manifest: str,
    namespace: str,
    raw_labels: list[str],
    label_counts: Counter[str],
    conditions: list[Condition],
    summaries: list[dict[str, float | int | str]],
    entries_by_condition: dict[str, list[LabelEntry]],
    embeddings_by_condition: dict[str, np.ndarray],
) -> None:
    lines = [
        "# Text Geometry Diagnostics",
        "",
        f"- manifest: `{manifest}`",
        f"- namespace: `{namespace}`",
        f"- raw labels: {len(raw_labels)}",
        f"- raw segments counted: {sum(label_counts.values())}",
        "",
        "## Conditions",
        "",
    ]
    for condition in conditions:
        entries = entries_by_condition[condition.name]
        lines.append(f"- `{condition.name}`: {condition.description}")
        if entries:
            lines.append(f"  - example prompt: `{entries[0].prompt}`")
    lines.extend(["", "## Summary", ""])
    lines.append(
        "| condition | labels | mean cos | same base | diff base | separation | top1 same-base | same-base rank | eff rank |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in summaries:
        lines.append(
            "| {condition} | {num_labels} | {mean_offdiag_cosine:.3f} | "
            "{mean_same_base_cosine:.3f} | {mean_different_base_cosine:.3f} | "
            "{same_minus_different:.3f} | {top1_same_base_accuracy:.3f} | "
            "{mean_nearest_same_base_rank:.2f} | {effective_rank:.2f} |".format(**row)
        )

    lines.extend(["", "## Distribution Diagnostics", ""])
    lines.append(
        "| condition | centroid norm | uniformity | silhouette | NN margin | PC1 | part. ratio | hub gini |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in summaries:
        lines.append(
            "| {condition} | {centroid_norm:.3f} | {uniformity_t2:.3f} | "
            "{base_silhouette_cosine:.3f} | {nearest_same_base_margin:.3f} | "
            "{pc1_explained:.3f} | {participation_ratio:.2f} | "
            "{top1_neighbor_indegree_gini:.3f} |".format(**row)
        )

    lines.extend(
        [
            "",
            "Metric notes: lower `centroid norm` means less shared embedding direction; lower `uniformity` means more spread on the unit sphere; higher `silhouette` and `NN margin` mean cleaner same-base grouping; lower `PC1` concentration and higher `participation ratio` mean less low-dimensional collapse; high `hub gini` means a few labels dominate nearest-neighbor lists.",
        ]
    )

    lines.extend(["", "## Nearest Neighbors", ""])
    for name, entries in entries_by_condition.items():
        sim = _similarity(embeddings_by_condition[name])
        lines.append(f"### {name}")
        for index, entry in enumerate(entries):
            nearest = [int(value) for value in np.argsort(sim[index])[::-1] if int(value) != index][:3]
            neighbors = ", ".join(
                f"{entries[neighbor].label} ({sim[index, neighbor]:.3f})"
                for neighbor in nearest
            )
            lines.append(f"- `{entry.label}` -> {neighbors}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


if __name__ == "__main__":
    main()
