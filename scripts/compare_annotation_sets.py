from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tismir.data.annotations import label_base
from tismir.data.jams import load_structure_sections, sections_to_intervals_labels
from tismir.data.schemas import Section


METRICS = [
    "Precision@0.5",
    "Recall@0.5",
    "F-measure@0.5",
    "Precision@3.0",
    "Recall@3.0",
    "F-measure@3.0",
    "Pairwise F-measure",
    "NCE F-measure",
]


@dataclass(frozen=True)
class TrackPair:
    track_id: str
    original_path: Path
    revised_path: Path


def main() -> None:
    args = _parse_args()
    original_dir = Path(args.original_dir)
    revised_dir = Path(args.revised_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs, missing = _collect_pairs(original_dir, revised_dir)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    if not pairs:
        raise ValueError("No matching JAMS files found")
    file_identity = _file_identity_summary(pairs)

    track_rows: list[dict[str, Any]] = []
    all_original_durations: list[float] = []
    all_revised_durations: list[float] = []
    all_orig_to_rev_distances: list[float] = []
    all_rev_to_orig_distances: list[float] = []
    all_signed_boundary_shifts: list[float] = []
    confusion_seconds: dict[tuple[str, str], float] = defaultdict(float)
    base_confusion_seconds: dict[tuple[str, str], float] = defaultdict(float)
    original_label_seconds: Counter[str] = Counter()
    revised_label_seconds: Counter[str] = Counter()
    original_family_seconds: Counter[str] = Counter()
    revised_family_seconds: Counter[str] = Counter()
    original_label_counts: Counter[str] = Counter()
    revised_label_counts: Counter[str] = Counter()

    for index, pair in enumerate(pairs, start=1):
        if args.progress and (index == 1 or index % 50 == 0 or index == len(pairs)):
            print(f"[{index}/{len(pairs)}] {pair.track_id}", flush=True)
        original = _load_sections(pair.original_path, namespace=args.namespace)
        revised = _load_sections(
            pair.revised_path,
            namespace=args.namespace,
            fallback_end=_annotation_end(original),
        )
        row = _compare_track(pair.track_id, original, revised, trim=args.trim)
        track_rows.append(row)

        original_label_counts.update(section.label for section in original)
        revised_label_counts.update(section.label for section in revised)
        for section in original:
            original_label_seconds[section.label] += section.duration
            original_family_seconds[_label_family(section.label)] += section.duration
            all_original_durations.append(section.duration)
        for section in revised:
            revised_label_seconds[section.label] += section.duration
            revised_family_seconds[_label_family(section.label)] += section.duration
            all_revised_durations.append(section.duration)

        overlap = _overlap_seconds(original, revised)
        for labels, seconds in overlap.items():
            confusion_seconds[labels] += seconds
            base_confusion_seconds[(label_base(labels[0]), label_base(labels[1]))] += seconds

        all_orig_to_rev_distances.extend(row["_orig_to_rev_distances"])
        all_rev_to_orig_distances.extend(row["_rev_to_orig_distances"])
        all_signed_boundary_shifts.extend(row["_signed_boundary_shifts"])
        del row["_orig_to_rev_distances"]
        del row["_rev_to_orig_distances"]
        del row["_signed_boundary_shifts"]

    _write_csv(output_dir / "track_comparison.csv", track_rows)
    _write_label_seconds(output_dir / "label_duration_seconds.csv", original_label_seconds, revised_label_seconds)
    _write_label_seconds(output_dir / "label_family_duration_seconds.csv", original_family_seconds, revised_family_seconds)
    _write_confusion_csv(output_dir / "label_overlap_seconds.csv", confusion_seconds)
    _write_confusion_csv(output_dir / "base_label_overlap_seconds.csv", base_confusion_seconds)

    summary = _build_summary(
        pairs=pairs,
        missing=missing,
        file_identity=file_identity,
        rows=track_rows,
        original_label_seconds=original_label_seconds,
        revised_label_seconds=revised_label_seconds,
        original_label_counts=original_label_counts,
        revised_label_counts=revised_label_counts,
        original_family_seconds=original_family_seconds,
        revised_family_seconds=revised_family_seconds,
        confusion_seconds=confusion_seconds,
        base_confusion_seconds=base_confusion_seconds,
        original_segment_durations=all_original_durations,
        revised_segment_durations=all_revised_durations,
        orig_to_rev_distances=all_orig_to_rev_distances,
        rev_to_orig_distances=all_rev_to_orig_distances,
        signed_boundary_shifts=all_signed_boundary_shifts,
    )
    _write_json(output_dir / "summary.json", summary)
    _write_markdown_report(output_dir / "report.md", summary)
    _make_plots(
        output_dir=output_dir,
        rows=track_rows,
        original_segment_durations=all_original_durations,
        revised_segment_durations=all_revised_durations,
        orig_to_rev_distances=all_orig_to_rev_distances,
        rev_to_orig_distances=all_rev_to_orig_distances,
        signed_boundary_shifts=all_signed_boundary_shifts,
        original_label_seconds=original_label_seconds,
        revised_label_seconds=revised_label_seconds,
        original_family_seconds=original_family_seconds,
        revised_family_seconds=revised_family_seconds,
        confusion_seconds=confusion_seconds,
    )
    print(f"Wrote comparison outputs to {output_dir}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-dir", required=True)
    parser.add_argument("--revised-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--namespace", default="segment_open")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--trim", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _collect_pairs(original_dir: Path, revised_dir: Path) -> tuple[list[TrackPair], dict[str, list[str]]]:
    original = _annotation_files(original_dir)
    revised = _annotation_files(revised_dir)
    shared = sorted(set(original) & set(revised))
    pairs = [
        TrackPair(track_id=stem, original_path=original[stem], revised_path=revised[stem])
        for stem in shared
    ]
    missing = {
        "missing_from_revised": sorted(set(original) - set(revised)),
        "missing_from_original": sorted(set(revised) - set(original)),
    }
    return pairs, missing


def _annotation_files(directory: Path) -> dict[str, Path]:
    paths = []
    for suffix in ("*.jams", "*.txt"):
        paths.extend(directory.glob(suffix))
    return {path.stem: path for path in sorted(paths)}


def _load_sections(
    path: Path,
    namespace: str,
    fallback_end: float | None = None,
) -> list[Section]:
    if path.suffix.lower() == ".jams":
        return load_structure_sections(path, namespace=namespace)
    if path.suffix.lower() == ".txt":
        return _load_txt_sections(path, fallback_end=fallback_end)
    raise ValueError(f"Unsupported annotation file type: {path}")


def _load_txt_sections(path: Path, fallback_end: float | None = None) -> list[Section]:
    entries: list[tuple[float, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Invalid annotation row in {path}:{line_number}: {line!r}")
        try:
            start = float(parts[0])
        except ValueError as exc:
            raise ValueError(f"Invalid time in {path}:{line_number}: {parts[0]!r}") from exc
        entries.append((start, parts[1].strip()))
    if not entries:
        return []

    sections: list[Section] = []
    for index, (start, label) in enumerate(entries):
        normalized_label = label.strip()
        if normalized_label.lower() in {"end", "eof"}:
            continue
        if index + 1 < len(entries):
            end = entries[index + 1][0]
        elif fallback_end is not None:
            end = float(fallback_end)
        else:
            continue
        if end <= start:
            continue
        sections.append(
            Section(
                start=start,
                end=end,
                label=normalized_label,
                confidence=None,
                metadata={},
            )
        )
    return sections


def _file_identity_summary(pairs: list[TrackPair]) -> dict[str, Any]:
    import hashlib

    same_size = 0
    same_hash = 0
    different_hashes: list[str] = []
    for pair in pairs:
        if pair.original_path.stat().st_size == pair.revised_path.stat().st_size:
            same_size += 1
        original_hash = hashlib.sha256(pair.original_path.read_bytes()).hexdigest()
        revised_hash = hashlib.sha256(pair.revised_path.read_bytes()).hexdigest()
        if original_hash == revised_hash:
            same_hash += 1
        else:
            different_hashes.append(pair.track_id)
    return {
        "same_size_count": same_size,
        "same_sha256_count": same_hash,
        "different_sha256_count": len(different_hashes),
        "different_sha256_track_ids": different_hashes,
    }


def _compare_track(track_id: str, original: list[Section], revised: list[Section], trim: bool) -> dict[str, Any]:
    import mir_eval

    original_duration = _annotation_end(original)
    revised_duration = _annotation_end(revised)
    duration = max(original_duration, revised_duration)
    original_boundaries = _internal_boundaries(original)
    revised_boundaries = _internal_boundaries(revised)
    orig_to_rev = _nearest_distances(original_boundaries, revised_boundaries)
    rev_to_orig = _nearest_distances(revised_boundaries, original_boundaries)
    signed_shifts = _signed_nearest_shifts(original_boundaries, revised_boundaries, max_distance=3.0)
    boundary_scores = {}
    for tolerance in (0.5, 3.0):
        precision, recall, f_measure = _boundary_prf(
            reference=original_boundaries,
            estimate=revised_boundaries,
            tolerance=tolerance,
        )
        suffix = str(tolerance)
        boundary_scores[f"Boundary Precision@{suffix}"] = precision
        boundary_scores[f"Boundary Recall@{suffix}"] = recall
        boundary_scores[f"Boundary F-measure@{suffix}"] = f_measure

    original_intervals, original_labels = sections_to_intervals_labels(original)
    revised_intervals, revised_labels = sections_to_intervals_labels(revised)
    original_intervals, original_labels = mir_eval.util.adjust_intervals(
        original_intervals,
        list(original_labels),
        t_min=0.0,
        t_max=duration,
    )
    revised_intervals, revised_labels = mir_eval.util.adjust_intervals(
        revised_intervals,
        list(revised_labels),
        t_min=0.0,
        t_max=duration,
    )
    scores = mir_eval.segment.evaluate(
        original_intervals,
        original_labels,
        revised_intervals,
        revised_labels,
        trim=trim,
    )
    overlap = _overlap_seconds(original, revised)
    total_overlap = sum(overlap.values())
    same_label_seconds = sum(seconds for (left, right), seconds in overlap.items() if left == right)
    same_base_seconds = sum(
        seconds for (left, right), seconds in overlap.items() if label_base(left) == label_base(right)
    )

    row = {
        "track_id": track_id,
        "duration_original": original_duration,
        "duration_revised": revised_duration,
        "duration_delta_revised_minus_original": revised_duration - original_duration,
        "n_segments_original": len(original),
        "n_segments_revised": len(revised),
        "segment_count_delta_revised_minus_original": len(revised) - len(original),
        "n_boundaries_original": len(original_boundaries),
        "n_boundaries_revised": len(revised_boundaries),
        "boundary_count_delta_revised_minus_original": len(revised_boundaries) - len(original_boundaries),
        "n_unique_labels_original": len({section.label for section in original}),
        "n_unique_labels_revised": len({section.label for section in revised}),
        "mean_segment_duration_original": _safe_mean([section.duration for section in original]),
        "mean_segment_duration_revised": _safe_mean([section.duration for section in revised]),
        "median_segment_duration_original": _safe_median([section.duration for section in original]),
        "median_segment_duration_revised": _safe_median([section.duration for section in revised]),
        "consecutive_duplicate_labels_original": _consecutive_duplicate_count(original),
        "consecutive_duplicate_labels_revised": _consecutive_duplicate_count(revised),
        "total_gap_seconds_original": _gap_overlap_seconds(original)[0],
        "total_overlap_seconds_original": _gap_overlap_seconds(original)[1],
        "total_gap_seconds_revised": _gap_overlap_seconds(revised)[0],
        "total_overlap_seconds_revised": _gap_overlap_seconds(revised)[1],
        "label_agreement_seconds": same_label_seconds,
        "base_label_agreement_seconds": same_base_seconds,
        "label_agreement_fraction": same_label_seconds / total_overlap if total_overlap else float("nan"),
        "base_label_agreement_fraction": same_base_seconds / total_overlap if total_overlap else float("nan"),
        "orig_to_revised_boundary_median_abs_seconds": _safe_median(orig_to_rev),
        "orig_to_revised_boundary_p90_abs_seconds": _safe_percentile(orig_to_rev, 90),
        "revised_to_orig_boundary_median_abs_seconds": _safe_median(rev_to_orig),
        "revised_to_orig_boundary_p90_abs_seconds": _safe_percentile(rev_to_orig, 90),
        "added_boundaries_not_within_0.5s": _unmatched_count(revised_boundaries, original_boundaries, 0.5),
        "removed_boundaries_not_within_0.5s": _unmatched_count(original_boundaries, revised_boundaries, 0.5),
        "_orig_to_rev_distances": orig_to_rev,
        "_rev_to_orig_distances": rev_to_orig,
        "_signed_boundary_shifts": signed_shifts,
    }
    for metric in METRICS:
        row[metric] = float(scores[metric])
    row.update(boundary_scores)
    return row


def _annotation_end(sections: list[Section]) -> float:
    return max((section.end for section in sections), default=0.0)


def _internal_boundaries(sections: list[Section]) -> np.ndarray:
    if len(sections) <= 1:
        return np.asarray([], dtype=float)
    return np.asarray([section.end for section in sections[:-1]], dtype=float)


def _nearest_distances(query: np.ndarray, reference: np.ndarray) -> list[float]:
    if len(query) == 0 or len(reference) == 0:
        return []
    distances = np.abs(query[:, None] - reference[None, :])
    return distances.min(axis=1).astype(float).tolist()


def _signed_nearest_shifts(query: np.ndarray, reference: np.ndarray, max_distance: float) -> list[float]:
    if len(query) == 0 or len(reference) == 0:
        return []
    differences = reference[None, :] - query[:, None]
    nearest = np.abs(differences).argmin(axis=1)
    shifts = differences[np.arange(len(query)), nearest]
    return shifts[np.abs(shifts) <= max_distance].astype(float).tolist()


def _boundary_prf(reference: np.ndarray, estimate: np.ndarray, tolerance: float) -> tuple[float, float, float]:
    matched_estimates = _matched_count(estimate, reference, tolerance)
    matched_references = _matched_count(reference, estimate, tolerance)
    precision = matched_estimates / len(estimate) if len(estimate) else float(len(reference) == 0)
    recall = matched_references / len(reference) if len(reference) else float(len(estimate) == 0)
    f_measure = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    return precision, recall, f_measure


def _matched_count(query: np.ndarray, reference: np.ndarray, tolerance: float) -> int:
    if len(query) == 0 or len(reference) == 0:
        return 0
    distances = np.abs(query[:, None] - reference[None, :])
    return int((distances.min(axis=1) <= tolerance).sum())


def _unmatched_count(query: np.ndarray, reference: np.ndarray, tolerance: float) -> int:
    return len(query) - _matched_count(query, reference, tolerance)


def _overlap_seconds(original: list[Section], revised: list[Section]) -> dict[tuple[str, str], float]:
    output: dict[tuple[str, str], float] = defaultdict(float)
    left_index = 0
    right_index = 0
    while left_index < len(original) and right_index < len(revised):
        left = original[left_index]
        right = revised[right_index]
        start = max(left.start, right.start)
        end = min(left.end, right.end)
        if end > start:
            output[(left.label, right.label)] += end - start
        if left.end <= right.end:
            left_index += 1
        else:
            right_index += 1
    return output


def _gap_overlap_seconds(sections: list[Section]) -> tuple[float, float]:
    gaps = 0.0
    overlaps = 0.0
    for left, right in zip(sections, sections[1:]):
        delta = right.start - left.end
        if delta > 0:
            gaps += delta
        elif delta < 0:
            overlaps += -delta
    return gaps, overlaps


def _consecutive_duplicate_count(sections: list[Section]) -> int:
    return sum(
        1
        for left, right in zip(sections, sections[1:])
        if label_base(left.label) == label_base(right.label)
    )


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _safe_median(values: list[float] | np.ndarray) -> float:
    return float(np.median(values)) if len(values) else float("nan")


def _safe_percentile(values: list[float] | np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if len(values) else float("nan")


def _label_family(label: str) -> str:
    base = label_base(label)
    if "pre chorus" in base or "prechorus" in base:
        return "prechorus"
    if "post chorus" in base or "postchorus" in base:
        return "postchorus"
    if "chorus" in base:
        return "chorus"
    if "verse" in base:
        return "verse"
    if "bridge" in base:
        return "bridge"
    if "intro" in base:
        return "intro"
    if "outro" in base or "fade out" in base or "fadeout" in base:
        return "outro"
    if "inst" in base or "instrument" in base or "solo" in base or "gtr" in base:
        return "instrumental"
    if "break" in base or "interlude" in base or "transition" in base:
        return "break_transition"
    if "silence" in base or base in {"none", "no function"}:
        return "silence"
    return "other"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_label_seconds(path: Path, original: Counter[str], revised: Counter[str]) -> None:
    labels = sorted(set(original) | set(revised))
    rows = [
        {
            "label": label,
            "original_seconds": float(original[label]),
            "revised_seconds": float(revised[label]),
            "delta_revised_minus_original_seconds": float(revised[label] - original[label]),
        }
        for label in labels
    ]
    _write_csv(path, rows)


def _write_confusion_csv(path: Path, confusion: dict[tuple[str, str], float]) -> None:
    rows = [
        {"original_label": left, "revised_label": right, "seconds": float(seconds)}
        for (left, right), seconds in sorted(confusion.items(), key=lambda item: item[1], reverse=True)
    ]
    _write_csv(path, rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _build_summary(
    pairs: list[TrackPair],
    missing: dict[str, list[str]],
    file_identity: dict[str, Any],
    rows: list[dict[str, Any]],
    original_label_seconds: Counter[str],
    revised_label_seconds: Counter[str],
    original_label_counts: Counter[str],
    revised_label_counts: Counter[str],
    original_family_seconds: Counter[str],
    revised_family_seconds: Counter[str],
    confusion_seconds: dict[tuple[str, str], float],
    base_confusion_seconds: dict[tuple[str, str], float],
    original_segment_durations: list[float],
    revised_segment_durations: list[float],
    orig_to_rev_distances: list[float],
    rev_to_orig_distances: list[float],
    signed_boundary_shifts: list[float],
) -> dict[str, Any]:
    metric_summary = {
        metric: _summarize_values([row[metric] for row in rows])
        for metric in METRICS
        if metric in rows[0]
    }
    track_stats = {
        "n_segments_original": _summarize_values([row["n_segments_original"] for row in rows]),
        "n_segments_revised": _summarize_values([row["n_segments_revised"] for row in rows]),
        "segment_count_delta_revised_minus_original": _summarize_values(
            [row["segment_count_delta_revised_minus_original"] for row in rows]
        ),
        "n_unique_labels_original": _summarize_values([row["n_unique_labels_original"] for row in rows]),
        "n_unique_labels_revised": _summarize_values([row["n_unique_labels_revised"] for row in rows]),
        "label_agreement_fraction": _summarize_values([row["label_agreement_fraction"] for row in rows]),
        "base_label_agreement_fraction": _summarize_values([row["base_label_agreement_fraction"] for row in rows]),
        "Boundary F-measure@0.5": _summarize_values([row["Boundary F-measure@0.5"] for row in rows]),
        "Boundary F-measure@3.0": _summarize_values([row["Boundary F-measure@3.0"] for row in rows]),
    }
    original_labels = set(original_label_counts)
    revised_labels = set(revised_label_counts)
    original_bases = {label_base(label) for label in original_label_counts}
    revised_bases = {label_base(label) for label in revised_label_counts}
    changed_pairs = [
        {
            "original_label": left,
            "revised_label": right,
            "seconds": float(seconds),
        }
        for (left, right), seconds in sorted(confusion_seconds.items(), key=lambda item: item[1], reverse=True)
        if left != right
    ][:40]
    changed_base_pairs = [
        {
            "original_base": left,
            "revised_base": right,
            "seconds": float(seconds),
        }
        for (left, right), seconds in sorted(base_confusion_seconds.items(), key=lambda item: item[1], reverse=True)
        if left != right
    ][:40]
    return {
        "num_matched_tracks": len(pairs),
        "missing": missing,
        "file_identity": file_identity,
        "metrics": metric_summary,
        "track_stats": track_stats,
        "segment_duration_seconds": {
            "original": _summarize_values(original_segment_durations),
            "revised": _summarize_values(revised_segment_durations),
        },
        "boundary_nearest_distance_seconds": {
            "original_to_revised": _summarize_values(orig_to_rev_distances),
            "revised_to_original": _summarize_values(rev_to_orig_distances),
            "signed_revised_minus_original_within_3s": _summarize_values(signed_boundary_shifts),
        },
        "labels": {
            "num_original": len(original_labels),
            "num_revised": len(revised_labels),
            "num_shared": len(original_labels & revised_labels),
            "original_only": sorted(original_labels - revised_labels),
            "revised_only": sorted(revised_labels - original_labels),
            "num_original_bases": len(original_bases),
            "num_revised_bases": len(revised_bases),
            "num_shared_bases": len(original_bases & revised_bases),
            "original_only_bases": sorted(original_bases - revised_bases),
            "revised_only_bases": sorted(revised_bases - original_bases),
        },
        "top_original_labels_by_duration": _top_counter(original_label_seconds, 30),
        "top_revised_labels_by_duration": _top_counter(revised_label_seconds, 30),
        "label_families_by_duration": {
            "original": _top_counter(original_family_seconds, 20),
            "revised": _top_counter(revised_family_seconds, 20),
        },
        "top_changed_label_overlaps": changed_pairs,
        "top_changed_base_label_overlaps": changed_base_pairs,
        "tracks_lowest_F_measure_at_3": _top_rows(rows, "F-measure@3.0", reverse=False, count=20),
        "tracks_largest_absolute_segment_count_delta": sorted(
            (
                {
                    "track_id": row["track_id"],
                    "segment_count_delta_revised_minus_original": row[
                        "segment_count_delta_revised_minus_original"
                    ],
                    "F-measure@3.0": row["F-measure@3.0"],
                    "label_agreement_fraction": row["label_agreement_fraction"],
                }
                for row in rows
            ),
            key=lambda row: abs(row["segment_count_delta_revised_minus_original"]),
            reverse=True,
        )[:20],
        "tracks_most_added_boundaries": _top_rows(rows, "added_boundaries_not_within_0.5s", count=20),
        "tracks_most_removed_boundaries": _top_rows(rows, "removed_boundaries_not_within_0.5s", count=20),
    }


def _summarize_values(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"count": 0}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _top_counter(counter: Counter[str], count: int) -> list[dict[str, Any]]:
    total = sum(counter.values())
    return [
        {
            "label": label,
            "seconds": float(seconds),
            "hours": float(seconds / 3600.0),
            "share": float(seconds / total) if total else float("nan"),
        }
        for label, seconds in counter.most_common(count)
    ]


def _top_rows(rows: list[dict[str, Any]], key: str, count: int, reverse: bool = True) -> list[dict[str, Any]]:
    keep = [
        "track_id",
        key,
        "F-measure@0.5",
        "F-measure@3.0",
        "Pairwise F-measure",
        "NCE F-measure",
        "n_segments_original",
        "n_segments_revised",
        "label_agreement_fraction",
        "base_label_agreement_fraction",
    ]
    return [
        {field: row[field] for field in keep if field in row}
        for row in sorted(rows, key=lambda row: row[key], reverse=reverse)[:count]
    ]


def _write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Harmonix Annotation Comparison",
        "",
        f"Matched tracks: {summary['num_matched_tracks']}",
        f"Exact SHA-256 identical files: {summary['file_identity']['same_sha256_count']}",
        "",
        "## Segment Agreement",
        "",
    ]
    for metric in ("F-measure@0.5", "F-measure@3.0", "Pairwise F-measure", "NCE F-measure"):
        stats = summary["metrics"][metric]
        lines.append(f"- {metric}: {stats['mean']:.4f} +/- {stats['std']:.4f}")
    lines.extend(["", "## Boundary Agreement", ""])
    for metric in ("Boundary F-measure@0.5", "Boundary F-measure@3.0"):
        stats = summary["track_stats"][metric]
        lines.append(f"- {metric}: {stats['mean']:.4f} +/- {stats['std']:.4f}")
    lines.extend(["", "## Granularity", ""])
    for key in ("n_segments_original", "n_segments_revised", "segment_count_delta_revised_minus_original"):
        stats = summary["track_stats"][key]
        lines.append(f"- {key}: mean={stats['mean']:.2f}, median={stats['median']:.2f}")
    lines.extend(["", "## Label Vocab", ""])
    labels = summary["labels"]
    lines.append(
        f"- Raw labels: original={labels['num_original']}, revised={labels['num_revised']}, shared={labels['num_shared']}"
    )
    lines.append(
        f"- Base labels: original={labels['num_original_bases']}, revised={labels['num_revised_bases']}, shared={labels['num_shared_bases']}"
    )
    lines.extend(["", "## Top Changed Label Overlaps", ""])
    for item in summary["top_changed_label_overlaps"][:15]:
        lines.append(
            f"- {item['original_label']} -> {item['revised_label']}: {item['seconds'] / 3600.0:.2f} h"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _make_plots(
    output_dir: Path,
    rows: list[dict[str, Any]],
    original_segment_durations: list[float],
    revised_segment_durations: list[float],
    orig_to_rev_distances: list[float],
    rev_to_orig_distances: list[float],
    signed_boundary_shifts: list[float],
    original_label_seconds: Counter[str],
    revised_label_seconds: Counter[str],
    original_family_seconds: Counter[str],
    revised_family_seconds: Counter[str],
    confusion_seconds: dict[tuple[str, str], float],
) -> None:
    import matplotlib.pyplot as plt

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    _plot_metric_distributions(plots_dir / "metric_distributions.png", rows, plt)
    _plot_segment_counts(plots_dir / "segment_counts.png", rows, plt)
    _plot_segment_duration_hist(
        plots_dir / "segment_duration_histogram.png",
        original_segment_durations,
        revised_segment_durations,
        plt,
    )
    _plot_boundary_distances(
        plots_dir / "boundary_distance_histogram.png",
        orig_to_rev_distances,
        rev_to_orig_distances,
        plt,
    )
    _plot_boundary_shifts(plots_dir / "boundary_signed_shift_histogram.png", signed_boundary_shifts, plt)
    _plot_label_durations(
        plots_dir / "top_label_durations.png",
        original_label_seconds,
        revised_label_seconds,
        plt,
    )
    _plot_label_durations(
        plots_dir / "label_family_durations.png",
        original_family_seconds,
        revised_family_seconds,
        plt,
        top_n=12,
        xlabel="hours",
    )
    _plot_metric_vs_delta(plots_dir / "f3_vs_segment_count_delta.png", rows, plt)
    _plot_boundary_counts(plots_dir / "boundary_counts.png", rows, plt)
    _plot_confusion(plots_dir / "label_overlap_heatmap.png", confusion_seconds, plt)
    plt.close("all")


def _plot_metric_distributions(path: Path, rows: list[dict[str, Any]], plt) -> None:
    metrics = ["F-measure@0.5", "F-measure@3.0", "Pairwise F-measure", "NCE F-measure"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, metric in zip(axes.ravel(), metrics):
        values = [row[metric] for row in rows]
        ax.hist(values, bins=30, color="#4c78a8", alpha=0.85)
        ax.axvline(np.mean(values), color="#f58518", linewidth=2, label=f"mean={np.mean(values):.3f}")
        ax.set_title(metric)
        ax.set_xlim(0, 1)
        ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_segment_counts(path: Path, rows: list[dict[str, Any]], plt) -> None:
    original = np.asarray([row["n_segments_original"] for row in rows])
    revised = np.asarray([row["n_segments_revised"] for row in rows])
    delta = revised - original
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].scatter(original, revised, s=14, alpha=0.65)
    limit = max(original.max(), revised.max()) + 1
    axes[0].plot([0, limit], [0, limit], color="black", linewidth=1)
    axes[0].set_xlabel("original segments")
    axes[0].set_ylabel("revised segments")
    axes[0].set_title("Track-Level Segment Counts")
    axes[1].hist(delta, bins=np.arange(delta.min() - 0.5, delta.max() + 1.5), color="#54a24b")
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_xlabel("revised - original segment count")
    axes[1].set_title("Granularity Shift")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_segment_duration_hist(path: Path, original: list[float], revised: list[float], plt) -> None:
    bins = np.linspace(0, 80, 81)
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    ax.hist(original, bins=bins, alpha=0.55, label="original", density=True)
    ax.hist(revised, bins=bins, alpha=0.55, label="revised", density=True)
    ax.set_xlabel("segment duration (s)")
    ax.set_ylabel("density")
    ax.set_title("Segment Duration Distribution")
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_boundary_distances(path: Path, original_to_revised: list[float], revised_to_original: list[float], plt) -> None:
    bins = np.linspace(0, 10, 80)
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    ax.hist(np.clip(original_to_revised, 0, 10), bins=bins, alpha=0.55, density=True, label="original -> nearest revised")
    ax.hist(np.clip(revised_to_original, 0, 10), bins=bins, alpha=0.55, density=True, label="revised -> nearest original")
    ax.axvline(0.5, color="#e45756", linestyle="--", label="0.5s")
    ax.axvline(3.0, color="#f58518", linestyle="--", label="3.0s")
    ax.set_xlabel("nearest boundary distance (s), clipped at 10")
    ax.set_ylabel("density")
    ax.set_title("Boundary Timing Differences")
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_boundary_shifts(path: Path, shifts: list[float], plt) -> None:
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    ax.hist(shifts, bins=np.linspace(-3, 3, 80), color="#b279a2", alpha=0.85)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("nearest revised boundary - original boundary (s)")
    ax.set_ylabel("count")
    ax.set_title("Signed Boundary Shifts Within 3s")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_label_durations(
    path: Path,
    original: Counter[str],
    revised: Counter[str],
    plt,
    top_n: int = 25,
    xlabel: str = "hours",
) -> None:
    labels = [
        label
        for label, _ in Counter({label: original[label] + revised[label] for label in set(original) | set(revised)}).most_common(top_n)
    ]
    y = np.arange(len(labels))
    original_values = np.asarray([original[label] / 3600.0 for label in labels])
    revised_values = np.asarray([revised[label] / 3600.0 for label in labels])
    fig, ax = plt.subplots(figsize=(10, max(5, 0.25 * len(labels))), constrained_layout=True)
    ax.barh(y - 0.18, original_values, height=0.36, label="original")
    ax.barh(y + 0.18, revised_values, height=0.36, label="revised")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title("Top Label Durations")
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_metric_vs_delta(path: Path, rows: list[dict[str, Any]], plt) -> None:
    delta = np.asarray([row["segment_count_delta_revised_minus_original"] for row in rows])
    f3 = np.asarray([row["F-measure@3.0"] for row in rows])
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.scatter(delta, f3, s=18, alpha=0.65)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("revised - original segment count")
    ax.set_ylabel("F-measure@3.0")
    ax.set_title("Agreement vs Granularity Change")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_boundary_counts(path: Path, rows: list[dict[str, Any]], plt) -> None:
    original = np.asarray([row["n_boundaries_original"] for row in rows])
    revised = np.asarray([row["n_boundaries_revised"] for row in rows])
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.scatter(original, revised, s=14, alpha=0.65)
    limit = max(original.max(), revised.max()) + 1
    ax.plot([0, limit], [0, limit], color="black", linewidth=1)
    ax.set_xlabel("original boundaries")
    ax.set_ylabel("revised boundaries")
    ax.set_title("Boundary Counts")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_confusion(path: Path, confusion: dict[tuple[str, str], float], plt) -> None:
    label_totals = Counter()
    for (left, right), seconds in confusion.items():
        label_totals[left] += seconds
        label_totals[right] += seconds
    labels = [label for label, _ in label_totals.most_common(25)]
    index = {label: idx for idx, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=float)
    for (left, right), seconds in confusion.items():
        if left in index and right in index:
            matrix[index[left], index[right]] += seconds / 3600.0
    fig, ax = plt.subplots(figsize=(11, 9), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(labels)), labels, rotation=70, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("revised label")
    ax.set_ylabel("original label")
    ax.set_title("Label Overlap Heatmap (hours)")
    fig.colorbar(image, ax=ax, label="hours")
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
