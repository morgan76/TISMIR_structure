#!/usr/bin/env python3
"""Convert SALAMI function-level annotations to segment_open JAMS files.

Input is the public SALAMI annotation dump: one directory per track id under
--annotations-dir, each containing parsed/textfile{1,2}_functions.txt with
lines "time<TAB>label" (time in seconds). These files are the output of the
official SALAMI parser and already apply funct_vocab_dictionary.txt (typo
fixes, letter labels -> no_function); labels are preserved verbatim as JAMS
segment values.

Conventions (mirroring scripts/convert_harmonix_corrected_to_jams.py on
dataset/harmonix):
- The single "End" event marks the track end: it sets file_metadata.duration
  and is not emitted as a segment.
- "Silence" is kept as a regular segment (the pipeline can map synthetic
  boundaries to silence-like labels).
- Annotator 1 (textfile1) is the primary annotation; if it is missing,
  annotator 2 (textfile2) becomes primary. If both exist, annotator 2 is
  emitted as a second segment_open annotation in the same JAMS
  (annotation.sandbox.annotator records which is which).
- The source collection (Codaich / IA / Isophonics / RWC, from
  metadata/metadata.csv) is stored in jam.sandbox.collection and each
  annotation's sandbox, so manifests can later be filtered to subsets that
  have audio.
- Rows sharing an identical timestamp with the previous row are duplicate
  labels for the same boundary; only the FIRST label of such a tie group is
  kept. Ties arise from (a) raw lines carrying two co-occurring functions
  (e.g. "Instrumental, Solo" -> keep the first-listed, primary function),
  (b) a synthetic lowercase "silence" the official parser inserts at 0.0 when
  a song does not begin with silence (always ordered after the real label ->
  dropped), and (c) a zero-length trailing "Silence" at the same timestamp as
  "End". Dropped rows are counted and reported.

See data/datasets/SALAMI_LABELS.md for the full label-space survey.

Usage:
  PYTHONPATH=src python scripts/convert_salami_to_jams.py \
    --annotations-dir /scratch/ick/music_structure/salami/annotations \
    --metadata-csv /scratch/ick/music_structure/salami/metadata/metadata.csv \
    --output-dir /scratch/ick/music_structure/salami/jams
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import jams


def read_events(path: Path) -> list[tuple[float, str]]:
    """Read "time<TAB>label" rows. Files may lack a trailing newline."""
    events: list[tuple[float, str]] = []
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            fields = line.split("\t")
            if len(fields) != 2 or not fields[1]:
                print(f"WARNING: {path}:{lineno}: malformed row {line!r}, skipped")
                continue
            events.append((float(fields[0]), fields[1]))
    return events


def build_annotation(
    events: list[tuple[float, str]],
    annotator: int,
    collection: str | None,
    stats: Counter,
) -> tuple[jams.Annotation, float]:
    """Convert an event list to a segment_open annotation plus track duration."""
    deduped: list[tuple[float, str]] = []
    for time, label in events:
        if deduped and time == deduped[-1][0]:
            stats["rows_skipped_duplicate_timestamp"] += 1
            continue
        deduped.append((time, label))
    events = deduped

    end_times = [time for time, label in events if label == "End"]
    if end_times:
        duration = max(end_times)
    else:  # defensive; every surveyed file has exactly one End marker
        duration = max(time for time, _ in events)
        stats["files_without_end_marker"] += 1
    segments = [(time, label) for time, label in events if label != "End"]

    annotation = jams.Annotation(namespace="segment_open", duration=duration)
    annotation.annotation_metadata = jams.AnnotationMetadata(
        corpus="SALAMI v2.0 (functions level)",
        data_source=str(collection or ""),
    )
    annotation.sandbox.annotator = annotator
    annotation.sandbox.collection = collection
    annotation.sandbox.level = "functions"

    boundaries = [time for time, _ in segments] + [duration]
    for (start, label), end in zip(segments, boundaries[1:]):
        if end <= start:
            stats["rows_skipped_nonpositive_duration"] += 1
            continue
        annotation.append(time=start, duration=end - start, value=label)
    return annotation, duration


def load_metadata(metadata_csv: Path) -> dict[str, dict[str, str]]:
    with open(metadata_csv, encoding="utf-8", errors="replace") as handle:
        return {row["SONG_ID"]: row for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-dir", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="Do not emit annotator 2 as a second annotation when both exist.",
    )
    args = parser.parse_args()

    annotations_dir = Path(args.annotations_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(Path(args.metadata_csv))

    stats: Counter = Counter()
    per_collection: Counter = Counter()
    track_dirs = sorted(
        (path for path in annotations_dir.iterdir() if path.is_dir()),
        key=lambda path: int(path.name),
    )
    for track_dir in track_dirs:
        track_id = track_dir.name
        file1 = track_dir / "parsed" / "textfile1_functions.txt"
        file2 = track_dir / "parsed" / "textfile2_functions.txt"
        sources = [(1, file1), (2, file2)]
        available = [(annotator, path) for annotator, path in sources if path.is_file()]
        if not available:
            stats["tracks_skipped_no_functions"] += 1
            print(f"WARNING: {track_dir}: no functions file, skipped")
            continue
        if available[0][0] == 2:
            stats["tracks_fallback_annotator2"] += 1
        if args.primary_only:
            available = available[:1]

        row = metadata.get(track_id)
        if row is None:
            stats["tracks_missing_metadata"] += 1
        collection = row["SOURCE"] if row else None

        jam = jams.JAMS()
        jam.file_metadata.identifiers = {"salami_id": track_id}
        jam.sandbox.collection = collection
        if row:
            jam.file_metadata.title = row.get("SONG_TITLE", "") or ""
            jam.file_metadata.artist = row.get("ARTIST", "") or ""
            jam.sandbox.salami_class = row.get("CLASS", "") or ""
            jam.sandbox.genre = row.get("GENRE", "") or ""

        durations = []
        for annotator, path in available:
            events = read_events(path)
            if not events:
                stats["annotations_skipped_empty"] += 1
                continue
            annotation, duration = build_annotation(events, annotator, collection, stats)
            jam.annotations.append(annotation)
            durations.append(duration)
        if not jam.annotations:
            stats["tracks_skipped_no_functions"] += 1
            continue
        jam.file_metadata.duration = max(durations)
        for annotation in jam.annotations:
            annotation.duration = jam.file_metadata.duration

        jam.save(str(output_dir / f"{track_id}.jams"))
        stats["jams_written"] += 1
        per_collection[collection or "unknown"] += 1
        if len(jam.annotations) == 2:
            stats["tracks_with_both_annotators"] += 1

    print(f"wrote {stats['jams_written']} JAMS files to {output_dir}")
    for key in sorted(stats):
        if key != "jams_written":
            print(f"{key}: {stats[key]}")
    print("tracks per collection:")
    for collection, count in sorted(per_collection.items()):
        print(f"  {collection}: {count}")


if __name__ == "__main__":
    main()
