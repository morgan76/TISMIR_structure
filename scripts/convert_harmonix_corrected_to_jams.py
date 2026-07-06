#!/usr/bin/env python3
"""Convert corrected HarmonixSet annotations to segment_open JAMS files.

Input is the JSONL shipped with the HF dataset m-a-p/harmonixset_bigvgan
(harmonixset.corrected.20250821.jsonl): one row per track with
`data_id`, `msa_info` = [[start_seconds, label], ...] and `split`
(train/val/test). The final `end` event marks the track end and is used as
the duration terminator, not emitted as a segment.

Optionally also writes per-split manifest JSONL files pairing each JAMS with
its audio file, using the dataset's canonical splits.

Usage:
  python scripts/convert_harmonix_corrected_to_jams.py \
    --jsonl /scratch/ick/music_structure/harmonix/harmonixset.corrected.20250821.jsonl \
    --output-dir /scratch/ick/music_structure/harmonix/jams \
    --audio-dir /scratch/ick/music_structure/harmonix/audio \
    --manifest-dir /scratch/ick/music_structure/tismir_data/manifests \
    --manifest-name harmonix
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jams


def convert_row(row: dict, drop_labels: set[str]) -> jams.JAMS:
    events = row["msa_info"]
    duration = float(events[-1][0])
    jam = jams.JAMS()
    jam.file_metadata.duration = duration
    jam.file_metadata.title = row["data_id"]
    annotation = jams.Annotation(namespace="segment_open", duration=duration)
    annotation.annotation_metadata = jams.AnnotationMetadata(
        corpus="Harmonix Set (corrected 2025-08-21, m-a-p/harmonixset_bigvgan)",
        data_source=str(row.get("dataset_type", "")),
    )
    annotation.sandbox.split = row.get("split")
    for (start, label), (end, _) in zip(events[:-1], events[1:]):
        if label in drop_labels:
            continue
        if float(end) <= float(start):
            continue
        annotation.append(time=float(start), duration=float(end) - float(start), value=str(label))
    jam.annotations.append(annotation)
    return jam


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--drop-labels", nargs="*", default=["end"])
    parser.add_argument("--audio-dir", default=None, help="If set, also emit manifests for tracks with audio.")
    parser.add_argument("--audio-ext", default=".wav")
    parser.add_argument("--manifest-dir", default=None)
    parser.add_argument("--manifest-name", default="harmonix")
    parser.add_argument("--dataset", default="harmonix")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    drop_labels = set(args.drop_labels)

    rows = [json.loads(line) for line in open(args.jsonl)]
    by_split: dict[str, list[dict]] = {}
    missing_audio = 0
    for row in rows:
        jam = convert_row(row, drop_labels)
        jams_path = output_dir / f"{row['data_id']}.jams"
        jam.save(str(jams_path))
        if args.audio_dir is not None:
            audio_path = Path(args.audio_dir) / f"{row['data_id']}{args.audio_ext}"
            if not audio_path.exists():
                missing_audio += 1
                continue
            entry = {
                "track_id": row["data_id"],
                "audio_path": str(audio_path.resolve()),
                "jams_path": str(jams_path.resolve()),
                "dataset": args.dataset,
                "split": row.get("split"),
                "metadata": {},
            }
            by_split.setdefault(row.get("split") or "unsplit", []).append(entry)

    print(f"wrote {len(rows)} JAMS files to {output_dir}")

    if args.manifest_dir is not None and args.audio_dir is not None:
        manifest_dir = Path(args.manifest_dir)
        manifest_dir.mkdir(parents=True, exist_ok=True)
        all_entries = [entry for entries in by_split.values() for entry in entries]
        targets = {f"{args.manifest_name}.jsonl": all_entries}
        for split, entries in by_split.items():
            targets[f"{args.manifest_name}_{split}.jsonl"] = entries
        for filename, entries in targets.items():
            path = manifest_dir / filename
            with open(path, "w") as handle:
                for entry in entries:
                    handle.write(json.dumps(entry) + "\n")
            print(f"wrote {len(entries)} rows to {path}")
        if missing_audio:
            print(f"WARNING: {missing_audio} tracks skipped from manifests (no audio at {args.audio_dir})")


if __name__ == "__main__":
    main()
