#!/usr/bin/env python3
"""Convert AIST RWC-MDB-P-2001 CHORUS annotations to segment_open JAMS files.

Input: AIST.RWC-MDB-P-2001.CHORUS/RM-P###.CHORUS.TXT — tab-separated
`start<TAB>end<TAB>"label"` rows with times in 10 ms units (divide by 100
for seconds). Segments tile the annotated span contiguously.

Label space (kept verbatim as targets): intro, verse A/B/C, chorus A-D,
bridge A-D, pre-chorus, post-chorus, ending, nothing. The trailing letters
mark musically distinct variants of a section type (not occurrences).
`nothing` marks non-section regions and is already recognized as
silence-like by the training pipeline's synthetic-boundary mapping.

Optionally emits a manifest pairing each RM-P### with its RWC_P###.wav.

Usage:
  python scripts/convert_rwc_chorus_to_jams.py \
    --chorus-dir "/scratch/ick/music_structure/rwc_pop_structure_aist/AIST.RWC-MDB-P-2001.CHORUS" \
    --output-dir /scratch/ick/music_structure/rwc_pop_jams \
    --audio-dir /scratch/ick/music_structure/rwc_pop_audio/RWC-P \
    --manifest /scratch/ick/music_structure/tismir_data/manifests/rwc_pop.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import jams

TIME_UNIT = 0.01  # 10 ms units -> seconds
FILE_RE = re.compile(r"RM-P(\d{3})\.CHORUS\.TXT$")


def parse_chorus_file(path: Path) -> list[tuple[float, float, str]]:
    segments = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            raise ValueError(f"{path.name}: unexpected row {line!r}")
        start = float(parts[0]) * TIME_UNIT
        end = float(parts[1]) * TIME_UNIT
        label = parts[2].strip().strip('"')
        # Chorus rows may carry a 4th column like "(-10)": the AIST modulation
        # annotation (key shift in semitones). Not part of the label space.
        if end <= start:
            continue
        segments.append((start, end, label))
    return segments


def to_jams(segments: list[tuple[float, float, str]], track_id: str) -> jams.JAMS:
    duration = segments[-1][1]
    jam = jams.JAMS()
    jam.file_metadata.duration = duration
    jam.file_metadata.title = track_id
    annotation = jams.Annotation(namespace="segment_open", duration=duration)
    annotation.annotation_metadata = jams.AnnotationMetadata(
        corpus="RWC Popular Music (AIST CHORUS annotations)",
        data_source="AIST.RWC-MDB-P-2001.CHORUS",
    )
    for start, end, label in segments:
        annotation.append(time=start, duration=end - start, value=label)
    jam.annotations.append(annotation)
    return jam


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chorus-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audio-dir", default=None, help="If set, also emit a manifest for tracks with audio.")
    parser.add_argument("--audio-template", default="RWC_P{num}.wav")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--dataset", default="rwc_pop")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    missing_audio = 0
    files = sorted(Path(args.chorus_dir).glob("RM-P*.CHORUS.TXT"))
    for path in files:
        match = FILE_RE.search(path.name)
        if match is None:
            continue
        num = match.group(1)
        track_id = f"RM-P{num}"
        segments = parse_chorus_file(path)
        if not segments:
            print(f"WARNING: {path.name} has no segments, skipped")
            continue
        jam = to_jams(segments, track_id)
        jams_path = output_dir / f"{track_id}.jams"
        jam.save(str(jams_path))
        if args.audio_dir is not None:
            audio_path = Path(args.audio_dir) / args.audio_template.format(num=num)
            if not audio_path.exists():
                missing_audio += 1
                continue
            entries.append(
                {
                    "track_id": track_id,
                    "audio_path": str(audio_path.resolve()),
                    "jams_path": str(jams_path.resolve()),
                    "dataset": args.dataset,
                    "split": None,
                    "metadata": {},
                }
            )

    print(f"wrote {len(files)} JAMS files to {output_dir}")

    if args.manifest is not None and args.audio_dir is not None:
        manifest_path = Path(args.manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as handle:
            for entry in entries:
                handle.write(json.dumps(entry) + "\n")
        print(f"wrote {len(entries)} rows to {manifest_path}")
        if missing_audio:
            print(f"WARNING: {missing_audio} tracks had no audio at {args.audio_dir}")


if __name__ == "__main__":
    main()
