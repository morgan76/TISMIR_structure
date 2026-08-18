#!/usr/bin/env python3
"""Convert the HarmonixSet (bigvgan) corrected JSONL into per-track JAMS + manifests.

Input record shape (one JSON object per line)::

    {"data_id": "0001_12step", "dataset_type": "harmonixset_8class",
     "msa_info": [[time, label], ...], "split": "train|val|test"}

``msa_info`` is a list of ``[onset_time, label]`` pairs. Section ``i`` spans
``[t_i, t_{i+1})`` with ``label_i``; the final entry ``[t_last, "end"]`` marks
the track duration and is used as the closing boundary, not emitted as a
section.

Outputs:
- ``data/raw/harmonix/jams/<data_id>.jams`` in the ``segment_open`` namespace.
- ``data/manifests/harmonix_bigvgan.local.jsonl`` plus one manifest per official
  split (``harmonix_bigvgan_{train,val,test}.local.jsonl``), carrying ``split``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tismir.data.manifest import save_manifest
from tismir.data.schemas import Section, Track
from tismir.decoding.jams import save_segments_jams

END_LABEL = "end"


def sections_from_msa_info(msa_info: list) -> tuple[list[Section], float]:
    """Turn ``[[t, label], ...]`` onset pairs into sections + track duration.

    The trailing ``[t_last, "end"]`` entry is the closing boundary: it sets the
    duration and the end of the final real section, and is not itself a section.
    """

    if len(msa_info) < 2:
        raise ValueError("msa_info must contain at least one section and a closing boundary")

    boundaries = [(float(t), str(label)) for t, label in msa_info]
    duration = boundaries[-1][0]

    sections: list[Section] = []
    for (start, label), (end, _next_label) in zip(boundaries[:-1], boundaries[1:]):
        if label == END_LABEL:
            # Defensive: an 'end' label should only ever be the final entry.
            continue
        if end <= start:
            continue
        sections.append(Section(start=start, end=end, label=label))
    if not sections:
        raise ValueError("No positive-duration sections built from msa_info")
    return sections, duration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=Path("data/raw/harmonix_bigvgan/harmonixset.corrected.20250821.jsonl"),
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=Path("data/raw/harmonix/harmonixset_bigvgan/tracks"),
        help="Directory holding <data_id>.wav files.",
    )
    parser.add_argument(
        "--jams-root",
        type=Path,
        default=Path("data/raw/harmonix/jams"),
        help="Directory to write <data_id>.jams files into.",
    )
    parser.add_argument("--audio-ext", default=".wav")
    parser.add_argument("--dataset", default="harmonix")
    parser.add_argument("--namespace", default="segment_open")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/harmonix_bigvgan.local.jsonl"),
    )
    parser.add_argument(
        "--skip-missing-audio",
        action="store_true",
        help="Skip records whose <data_id>.wav is absent instead of erroring.",
    )
    args = parser.parse_args()

    args.jams_root.mkdir(parents=True, exist_ok=True)

    tracks: list[Track] = []
    missing = 0
    written = 0
    with args.jsonl.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            data_id = str(record["data_id"])
            split = record.get("split")

            audio_path = (args.audio_root / f"{data_id}{args.audio_ext}").resolve()
            if not audio_path.exists():
                missing += 1
                if args.skip_missing_audio:
                    continue
                raise FileNotFoundError(
                    f"Missing audio for {data_id} at {audio_path} (line {line_number}); "
                    f"pass --skip-missing-audio to skip."
                )

            sections, duration = sections_from_msa_info(record["msa_info"])
            jams_path = (args.jams_root / f"{data_id}.jams").resolve()
            save_segments_jams(
                jams_path,
                [(s.start, s.end, s.label) for s in sections],
                duration=duration,
                namespace=args.namespace,
            )
            written += 1

            tracks.append(
                Track(
                    track_id=data_id,
                    audio_path=audio_path,
                    jams_path=jams_path,
                    dataset=args.dataset,
                    split=None if split is None else str(split),
                )
            )

    if not tracks:
        raise ValueError("No Harmonix tracks produced; check paths.")

    save_manifest(args.manifest, tracks)
    print(f"Wrote {written} JAMS to {args.jams_root}")
    print(f"Wrote {len(tracks)} tracks to {args.manifest}")
    if missing:
        print(f"Skipped {missing} records with missing audio")

    # Per-split manifests from the embedded official split.
    by_split: dict[str, list[Track]] = {}
    for track in tracks:
        by_split.setdefault(track.split or "unknown", []).append(track)
    for split_name, split_tracks in sorted(by_split.items()):
        split_path = args.manifest.with_name(
            args.manifest.stem.replace(".local", "") + f"_{split_name}.local.jsonl"
        )
        save_manifest(split_path, split_tracks)
        print(f"  {split_name}: {len(split_tracks)} -> {split_path}")


if __name__ == "__main__":
    main()
