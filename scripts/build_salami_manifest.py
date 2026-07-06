#!/usr/bin/env python3
"""Build a SALAMI manifest from the audio we can actually source.

Covers two collections:
  * internetarchive — files fetched by fetch_salami_internetarchive.py,
    named <SONG_ID>.<ext> in --ia-audio-dir.
  * rwc — SALAMI's id_index_rwc.csv gives (collection, disc, track); the
    rwc-annotations metadata.csv maps that to RWCID (exact catalogue join,
    validated by duration agreement with the SALAMI annotation < 3 s).
    Audio roots must contain RWC_<X>NNN.wav files.

Codaich and Isophonics tracks have no sourceable audio and are skipped.

Usage:
  python scripts/build_salami_manifest.py \
    --salami-root /scratch/ick/music_structure/salami \
    --rwc-metadata /scratch/ick/music_structure/rwc_pop_annotations/metadata.csv \
    --rwc-audio-root /scratch/ick/music_structure/rwc_pop_audio/RWC-P \
    --rwc-audio-root /scratch/ick/music_structure/rwc_cgj/RWC-C \
    --rwc-audio-root /scratch/ick/music_structure/rwc_cgj/RWC-G \
    --rwc-audio-root /scratch/ick/music_structure/rwc_cgj/RWC-J \
    --ia-audio-dir /scratch/ick/music_structure/salami/audio/internetarchive \
    --output /scratch/ick/music_structure/tismir_data/manifests/salami.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

DURATION_TOLERANCE = 3.0


def rwc_entries(salami_root: Path, rwc_metadata: str, audio_roots: list[Path], jams_dir: Path) -> tuple[list[dict], list[str]]:
    catalogue = {}
    for row in csv.DictReader(open(rwc_metadata), delimiter=";"):
        key = (row["CollID"], int(row["CDNo"]), int(row["TrackNo"]))
        catalogue[key] = (row["RWCID"], float(row["duration"]))

    wavs = {}
    for root in audio_roots:
        for wav in root.glob("*.wav"):
            wavs[wav.stem] = wav

    entries, problems = [], []
    index = salami_root / "metadata" / "id_index_rwc.csv"
    for row in csv.DictReader(open(index)):
        song_id = row["SONG_ID"]
        jams_path = jams_dir / f"{song_id}.jams"
        if not jams_path.exists():
            problems.append(f"{song_id}: no SALAMI annotation")
            continue
        collection = row["RWC_ID"].split("_")[2]
        disc = int(row["RWC_ID"].split("_M")[-1])
        track = int(row["TRACK_NUMBER"])
        hit = catalogue.get((collection, disc, track))
        if hit is None:
            problems.append(f"{song_id}: no catalogue entry for {collection} disc {disc} track {track}")
            continue
        rwcid, cat_duration = hit
        jams_duration = json.load(open(jams_path))["file_metadata"]["duration"]
        if abs(cat_duration - jams_duration) > DURATION_TOLERANCE:
            problems.append(f"{song_id}: duration mismatch {rwcid} ({cat_duration:.1f}s vs {jams_duration:.1f}s)")
            continue
        wav = wavs.get(rwcid)
        if wav is None:
            problems.append(f"{song_id}: audio {rwcid}.wav not found")
            continue
        entries.append(_entry(song_id, wav, jams_path, collection=f"rwc ({rwcid})"))
    return entries, problems


def ia_entries(salami_root: Path, ia_dir: Path, jams_dir: Path) -> tuple[list[dict], list[str]]:
    entries, problems = [], []
    index = salami_root / "metadata" / "id_index_internetarchive.csv"
    for row in csv.DictReader(open(index)):
        song_id = row["SONG_ID"]
        jams_path = jams_dir / f"{song_id}.jams"
        audio = sorted(ia_dir.glob(f"{song_id}.*"))
        if not audio:
            problems.append(f"{song_id}: not fetched")
            continue
        if not jams_path.exists():
            problems.append(f"{song_id}: no SALAMI annotation")
            continue
        entries.append(_entry(song_id, audio[0], jams_path, collection="internetarchive"))
    return entries, problems


def _entry(song_id: str, audio: Path, jams_path: Path, collection: str) -> dict:
    return {
        "track_id": song_id,
        "audio_path": str(audio.resolve()),
        "jams_path": str(jams_path.resolve()),
        "dataset": "salami",
        "split": None,
        "metadata": {"collection": collection},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--salami-root", required=True)
    parser.add_argument("--rwc-metadata", required=True)
    parser.add_argument("--rwc-audio-root", action="append", default=[])
    parser.add_argument("--ia-audio-dir", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    salami_root = Path(args.salami_root)
    jams_dir = salami_root / "jams"
    entries, problems = rwc_entries(
        salami_root, args.rwc_metadata, [Path(p) for p in args.rwc_audio_root], jams_dir
    )
    print(f"rwc: {len(entries)} entries")
    if args.ia_audio_dir is not None:
        ia, ia_problems = ia_entries(salami_root, Path(args.ia_audio_dir), jams_dir)
        entries += ia
        problems += ia_problems
        print(f"internetarchive: {len(ia)} entries")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")
    print(f"wrote {len(entries)} rows to {output}")
    if problems:
        print(f"{len(problems)} tracks skipped; first few:")
        for line in problems[:10]:
            print("  -", line)


if __name__ == "__main__":
    main()
