#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from tismir.data.manifest import save_manifest
from tismir.data.schemas import Track


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a JSONL manifest from paired audio and JAMS folders.")
    parser.add_argument("--audio-dir", required=True, type=Path)
    parser.add_argument("--jams-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audio-ext", default=".wav")
    parser.add_argument("--jams-ext", default=".jams")
    parser.add_argument("--split", default=None)
    parser.add_argument("--absolute-paths", action="store_true")
    args = parser.parse_args()

    tracks = build_tracks(
        audio_dir=args.audio_dir,
        jams_dir=args.jams_dir,
        dataset=args.dataset,
        audio_ext=args.audio_ext,
        jams_ext=args.jams_ext,
        split=args.split,
        absolute_paths=args.absolute_paths,
    )
    save_manifest(args.output, tracks)
    print(f"Wrote {len(tracks)} tracks to {args.output}")


def build_tracks(
    audio_dir: Path,
    jams_dir: Path,
    dataset: str,
    audio_ext: str,
    jams_ext: str,
    split: str | None,
    absolute_paths: bool,
) -> list[Track]:
    audio_files = sorted(audio_dir.glob(f"*{audio_ext}"), key=lambda path: natural_key(path.stem))
    tracks: list[Track] = []
    missing_jams: list[str] = []

    for audio_path in audio_files:
        jams_path = jams_dir / f"{audio_path.stem}{jams_ext}"
        if not jams_path.exists():
            missing_jams.append(audio_path.stem)
            continue
        if absolute_paths:
            audio_value = audio_path.resolve()
            jams_value = jams_path.resolve()
        else:
            audio_value = audio_path
            jams_value = jams_path
        tracks.append(
            Track(
                track_id=audio_path.stem,
                audio_path=audio_value,
                jams_path=jams_value,
                dataset=dataset,
                split=split,
            )
        )

    if missing_jams:
        preview = ", ".join(missing_jams[:10])
        raise ValueError(f"Missing JAMS files for {len(missing_jams)} audio files: {preview}")
    return tracks


def natural_key(value: str) -> tuple:
    if value.isdigit():
        return (0, int(value))
    return (1, value)


if __name__ == "__main__":
    main()
