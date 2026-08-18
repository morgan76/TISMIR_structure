#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from tismir.data.manifest import save_manifest
from tismir.data.schemas import Track


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a manifest from a local SALAMI audio/reference folder."
    )
    parser.add_argument(
        "--layout",
        choices=["flat-wav", "nested-mp3"],
        default="nested-mp3",
        help=(
            "flat-wav: audio/<prefix><id>.wav + references/<id>.jams (original layout). "
            "nested-mp3: audio/<id>/audio.mp3 + jams/<id>.jams (Google-Drive dump)."
        ),
    )
    parser.add_argument("--root", default="data/raw/salami", type=Path)
    parser.add_argument("--dataset", default="salami")
    parser.add_argument("--output", default="data/manifests/salami.local.jsonl", type=Path)
    parser.add_argument("--audio-prefix", default="SALAMI_", help="flat-wav layout only.")
    parser.add_argument("--audio-ext", default=".wav", help="flat-wav layout only.")
    parser.add_argument("--jams-ext", default=".jams")
    args = parser.parse_args()

    if args.layout == "nested-mp3":
        tracks = build_salami_tracks_nested_mp3(
            root=args.root,
            dataset=args.dataset,
            jams_ext=args.jams_ext,
        )
    else:
        tracks = build_salami_tracks(
            root=args.root,
            dataset=args.dataset,
            audio_prefix=args.audio_prefix,
            audio_ext=args.audio_ext,
            jams_ext=args.jams_ext,
        )
    save_manifest(args.output, tracks)
    print(f"Wrote {len(tracks)} tracks to {args.output}")


def build_salami_tracks_nested_mp3(
    root: Path,
    dataset: str,
    jams_ext: str,
) -> list[Track]:
    """Build tracks for the nested-mp3 SALAMI dump.

    Layout: ``<root>/audio/<id>/audio.mp3`` and ``<root>/jams/<id>.jams``. A
    track is included only when both the jams file and its audio exist (some
    audio dirs have no annotation, and vice versa).
    """

    jams_dir = root / "jams"
    audio_dir = root / "audio"
    tracks: list[Track] = []
    missing_audio = 0
    for jams_path in sorted(jams_dir.glob(f"*{jams_ext}"), key=lambda path: _natural_key(path.stem)):
        salami_id = jams_path.stem
        audio_path = audio_dir / salami_id / "audio.mp3"
        if not audio_path.exists():
            missing_audio += 1
            continue
        tracks.append(
            Track(
                track_id=salami_id,
                audio_path=audio_path.resolve(),
                jams_path=jams_path.resolve(),
                dataset=dataset,
            )
        )
    if not tracks:
        raise ValueError(f"No matched SALAMI tracks found under {root} (nested-mp3 layout)")
    if missing_audio:
        print(f"Skipped {missing_audio} jams with no matching audio/<id>/audio.mp3")
    return tracks


def build_salami_tracks(
    root: Path,
    dataset: str,
    audio_prefix: str,
    audio_ext: str,
    jams_ext: str,
) -> list[Track]:
    audio_dir = root / "audio"
    references_dir = root / "references"
    tracks: list[Track] = []
    for jams_path in sorted(references_dir.glob(f"*{jams_ext}"), key=lambda path: _natural_key(path.stem)):
        salami_id = jams_path.stem
        audio_path = audio_dir / f"{audio_prefix}{salami_id}{audio_ext}"
        if not audio_path.exists():
            continue
        tracks.append(
            Track(
                track_id=salami_id,
                audio_path=audio_path.resolve(),
                jams_path=jams_path.resolve(),
                dataset=dataset,
            )
        )
    if not tracks:
        raise ValueError(f"No matched SALAMI tracks found under {root}")
    return tracks


def _natural_key(value: str) -> tuple[int, int | str]:
    if value.isdigit():
        return (0, int(value))
    return (1, value)


if __name__ == "__main__":
    main()
