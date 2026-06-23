#!/usr/bin/env python3
"""Source Harmonix Set audio from the YouTube URLs shipped with the dataset.

The Harmonix Set does not distribute audio; it provides
``dataset/youtube_urls.csv`` mapping each track id to a YouTube video. This
script downloads those videos as audio files (via ``yt-dlp`` + ``ffmpeg``),
named by the Harmonix track id so they pair 1:1 with the JAMS annotations.

Sourced YouTube audio is not frame-aligned to the annotations. Use
``dataset/youtube_alignment_scores.csv`` (``--min-score``) to keep only
well-matched tracks, and the repo's "Audio Alignment.ipynb" (DTW) to align the
audio to the annotation timeline before training.

Requires ``yt-dlp`` (``pip install yt-dlp``) and ``ffmpeg`` on PATH.

Examples
--------
    # Smoke test: two tracks
    python scripts/source_harmonix_audio.py \
        --harmonix-dir /scratch/ick/music_structure/harmonix \
        --dest /scratch/ick/music_structure/harmonix/audio \
        --limit 2

    # Only download tracks whose YouTube audio aligns well (score >= 0.95)
    python scripts/source_harmonix_audio.py \
        --harmonix-dir /scratch/ick/music_structure/harmonix \
        --dest /scratch/ick/music_structure/harmonix/audio \
        --min-score 0.95
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Harmonix Set audio from its YouTube URLs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--harmonix-dir",
        type=Path,
        required=True,
        help="Path to the cloned harmonix repo (contains dataset/youtube_urls.csv).",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        required=True,
        help="Output directory for audio files (named <track_id>.<ext>).",
    )
    parser.add_argument("--audio-format", default="wav", help="ffmpeg audio format/extension.")
    parser.add_argument(
        "--audio-quality",
        default="0",
        help="yt-dlp --audio-quality (0=best for lossy formats; ignored for wav).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Keep only tracks with youtube_alignment_scores.csv score >= this.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only attempt the first N tracks.")
    parser.add_argument(
        "--cookies",
        type=Path,
        default=None,
        help="Netscape-format cookies.txt exported from a logged-in browser. "
        "Usually REQUIRED: YouTube bot-blocks datacenter/server IPs without it.",
    )
    parser.add_argument(
        "--cookies-from-browser",
        default=None,
        help="Pull cookies directly from a local browser (e.g. 'firefox', 'chrome'). "
        "Only works where that browser profile exists.",
    )
    parser.add_argument(
        "--sleep-requests",
        type=float,
        default=1.0,
        help="Seconds to sleep between requests (politeness / throttling avoidance).",
    )
    parser.add_argument(
        "--remote-components",
        default="ejs:github",
        help="yt-dlp --remote-components value. Required for current YouTube: it fetches "
        "the EJS challenge-solver script. Needs a JS runtime (e.g. Deno) on PATH. "
        "Pass empty string to disable.",
    )
    parser.add_argument(
        "--urls-csv",
        type=Path,
        default=None,
        help="Override path to youtube_urls.csv (default: <harmonix-dir>/dataset/youtube_urls.csv).",
    )
    parser.add_argument(
        "--scores-csv",
        type=Path,
        default=None,
        help="Override path to youtube_alignment_scores.csv.",
    )
    args = parser.parse_args()

    if shutil.which("yt-dlp") is None:
        sys.exit("yt-dlp not found on PATH. Install it with: pip install yt-dlp")
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found on PATH (required to extract audio).")

    urls_csv = args.urls_csv or args.harmonix_dir / "dataset" / "youtube_urls.csv"
    scores_csv = args.scores_csv or args.harmonix_dir / "dataset" / "youtube_alignment_scores.csv"
    if not urls_csv.is_file():
        sys.exit(f"YouTube URL list not found: {urls_csv}")

    tracks = read_urls(urls_csv)
    if args.min_score is not None:
        scores = read_scores(scores_csv)
        kept = [(fid, url) for fid, url in tracks if scores.get(fid, -1.0) >= args.min_score]
        print(f"Alignment filter (score >= {args.min_score}): {len(kept)}/{len(tracks)} tracks")
        tracks = kept
    if args.limit is not None:
        tracks = tracks[: args.limit]

    args.dest.mkdir(parents=True, exist_ok=True)
    archive = args.dest / ".yt-dlp-archive.txt"

    ok, skipped, failed = 0, 0, []
    for i, (file_id, url) in enumerate(tracks, 1):
        out = args.dest / f"{file_id}.{args.audio_format}"
        if out.exists():
            skipped += 1
            continue
        print(f"[{i}/{len(tracks)}] {file_id}  {url}")
        if download_one(url, file_id, args.dest, args.audio_format, args.audio_quality, archive, args):
            ok += 1
        else:
            failed.append((file_id, url))

    print(f"\nDone. downloaded={ok} skipped_existing={skipped} failed={len(failed)} "
          f"-> {args.dest}")
    if failed:
        report = args.dest / "failed_downloads.csv"
        with report.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["File", "URL"])
            writer.writerows(failed)
        print(f"Failed track ids written to {report}")


def read_urls(path: Path) -> list[tuple[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return [(row["File"], row["URL"]) for row in reader if row.get("URL")]


def read_scores(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return {row["File"]: float(row["score"]) for row in reader if row.get("score")}


def download_one(
    url: str,
    file_id: str,
    dest: Path,
    audio_format: str,
    audio_quality: str,
    archive: Path,
    args: argparse.Namespace,
) -> bool:
    cmd = [
        "yt-dlp",
        url,
        "-x",
        "--audio-format", audio_format,
        "--audio-quality", audio_quality,
        "--no-playlist",
        "--no-overwrites",
        "--download-archive", str(archive),
        "--retries", "3",
        "--sleep-requests", str(args.sleep_requests),
        "--quiet",
        "--no-warnings",
        "-o", str(dest / f"{file_id}.%(ext)s"),
    ]
    if args.remote_components:
        cmd += ["--remote-components", args.remote_components]
    if args.cookies is not None:
        cmd += ["--cookies", str(args.cookies)]
    if args.cookies_from_browser is not None:
        cmd += ["--cookies-from-browser", args.cookies_from_browser]
    result = subprocess.run(cmd)
    return result.returncode == 0


if __name__ == "__main__":
    main()
