#!/usr/bin/env python3
"""Fetch the Internet Archive portion of SALAMI audio.

The 2011 URLs in metadata/id_index_internetarchive.csv have partially rotted.
For each row this script:
  1. tries the stored URL directly (https),
  2. on failure queries https://archive.org/metadata/<item> and looks for the
     stored filename, or any audio file sharing the track token (e.g. "d1t03"),
     preferring VBR MP3 > MP3 > OGG > original lossless,
  3. saves to <output-dir>/<SONG_ID>.<ext> and appends to a coverage CSV.

Usage:
  python scripts/fetch_salami_internetarchive.py \
    --index /scratch/ick/music_structure/salami/metadata/id_index_internetarchive.csv \
    --output-dir /scratch/ick/music_structure/salami/audio/internetarchive \
    --report /scratch/ick/music_structure/salami/audio/ia_coverage.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

UA = {"User-Agent": "salami-audio-fetch/1.0 (research; MSA benchmark reproduction)"}
AUDIO_EXTS = (".mp3", ".ogg", ".flac", ".shn", ".wav")
TRACK_TOKEN_RE = re.compile(r"(d\d+t\d+|(?<![a-z0-9])t\d+)", re.IGNORECASE)


def http_get(url: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def pick_candidate(files: list[dict], stored_name: str, token: str | None) -> str | None:
    names = [f.get("name", "") for f in files]
    if stored_name in names:
        return stored_name
    audio = [n for n in names if n.lower().endswith(AUDIO_EXTS)]
    if token:
        # Require a non-digit after the token so "t1" cannot match "t10".
        token_re = re.compile(re.escape(token) + r"(?!\d)", re.IGNORECASE)
        tokened = [n for n in audio if token_re.search(n)]
        if tokened:
            for pref in ("_vbr.mp3", ".mp3", ".ogg"):
                for name in tokened:
                    if name.lower().endswith(pref):
                        return name
            return tokened[0]
    return None


def fetch_row(row: dict, output_dir: Path) -> tuple[str, str]:
    """Return (status, detail). status in {ok, ok_via_metadata, missing_item, no_match, error}."""
    url = row["URL"].replace("http://", "https://")
    parsed = urllib.parse.urlparse(url)
    parts = parsed.path.split("/")
    item = parts[2] if len(parts) > 2 else ""
    stored_name = urllib.parse.unquote(parts[-1])
    token_match = TRACK_TOKEN_RE.search(stored_name)
    token = token_match.group(1) if token_match else None
    song_id = row["SONG_ID"]

    def save(data: bytes, name: str) -> str:
        ext = Path(name).suffix.lower() or ".mp3"
        target = output_dir / f"{song_id}{ext}"
        target.write_bytes(data)
        return str(target)

    try:
        return "ok", save(http_get(url), stored_name)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
        pass

    try:
        meta = json.loads(http_get(f"https://archive.org/metadata/{item}", timeout=30))
    except Exception as exc:  # noqa: BLE001 - report and continue
        return "error", f"metadata fetch failed: {exc}"
    files = meta.get("files") or []
    if not files:
        return "missing_item", item
    candidate = pick_candidate(files, stored_name, token)
    if candidate is None:
        return "no_match", f"{item}: {len(files)} files, none match {token or stored_name}"
    server = meta.get("server", "archive.org")
    dir_ = meta.get("dir", f"/download/{item}")
    candidate_url = f"https://{server}{dir_}/{urllib.parse.quote(candidate)}"
    try:
        return "ok_via_metadata", save(http_get(candidate_url), candidate)
    except Exception as exc:  # noqa: BLE001
        return "error", f"candidate download failed: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.5, help="Delay between rows (be polite to IA).")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(args.index)))
    if args.limit:
        rows = rows[: args.limit]

    counts: dict[str, int] = {}
    with open(args.report, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["SONG_ID", "status", "detail"])
        for index, row in enumerate(rows, start=1):
            song_id = row["SONG_ID"]
            existing = list(output_dir.glob(f"{song_id}.*"))
            if existing:
                status, detail = "ok", str(existing[0])
            else:
                status, detail = fetch_row(row, output_dir)
                time.sleep(args.sleep)
            counts[status] = counts.get(status, 0) + 1
            writer.writerow([song_id, status, detail])
            handle.flush()
            print(f"[{index}/{len(rows)}] {song_id}: {status}")

    print("coverage:", json.dumps(counts))


if __name__ == "__main__":
    main()
