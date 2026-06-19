#!/usr/bin/env python3
from __future__ import annotations

import argparse

from tismir.data.manifest import load_manifest
from tismir.data.validation import format_summary, summarize_validation, validate_track, write_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a dataset manifest and summarize JAMS structure labels.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--namespace", default="segment_open")
    parser.add_argument("--summary-json", default=None)
    args = parser.parse_args()

    tracks = load_manifest(args.manifest)
    results = [validate_track(track, namespace=args.namespace) for track in tracks]
    summary = summarize_validation(results)

    print(format_summary(summary))
    if args.summary_json is not None:
        write_summary(args.summary_json, summary)

    if summary.num_invalid_tracks:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
