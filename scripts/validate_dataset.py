#!/usr/bin/env python3
from __future__ import annotations

import argparse

from tismir.data.manifest import load_manifest
from tismir.data.validation import format_summary, summarize_validation, validate_track, write_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a dataset manifest and summarize JAMS structure labels.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--namespace", default="segment_open")
    parser.add_argument(
        "--annotation-policy",
        choices=[
            "keep",
            "merge",
            "base_labels",
            "enumerate_all_occurrences",
            "enumerate_base_occurrences",
            "enumerate_consecutive_repeats",
            "salami_function_merge",
            "salami_function_occurrences",
            "salami_function_projected_lower",
        ],
        default=None,
    )
    parser.add_argument("--summary-json", default=None)
    args = parser.parse_args()

    tracks = load_manifest(args.manifest)
    annotation_processing = (
        None
        if args.annotation_policy is None
        else {"policy": args.annotation_policy}
    )
    results = [
        validate_track(
            track,
            namespace=args.namespace,
            annotation_processing=annotation_processing,
        )
        for track in tracks
    ]
    summary = summarize_validation(results)

    print(format_summary(summary))
    if args.summary_json is not None:
        write_summary(args.summary_json, summary)

    if summary.num_invalid_tracks:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
