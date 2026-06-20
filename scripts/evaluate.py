#!/usr/bin/env python3
from __future__ import annotations

import argparse

from tismir.evaluation import evaluate_prediction_manifest, format_evaluation, save_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predicted music structure segmentations.")
    parser.add_argument("--manifest", required=True, help="Reference manifest with JAMS paths.")
    parser.add_argument("--predictions-root", required=True, help="Root containing {dataset}/{track_id}.jams predictions.")
    parser.add_argument("--namespace", default="segment_open")
    parser.add_argument("--no-trim", action="store_true")
    parser.add_argument(
        "--reference-annotation-policy",
        choices=[
            "keep",
            "merge",
            "enumerate_all_occurrences",
            "enumerate_consecutive_repeats",
        ],
        default=None,
    )
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    evaluation = evaluate_prediction_manifest(
        reference_manifest=args.manifest,
        predictions_root=args.predictions_root,
        namespace=args.namespace,
        trim=not args.no_trim,
        reference_annotation_processing=(
            None
            if args.reference_annotation_policy is None
            else {"policy": args.reference_annotation_policy}
        ),
    )
    print(format_evaluation(evaluation))
    if args.output_json is not None:
        save_evaluation(args.output_json, evaluation)


if __name__ == "__main__":
    main()
