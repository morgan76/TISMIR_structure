#!/usr/bin/env python3
from __future__ import annotations

import argparse

from tismir.evaluation import evaluate_prediction_manifest, format_evaluation, save_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predicted music structure segmentations.")
    parser.add_argument("--manifest", required=True, help="Reference manifest with JAMS paths.")
    parser.add_argument("--predictions-root", required=True, help="Root containing {dataset}/{track_id}.jams predictions.")
    parser.add_argument("--namespace", default="segment_open")
    parser.add_argument("--prediction-namespace", default=None)
    parser.add_argument("--no-trim", action="store_true")
    parser.add_argument("--audio-embedding-root", default=None)
    parser.add_argument("--audio-encoder", default=None)
    parser.add_argument("--text-embedding-root", default=None)
    parser.add_argument("--text-encoder", default=None)
    parser.add_argument("--audio-embedding-key", default="beat_sync")
    parser.add_argument(
        "--candidate-label-strategy",
        choices=["dataset_labels", "track_labels"],
        default="track_labels",
    )
    parser.add_argument(
        "--reference-annotation-policy",
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
    parser.add_argument(
        "--reference-annotation-selection",
        choices=["first", "richest_function"],
        default=None,
        help="Optional annotation-selection strategy for multi-annotator JAMS files.",
    )
    parser.add_argument(
        "--reference-track-filter-min-useful-labels",
        type=int,
        default=None,
        help="Exclude reference tracks with fewer useful labels after preprocessing.",
    )
    parser.add_argument(
        "--reference-track-filter-ignore-label",
        action="append",
        default=[],
        help="Label ignored by the useful-label reference filter; can be repeated.",
    )
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()
    annotation_processing = None
    if args.reference_annotation_policy is not None:
        annotation_processing = {"policy": args.reference_annotation_policy}
        if args.reference_annotation_selection is not None:
            annotation_processing["annotation_selection"] = args.reference_annotation_selection
    elif args.reference_annotation_selection is not None:
        raise ValueError("--reference-annotation-selection requires --reference-annotation-policy")

    reference_track_filter = None
    if args.reference_track_filter_min_useful_labels is not None:
        reference_track_filter = {
            "enabled": True,
            "min_useful_labels": args.reference_track_filter_min_useful_labels,
            "ignore_labels": args.reference_track_filter_ignore_label,
        }

    evaluation = evaluate_prediction_manifest(
        reference_manifest=args.manifest,
        predictions_root=args.predictions_root,
        namespace=args.namespace,
        prediction_namespace=args.prediction_namespace,
        trim=not args.no_trim,
        reference_annotation_processing=annotation_processing,
        audio_embedding_root=args.audio_embedding_root,
        audio_encoder=args.audio_encoder,
        text_embedding_root=args.text_embedding_root,
        text_encoder=args.text_encoder,
        audio_embedding_key=args.audio_embedding_key,
        candidate_label_strategy=args.candidate_label_strategy,
        reference_track_filter=reference_track_filter,
    )
    print(format_evaluation(evaluation))
    if args.output_json is not None:
        save_evaluation(args.output_json, evaluation)


if __name__ == "__main__":
    main()
