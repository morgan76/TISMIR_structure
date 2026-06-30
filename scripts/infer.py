#!/usr/bin/env python3
from __future__ import annotations

import argparse

from tismir.inference import run_baseline_inference


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline label-set-conditioned structure inference.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--audio-embedding-root", default="data/embeddings/audio")
    parser.add_argument("--audio-encoder", required=True)
    parser.add_argument("--text-embedding-root", default="data/embeddings/text")
    parser.add_argument("--text-encoder", required=True)
    parser.add_argument("--audio-embedding-key", default="beat_sync")
    parser.add_argument("--namespace", default="segment_open")
    parser.add_argument("--prediction-namespace", default=None)
    parser.add_argument("--output-dir", default="outputs/infer/baseline")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--candidate-label-strategy",
        choices=["track_labels", "dataset_labels"],
        default="track_labels",
    )
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
        help="Override checkpoint annotation_processing.policy.",
    )
    parser.add_argument("--smoothing-window", type=int, default=7)
    parser.add_argument("--smoothing-mode", choices=["mean", "median"], default="mean")
    parser.add_argument("--decoder", choices=["argmax", "viterbi"], default="viterbi")
    parser.add_argument("--transition-penalty", type=float, default=8.0)
    parser.add_argument("--min-segment-duration", type=float, default=0.0)
    parser.add_argument(
        "--boundary-decoding",
        choices=["auto", "on", "off"],
        default="auto",
        help="Use boundary-head probabilities as a Viterbi transition prior when available.",
    )
    parser.add_argument("--boundary-weight", type=float, default=1.0)
    parser.add_argument("--boundary-eps", type=float, default=1e-4)
    args = parser.parse_args()
    boundary_decoding = (
        "auto"
        if args.boundary_decoding == "auto"
        else {
            "enabled": args.boundary_decoding == "on",
            "weight": args.boundary_weight,
            "eps": args.boundary_eps,
        }
    )

    run_baseline_inference(
        checkpoint_path=args.checkpoint,
        manifest=args.manifest,
        audio_embedding_root=args.audio_embedding_root,
        audio_encoder=args.audio_encoder,
        text_embedding_root=args.text_embedding_root,
        text_encoder=args.text_encoder,
        audio_embedding_key=args.audio_embedding_key,
        namespace=args.namespace,
        prediction_namespace=args.prediction_namespace,
        output_dir=args.output_dir,
        device=args.device,
        limit=args.limit,
        candidate_label_strategy=args.candidate_label_strategy,
        annotation_processing=None if args.annotation_policy is None else {"policy": args.annotation_policy},
        smoothing_window=args.smoothing_window,
        smoothing_mode=args.smoothing_mode,
        decoder=args.decoder,
        transition_penalty=args.transition_penalty,
        min_segment_duration=args.min_segment_duration,
        boundary_decoding=boundary_decoding,
    )


if __name__ == "__main__":
    main()
