#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

if not os.environ.get("LOKY_MAX_CPU_COUNT"):
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)
if not os.environ.get("NUMBA_CACHE_DIR"):
    os.environ["NUMBA_CACHE_DIR"] = os.path.abspath(".numba-cache")

from tismir.diagnostics import run_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run token-similarity diagnostics for a checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--audio-embedding-root", default="data/embeddings/audio")
    parser.add_argument("--audio-encoder", required=True)
    parser.add_argument("--text-embedding-root", default="data/embeddings/text")
    parser.add_argument("--text-encoder", required=True)
    parser.add_argument("--audio-embedding-key", default="beat_sync")
    parser.add_argument("--namespace", default="segment_open")
    parser.add_argument("--output-dir", default="outputs/diagnostics/baseline")
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
    parser.add_argument("--max-plots", type=int, default=20)
    parser.add_argument("--audio-audio-max-frames", type=int, default=512)
    parser.add_argument("--smoothing-window", type=int, default=None)
    parser.add_argument("--smoothing-mode", choices=["mean", "median"], default=None)
    parser.add_argument("--decoder", choices=["argmax", "viterbi"], default=None)
    parser.add_argument("--transition-penalty", type=float, default=None)
    parser.add_argument(
        "--boundary-decoding",
        choices=["auto", "on", "off"],
        default="auto",
        help="Use boundary-head probabilities in the decoded diagnostic row.",
    )
    parser.add_argument("--boundary-weight", type=float, default=None)
    parser.add_argument("--boundary-eps", type=float, default=None)
    args = parser.parse_args()
    boundary_decoding = (
        None
        if args.boundary_decoding == "auto"
        else {
            "enabled": args.boundary_decoding == "on",
        }
    )

    run_diagnostics(
        checkpoint_path=args.checkpoint,
        manifest=args.manifest,
        audio_embedding_root=args.audio_embedding_root,
        audio_encoder=args.audio_encoder,
        text_embedding_root=args.text_embedding_root,
        text_encoder=args.text_encoder,
        audio_embedding_key=args.audio_embedding_key,
        namespace=args.namespace,
        output_dir=args.output_dir,
        device=args.device,
        limit=args.limit,
        candidate_label_strategy=args.candidate_label_strategy,
        annotation_processing=None if args.annotation_policy is None else {"policy": args.annotation_policy},
        beat_subsampling=None,
        max_plots=args.max_plots,
        audio_audio_max_frames=args.audio_audio_max_frames,
        smoothing_window=args.smoothing_window,
        smoothing_mode=args.smoothing_mode,
        decoder=args.decoder,
        transition_penalty=args.transition_penalty,
        boundary_decoding=boundary_decoding,
        boundary_weight=args.boundary_weight,
        boundary_eps=args.boundary_eps,
    )


if __name__ == "__main__":
    main()
