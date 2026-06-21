#!/usr/bin/env python3
from __future__ import annotations

import argparse

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
            "enumerate_all_occurrences",
            "enumerate_base_occurrences",
            "enumerate_consecutive_repeats",
        ],
        default=None,
        help="Override checkpoint annotation_processing.policy.",
    )
    parser.add_argument("--max-plots", type=int, default=20)
    parser.add_argument("--audio-audio-max-frames", type=int, default=512)
    args = parser.parse_args()

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
        max_plots=args.max_plots,
        audio_audio_max_frames=args.audio_audio_max_frames,
    )


if __name__ == "__main__":
    main()
