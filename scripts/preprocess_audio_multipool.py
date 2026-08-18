#!/usr/bin/env python3
"""Precompute beat-synchronous embeddings for several pooling methods in one pass.

The dense MERT frames and beat tracking are identical across pooling methods, so
this script encodes each track once and writes every pooling variant from the
shared frames. That makes an N-method sweep cost ~1 encode instead of N.

Pass the per-method preprocessing configs; each contributes its ``output_root``
and ``pooling`` block. The audio encoder and beat tracker are taken from the
first config (they must match across configs).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tismir.data.manifest import load_manifest
from tismir.encoders.audio import audio_encoders
from tismir.encoders.beats import beat_trackers
from tismir.io import load_yaml
from tismir.preprocessing.audio import preprocess_track_audio_multi_pool

DEFAULT_CONFIGS = [
    "configs/preprocessing/audio_mert_madmom_pool_mean.yaml",
    "configs/preprocessing/audio_mert_madmom_pool_max.yaml",
    "configs/preprocessing/audio_mert_madmom_pool_energy.yaml",
    "configs/preprocessing/audio_mert_madmom_pool_multistat.yaml",
    "configs/preprocessing/audio_mert_madmom_pool_first.yaml",
    "configs/preprocessing/audio_mert_madmom_pool_last.yaml",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=DEFAULT_CONFIGS,
        help="Per-method preprocessing configs (each provides output_root + pooling).",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Override dataset_manifest (all configs must otherwise agree on it).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N tracks.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a track when every variant's beat_sync.npy already exists.",
    )
    args = parser.parse_args()

    configs = [load_yaml(path) for path in args.configs]
    if not configs:
        raise ValueError("At least one config is required.")

    base = configs[0]
    manifest = args.manifest or base["dataset_manifest"]

    audio_config = dict(base.get("audio_encoder", {}))
    beat_config = dict(base.get("beat_tracker", {}))
    audio_name = audio_config.pop("name")
    beat_name = beat_config.pop("name")
    audio_encoder = audio_encoders.build(audio_name, **audio_config)
    beat_tracker = beat_trackers.build(beat_name, **beat_config)

    variants = [(cfg["output_root"], cfg.get("pooling", {})) for cfg in configs]
    variant_labels = [
        f"{cfg.get('pooling', {}).get('method', 'mean')}->{cfg['output_root']}" for cfg in configs
    ]
    print("Pooling variants:")
    for label in variant_labels:
        print(f"  {label}")

    tracks = load_manifest(manifest)
    if args.limit is not None:
        tracks = tracks[: args.limit]

    processed = 0
    skipped = 0
    for index, track in enumerate(tracks, start=1):
        if args.skip_existing and _all_variants_exist(variants, audio_name, track):
            skipped += 1
            print(f"[{index}/{len(tracks)}] {track.track_id}: skipped existing")
            continue
        try:
            results = preprocess_track_audio_multi_pool(
                track=track,
                variants=variants,
                audio_encoder_name=audio_name,
                audio_encoder=audio_encoder,
                beat_tracker=beat_tracker,
            )
        except Exception as exc:  # keep the long sweep going; report and move on
            print(f"[{index}/{len(tracks)}] {track.track_id}: ERROR {type(exc).__name__}: {exc}")
            continue
        processed += 1
        shapes = ", ".join(f"{Path(r.output_dir).parts[-3]}={r.beat_sync_shape}" for r in results)
        print(f"[{index}/{len(tracks)}] {track.track_id}: {shapes}")

    print(f"Processed {processed} tracks across {len(variants)} variants; skipped {skipped}")


def _all_variants_exist(variants, audio_name: str, track) -> bool:
    for output_root, _pooling in variants:
        output_dir = Path(output_root) / audio_name / track.dataset / track.track_id
        if not (output_dir / "beat_sync.npy").exists():
            return False
    return True


if __name__ == "__main__":
    main()
