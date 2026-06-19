#!/usr/bin/env python3
from __future__ import annotations

import argparse

from tismir.data.manifest import load_manifest
from tismir.io import load_yaml
from tismir.preprocessing.audio import preprocess_track_audio, result_to_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute dense and beat-synchronous audio embeddings.")
    parser.add_argument("--config", default="configs/preprocessing/audio.yaml")
    parser.add_argument("--manifest", default=None, help="Override dataset_manifest from the config.")
    parser.add_argument("--output-root", default=None, help="Override output_root from the config.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N tracks.")
    args = parser.parse_args()

    config = load_yaml(args.config)
    manifest = args.manifest or config["dataset_manifest"]
    output_root = args.output_root or config["output_root"]
    tracks = load_manifest(manifest)
    if args.limit is not None:
        tracks = tracks[: args.limit]

    audio_config = dict(config.get("audio_encoder", {}))
    beat_config = dict(config.get("beat_tracker", {}))
    audio_name = audio_config.pop("name")
    beat_name = beat_config.pop("name")

    results = []
    for index, track in enumerate(tracks, start=1):
        result = preprocess_track_audio(
            track=track,
            output_root=output_root,
            audio_encoder_name=audio_name,
            audio_encoder_params=audio_config,
            beat_tracker_name=beat_name,
            beat_tracker_params=beat_config,
            pooling=config.get("pooling", {}),
        )
        results.append(result_to_dict(result))
        print(
            f"[{index}/{len(tracks)}] {track.track_id}: "
            f"dense={result.dense_shape}, beat_sync={result.beat_sync_shape}"
        )

    print(f"Processed {len(results)} tracks into {output_root}")


if __name__ == "__main__":
    main()
