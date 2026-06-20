#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from tismir.data.manifest import load_manifest
from tismir.encoders.audio import audio_encoders
from tismir.encoders.beats import beat_trackers
from tismir.io import load_yaml
from tismir.preprocessing.audio import preprocess_track_audio_with_backends, result_to_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute dense and beat-synchronous audio embeddings.")
    parser.add_argument("--config", default="configs/preprocessing/audio.yaml")
    parser.add_argument("--manifest", default=None, help="Override dataset_manifest from the config.")
    parser.add_argument("--output-root", default=None, help="Override output_root from the config.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N tracks.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip tracks with an existing beat_sync.npy.")
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
    audio_encoder = audio_encoders.build(audio_name, **audio_config)
    beat_tracker = beat_trackers.build(beat_name, **beat_config)

    results = []
    skipped = 0
    for index, track in enumerate(tracks, start=1):
        if args.skip_existing and _has_embedding(output_root, audio_name, track):
            skipped += 1
            print(f"[{index}/{len(tracks)}] {track.track_id}: skipped existing")
            continue
        result = preprocess_track_audio_with_backends(
            track=track,
            output_root=output_root,
            audio_encoder_name=audio_name,
            audio_encoder=audio_encoder,
            beat_tracker=beat_tracker,
            pooling=config.get("pooling", {}),
        )
        results.append(result_to_dict(result))
        print(
            f"[{index}/{len(tracks)}] {track.track_id}: "
            f"dense={result.dense_shape}, beat_sync={result.beat_sync_shape}"
        )

    print(f"Processed {len(results)} tracks into {output_root}; skipped {skipped}")


def _has_embedding(output_root: str, audio_name: str, track) -> bool:
    output_dir = Path(output_root) / audio_name / track.dataset / track.track_id
    return (output_dir / "beat_sync.npy").exists() and (output_dir / "beats.npy").exists()


if __name__ == "__main__":
    main()
