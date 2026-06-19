#!/usr/bin/env python3
from __future__ import annotations

import argparse

from tismir.io import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute dense and beat-synchronous audio embeddings.")
    parser.add_argument("--config", default="configs/preprocessing/audio.yaml")
    args = parser.parse_args()

    config = load_yaml(args.config)
    raise NotImplementedError(
        "Audio preprocessing scaffold is ready. Register an audio encoder and beat tracker "
        f"to run this config: {config.get('audio_encoder', {}).get('name')}"
    )


if __name__ == "__main__":
    main()
