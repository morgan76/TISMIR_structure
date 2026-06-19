#!/usr/bin/env python3
from __future__ import annotations

import argparse

from tismir.io import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute text label embeddings.")
    parser.add_argument("--config", default="configs/preprocessing/text.yaml")
    args = parser.parse_args()

    config = load_yaml(args.config)
    raise NotImplementedError(
        "Text preprocessing scaffold is ready. Register a text encoder "
        f"to run this config: {config.get('text_encoder', {}).get('name')}"
    )


if __name__ == "__main__":
    main()
