#!/usr/bin/env python3
from __future__ import annotations

import argparse

from tismir.io import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a text-conditioned structure model.")
    parser.add_argument("--config", default="configs/train/baseline.yaml")
    args = parser.parse_args()

    config = load_yaml(args.config)
    raise NotImplementedError(f"Training loop scaffold is ready for config seed {config.get('seed')}.")


if __name__ == "__main__":
    main()
