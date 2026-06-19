#!/usr/bin/env python3
from __future__ import annotations

import argparse

from tismir.io import load_yaml
from tismir.training import train_projection_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a text-conditioned structure model.")
    parser.add_argument("--config", default="configs/train/baseline.yaml")
    args = parser.parse_args()

    config = load_yaml(args.config)
    metrics = train_projection_baseline(config)
    print(f"saved checkpoint: {metrics['checkpoint']}")
    if metrics.get("best_checkpoint") is not None:
        print(f"saved best checkpoint: {metrics['best_checkpoint']}")
        print(f"best validation loss: {metrics['best_val_loss']:.6f}")
    print(f"final loss: {metrics['final_loss']:.6f}")


if __name__ == "__main__":
    main()
