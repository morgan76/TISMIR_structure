#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from tismir.data.manifest import load_manifest
from tismir.data.splits import save_split_manifests, split_tracks


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic train/val/test manifest splits.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("data/manifests"), type=Path)
    parser.add_argument("--name", required=True, help="Output basename prefix, e.g. rwc_pop_10.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--suffix", default=".local")
    args = parser.parse_args()

    tracks = load_manifest(args.manifest)
    splits = split_tracks(
        tracks,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )
    paths = save_split_manifests(splits, output_dir=args.output_dir, name=args.name, suffix=args.suffix)
    for split, path in paths.items():
        print(f"{split}: {len(splits[split])} tracks -> {path}")


if __name__ == "__main__":
    main()
