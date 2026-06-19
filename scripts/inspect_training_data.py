#!/usr/bin/env python3
from __future__ import annotations

import argparse

from tismir.io import load_yaml
from tismir.training.data import StructureEmbeddingDataset, collate_training_examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect precomputed training examples.")
    parser.add_argument("--config", default="configs/train/baseline.yaml")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    config = load_yaml(args.config)
    dataset = StructureEmbeddingDataset(**config["data"])
    print(f"tracks={len(dataset)}")

    examples = [dataset[index] for index in range(min(args.limit, len(dataset)))]
    for example in examples:
        print(
            f"{example.track_id}: audio={example.audio.shape}, text={example.text.shape}, "
            f"targets={example.targets.shape}, labels={len(example.labels)}"
        )

    batch = collate_training_examples(examples)
    print(
        f"batch: audio={tuple(batch['audio'].shape)}, text={tuple(batch['text'].shape)}, "
        f"targets={tuple(batch['targets'].shape)}, mask={tuple(batch['mask'].shape)}"
    )


if __name__ == "__main__":
    main()
