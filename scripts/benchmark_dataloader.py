#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from typing import Any

from tismir.io import load_yaml
from tismir.training.data import StructureEmbeddingDataset, collate_training_examples
from tismir.training.loop import _EpochIndexSampler, _build_data_loader, _data_loader_config, _require_torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark dataset loading/collation throughput.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--to-device", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    torch = _require_torch()
    dataset = StructureEmbeddingDataset(**config["data"])
    opt_config = dict(config.get("optimization", {}))
    opt_config["num_workers"] = args.num_workers
    opt_config["persistent_workers"] = args.num_workers > 0
    opt_config["prefetch_factor"] = 2 if args.num_workers > 0 else None
    data_loader_config = _data_loader_config(opt_config, seed=int(config.get("seed", 0)))
    batch_size = int(opt_config.get("batch_size", 1))
    device = torch.device(args.device)
    sampler = None
    loader = None
    if args.num_workers > 0:
        sampler = _EpochIndexSampler(
            len(dataset),
            seed=int(config.get("seed", 0)),
            shuffle=bool(opt_config.get("shuffle", True)),
        )
        loader = _build_data_loader(
            torch=torch,
            dataset=dataset,
            batch_size=batch_size,
            data_loader_config=data_loader_config,
            sampler=sampler,
        )

    total_batches = 0
    total_examples = 0
    start_time = time.perf_counter()
    for epoch in range(args.epochs):
        dataset.set_epoch(epoch)
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch_batches, epoch_examples = _run_epoch(
            dataset=dataset,
            batch_size=batch_size,
            loader=loader,
            epoch=epoch,
            seed=int(config.get("seed", 0)),
            shuffle=bool(opt_config.get("shuffle", True)),
            max_batches=args.max_batches,
            device=device,
            to_device=bool(args.to_device),
        )
        total_batches += epoch_batches
        total_examples += epoch_examples
    elapsed = time.perf_counter() - start_time
    print(f"num_workers={args.num_workers}")
    print(f"epochs={args.epochs}")
    print(f"batches={total_batches}")
    print(f"examples={total_examples}")
    print(f"elapsed={elapsed:.3f}s")
    print(f"batches_per_second={total_batches / max(elapsed, 1e-9):.3f}")
    print(f"examples_per_second={total_examples / max(elapsed, 1e-9):.3f}")


def _run_epoch(
    *,
    dataset: StructureEmbeddingDataset,
    batch_size: int,
    loader,
    epoch: int,
    seed: int,
    shuffle: bool,
    max_batches: int | None,
    device,
    to_device: bool,
) -> tuple[int, int]:
    if loader is not None:
        iterator = iter(loader)
    else:
        import random

        indices = list(range(len(dataset)))
        if shuffle:
            random.Random(seed + epoch * 1_000_003).shuffle(indices)
        starts = list(range(0, len(indices), batch_size))
        iterator = (
            collate_training_examples(
                [dataset[index] for index in indices[start : start + batch_size]]
            )
            for start in starts
        )

    batches = 0
    examples = 0
    for batch in iterator:
        if to_device:
            batch["audio"].to(device)
            batch["text"].to(device)
            batch["targets"].to(device)
            batch["base_targets"].to(device)
            batch["segment_targets"].to(device)
            batch["mask"].to(device)
        batches += 1
        examples += len(batch["track_ids"])
        if max_batches is not None and batches >= max_batches:
            break
    return batches, examples


if __name__ == "__main__":
    main()
