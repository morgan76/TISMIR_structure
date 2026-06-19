from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from tismir.losses import frame_label_cross_entropy
from tismir.models import build_model
from tismir.training.data import StructureEmbeddingDataset, collate_training_examples


def train_projection_baseline(config: dict[str, Any]) -> dict[str, Any]:
    """Train the projection baseline from a config dictionary."""

    torch = _require_torch()
    seed = int(config.get("seed", 0))
    _set_seed(seed, torch)
    device = _resolve_device(config.get("device", "auto"), torch)

    dataset = StructureEmbeddingDataset(**config["data"])
    validation_dataset = _build_validation_dataset(config)
    first = dataset[0]
    model = build_model(config.get("model", {}), first.audio.shape[1], first.text.shape[1]).to(device)

    opt_config = config.get("optimization", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_config.get("learning_rate", 1e-4)),
        weight_decay=float(opt_config.get("weight_decay", 1e-5)),
    )

    batch_size = int(opt_config.get("batch_size", 1))
    max_epochs = int(opt_config.get("max_epochs", 1))
    ignore_index = int(config.get("data", {}).get("ignore_index", -100))
    history: list[dict[str, float]] = []
    best_val_loss: float | None = None
    best_state_dict = None

    for epoch in range(max_epochs):
        indices = list(range(len(dataset)))
        if opt_config.get("shuffle", True):
            random.shuffle(indices)

        epoch_loss = 0.0
        epoch_items = 0
        model.train()
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            examples = [dataset[index] for index in batch_indices]
            batch = collate_training_examples(examples)
            audio = batch["audio"].to(device)
            text = batch["text"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(audio, text, audio_mask=mask)
            loss = frame_label_cross_entropy(logits, targets, ignore_index=ignore_index)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach().cpu()) * len(examples)
            epoch_items += len(examples)

        mean_loss = epoch_loss / max(epoch_items, 1)
        record = {"epoch": float(epoch), "loss": mean_loss}
        message = f"epoch={epoch + 1}/{max_epochs} loss={mean_loss:.6f}"
        if validation_dataset is not None:
            val_loss = _evaluate_loss(
                model=model,
                dataset=validation_dataset,
                batch_size=batch_size,
                device=device,
                ignore_index=ignore_index,
            )
            record["val_loss"] = val_loss
            message += f" val_loss={val_loss:.6f}"
            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
        history.append(record)
        print(message)

    output_dir = Path(config.get("output_dir", "outputs/train/baseline"))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "labels": first.labels,
        "audio_dim": first.audio.shape[1],
        "text_dim": first.text.shape[1],
    }
    torch.save(checkpoint, checkpoint_path)
    best_checkpoint_path = None
    if best_state_dict is not None:
        best_checkpoint_path = output_dir / "best_checkpoint.pt"
        torch.save({**checkpoint, "model_state_dict": best_state_dict}, best_checkpoint_path)
    metrics = {
        "checkpoint": str(checkpoint_path),
        "best_checkpoint": None if best_checkpoint_path is None else str(best_checkpoint_path),
        "final_loss": history[-1]["loss"] if history else None,
        "final_val_loss": history[-1].get("val_loss") if history else None,
        "best_val_loss": best_val_loss,
        "history": history,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    return metrics


def _build_validation_dataset(config: dict[str, Any]) -> StructureEmbeddingDataset | None:
    validation_config = config.get("validation")
    if validation_config is None:
        return None
    data_config = dict(config["data"])
    data_config.update(validation_config.get("data", {}))
    if validation_config.get("manifest") is not None:
        data_config["manifest"] = validation_config["manifest"]
    return StructureEmbeddingDataset(**data_config)


def _evaluate_loss(
    model,
    dataset: StructureEmbeddingDataset,
    batch_size: int,
    device,
    ignore_index: int,
) -> float:
    torch = _require_torch()
    model.eval()
    total_loss = 0.0
    total_items = 0
    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            examples = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
            batch = collate_training_examples(examples)
            audio = batch["audio"].to(device)
            text = batch["text"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)
            logits = model(audio, text, audio_mask=mask)
            loss = frame_label_cross_entropy(logits, targets, ignore_index=ignore_index)
            total_loss += float(loss.detach().cpu()) * len(examples)
            total_items += len(examples)
    model.train()
    return total_loss / max(total_items, 1)


def _resolve_device(device: str, torch):
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _set_seed(seed: int, torch) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install torch to train models.") from exc
    return torch
