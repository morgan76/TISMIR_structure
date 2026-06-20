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
    output_dir = Path(config.get("output_dir", "outputs/train/baseline"))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    best_checkpoint_path = output_dir / "best_checkpoint.pt" if validation_dataset is not None else None

    opt_config = config.get("optimization", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_config.get("learning_rate", 1e-4)),
        weight_decay=float(opt_config.get("weight_decay", 1e-5)),
    )
    lr_scheduler = _build_lr_scheduler(
        opt_config.get("lr_scheduler"),
        optimizer=optimizer,
        torch=torch,
        has_validation=validation_dataset is not None,
    )

    batch_size = int(opt_config.get("batch_size", 1))
    if batch_size < 1:
        raise ValueError("optimization.batch_size must be positive")
    gradient_accumulation_steps = int(opt_config.get("gradient_accumulation_steps", 1))
    if gradient_accumulation_steps < 1:
        raise ValueError("optimization.gradient_accumulation_steps must be positive")
    max_epochs = int(opt_config.get("max_epochs", 1))
    ignore_index = int(config.get("data", {}).get("ignore_index", -100))
    early_stopping = _early_stopping_config(opt_config.get("early_stopping"))
    history: list[dict[str, float]] = []
    best_val_loss: float | None = None
    best_epoch: int | None = None
    epochs_without_improvement = 0
    stopped_early = False
    stop_reason: str | None = None

    for epoch in range(max_epochs):
        indices = list(range(len(dataset)))
        if opt_config.get("shuffle", True):
            random.shuffle(indices)

        epoch_loss = 0.0
        epoch_items = 0
        optimizer_steps = 0
        model.train()
        microbatch_starts = list(range(0, len(indices), batch_size))
        optimizer.zero_grad(set_to_none=True)
        for microbatch_index, start in enumerate(microbatch_starts):
            batch_indices = indices[start : start + batch_size]
            examples = [dataset[index] for index in batch_indices]
            batch = collate_training_examples(examples)
            audio = batch["audio"].to(device)
            text = batch["text"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)

            logits = model(audio, text, audio_mask=mask)
            loss = frame_label_cross_entropy(logits, targets, ignore_index=ignore_index)
            loss_for_backward = loss / _accumulation_group_size(
                microbatch_index=microbatch_index,
                num_microbatches=len(microbatch_starts),
                gradient_accumulation_steps=gradient_accumulation_steps,
            )
            loss_for_backward.backward()
            if _should_step_optimizer(
                microbatch_index=microbatch_index,
                num_microbatches=len(microbatch_starts),
                gradient_accumulation_steps=gradient_accumulation_steps,
            ):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

            epoch_loss += float(loss.detach().cpu()) * len(examples)
            epoch_items += len(examples)

        mean_loss = epoch_loss / max(epoch_items, 1)
        record = {
            "epoch": float(epoch),
            "loss": mean_loss,
            "learning_rate": _current_lr(optimizer),
            "optimizer_steps": float(optimizer_steps),
        }
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
            if best_val_loss is None or val_loss < best_val_loss - early_stopping["min_delta"]:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(
                    _checkpoint_payload(
                        model=model,
                        config=config,
                        labels=first.labels,
                        audio_dim=first.audio.shape[1],
                        text_dim=first.text.shape[1],
                    ),
                    best_checkpoint_path,
                )
                message += " best"
            elif early_stopping["patience"] is not None:
                epochs_without_improvement += 1
                message += f" patience={epochs_without_improvement}/{early_stopping['patience']}"
                if epochs_without_improvement >= early_stopping["patience"]:
                    stopped_early = True
                    stop_reason = (
                        f"validation loss did not improve for {early_stopping['patience']} epochs"
                    )
                    record["stopped_early"] = True
            if lr_scheduler is not None:
                old_lr = _current_lr(optimizer)
                lr_scheduler.step(val_loss)
                new_lr = _current_lr(optimizer)
                record["learning_rate"] = new_lr
                if new_lr < old_lr:
                    message += f" lr={new_lr:.2e}"
        history.append(record)
        print(message)

        if stopped_early:
            break

    checkpoint = _checkpoint_payload(
        model=model,
        config=config,
        labels=first.labels,
        audio_dim=first.audio.shape[1],
        text_dim=first.text.shape[1],
    )
    torch.save(checkpoint, checkpoint_path)
    metrics = {
        "checkpoint": str(checkpoint_path),
        "best_checkpoint": (
            None
            if best_checkpoint_path is None or not best_checkpoint_path.exists()
            else str(best_checkpoint_path)
        ),
        "final_loss": history[-1]["loss"] if history else None,
        "final_val_loss": history[-1].get("val_loss") if history else None,
        "best_val_loss": best_val_loss,
        "best_epoch": None if best_epoch is None else best_epoch + 1,
        "epochs_trained": len(history),
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "final_learning_rate": history[-1].get("learning_rate") if history else None,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "history": history,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    return metrics


def _checkpoint_payload(
    model,
    config: dict[str, Any],
    labels: list[str],
    audio_dim: int,
    text_dim: int,
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "config": config,
        "labels": labels,
        "audio_dim": audio_dim,
        "text_dim": text_dim,
    }


def _early_stopping_config(value: Any) -> dict[str, Any]:
    if value in (None, False):
        return {"patience": None, "min_delta": 0.0}
    if value is True:
        return {"patience": 10, "min_delta": 0.0}
    if not isinstance(value, dict):
        raise TypeError("optimization.early_stopping must be a mapping, boolean, or null")
    patience = value.get("patience")
    if patience is not None:
        patience = int(patience)
        if patience < 1:
            raise ValueError("optimization.early_stopping.patience must be positive")
    min_delta = float(value.get("min_delta", 0.0))
    if min_delta < 0:
        raise ValueError("optimization.early_stopping.min_delta must be non-negative")
    return {"patience": patience, "min_delta": min_delta}


def _build_lr_scheduler(value: Any, optimizer, torch, has_validation: bool):
    if value in (None, False):
        return None
    if value is True:
        value = {"name": "reduce_on_plateau"}
    if not isinstance(value, dict):
        raise TypeError("optimization.lr_scheduler must be a mapping, boolean, or null")

    name = str(value.get("name", "reduce_on_plateau")).lower()
    if name not in {"reduce_on_plateau", "reduce_lr_on_plateau"}:
        raise ValueError(f"Unknown lr scheduler: {name}")
    if not has_validation:
        raise ValueError("ReduceLROnPlateau requires a validation manifest")

    patience = int(value.get("patience", 2))
    if patience < 1:
        raise ValueError("optimization.lr_scheduler.patience must be positive")
    factor = float(value.get("factor", 0.5))
    if not 0.0 < factor < 1.0:
        raise ValueError("optimization.lr_scheduler.factor must be between 0 and 1")
    threshold = float(value.get("threshold", 0.0))
    min_lr = float(value.get("min_lr", 0.0))
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=factor,
        patience=patience,
        threshold=threshold,
        threshold_mode=str(value.get("threshold_mode", "abs")),
        cooldown=int(value.get("cooldown", 0)),
        min_lr=min_lr,
    )


def _current_lr(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _accumulation_group_size(
    microbatch_index: int,
    num_microbatches: int,
    gradient_accumulation_steps: int,
) -> int:
    group_start = (microbatch_index // gradient_accumulation_steps) * gradient_accumulation_steps
    return min(gradient_accumulation_steps, num_microbatches - group_start)


def _should_step_optimizer(
    microbatch_index: int,
    num_microbatches: int,
    gradient_accumulation_steps: int,
) -> bool:
    is_accumulation_boundary = (microbatch_index + 1) % gradient_accumulation_steps == 0
    is_last_microbatch = microbatch_index == num_microbatches - 1
    return is_accumulation_boundary or is_last_microbatch


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
