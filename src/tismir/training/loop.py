from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from tismir.losses import (
    audio_audio_supervised_contrastive,
    audio_to_text_infonce,
    frame_label_cross_entropy,
    text_to_audio_infonce,
)
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
    loss_config = _loss_config(config.get("loss", {}))
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
        epoch_component_sums: dict[str, float] = {}
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

            loss, loss_components = _compute_loss(
                model=model,
                audio=audio,
                text=text,
                targets=targets,
                mask=mask,
                ignore_index=ignore_index,
                loss_config=loss_config,
            )
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
            for name, value in loss_components.items():
                epoch_component_sums[name] = (
                    epoch_component_sums.get(name, 0.0) + float(value.detach().cpu()) * len(examples)
                )
            epoch_items += len(examples)

        mean_loss = epoch_loss / max(epoch_items, 1)
        record = {
            "epoch": float(epoch),
            "loss": mean_loss,
            "learning_rate": _current_lr(optimizer),
            "optimizer_steps": float(optimizer_steps),
        }
        for name, total in sorted(epoch_component_sums.items()):
            record[f"loss_{name}"] = total / max(epoch_items, 1)
        message = f"epoch={epoch + 1}/{max_epochs} loss={mean_loss:.6f}"
        if validation_dataset is not None:
            val_loss, val_components = _evaluate_loss(
                model=model,
                dataset=validation_dataset,
                batch_size=batch_size,
                device=device,
                ignore_index=ignore_index,
                loss_config=loss_config,
            )
            record["val_loss"] = val_loss
            message += f" val_loss={val_loss:.6f}"
            for name, value in sorted(val_components.items()):
                record[f"val_loss_{name}"] = value
                message += f" val_{name}={value:.6f}"
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
        "loss_config": loss_config,
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
    loss_config: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    torch = _require_torch()
    model.eval()
    total_loss = 0.0
    total_items = 0
    component_sums: dict[str, float] = {}
    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            examples = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
            batch = collate_training_examples(examples)
            audio = batch["audio"].to(device)
            text = batch["text"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)
            loss, components = _compute_loss(
                model=model,
                audio=audio,
                text=text,
                targets=targets,
                mask=mask,
                ignore_index=ignore_index,
                loss_config=loss_config,
            )
            total_loss += float(loss.detach().cpu()) * len(examples)
            for name, value in components.items():
                component_sums[name] = (
                    component_sums.get(name, 0.0) + float(value.detach().cpu()) * len(examples)
                )
            total_items += len(examples)
    model.train()
    return (
        total_loss / max(total_items, 1),
        {
            name: total / max(total_items, 1)
            for name, total in sorted(component_sums.items())
        },
    )


def _compute_loss(
    model,
    audio,
    text,
    targets,
    mask,
    ignore_index: int,
    loss_config: dict[str, Any],
):
    needs_features = _has_auxiliary_losses(loss_config)
    if needs_features:
        if not hasattr(model, "extract_features"):
            raise ValueError("Auxiliary contrastive losses require a model with extract_features()")
        features = model.extract_features(audio, text, audio_mask=mask)
        logits = features["logits"]
    else:
        features = None
        logits = model(audio, text, audio_mask=mask)

    components = {}
    weighted_terms = []

    frame_label_weight = float(loss_config["frame_label_weight"])
    if frame_label_weight:
        frame_loss = frame_label_cross_entropy(logits, targets, ignore_index=ignore_index)
        components["frame_label"] = frame_loss.detach()
        weighted_terms.append(frame_label_weight * frame_loss)

    audio_to_text_config = loss_config["audio_to_text"]
    if audio_to_text_config["weight"]:
        a2t_logits = logits
        if features is not None and audio_to_text_config["temperature"] is not None:
            a2t_logits = features["similarity"] / float(audio_to_text_config["temperature"])
        a2t_loss = audio_to_text_infonce(a2t_logits, targets, ignore_index=ignore_index)
        components["audio_to_text"] = a2t_loss.detach()
        weighted_terms.append(float(audio_to_text_config["weight"]) * a2t_loss)

    text_to_audio_config = loss_config["text_to_audio"]
    if text_to_audio_config["weight"]:
        t2a_loss = text_to_audio_infonce(
            features["audio_tokens"],
            features["text_tokens"],
            targets,
            temperature=float(text_to_audio_config["temperature"]),
            ignore_index=ignore_index,
        )
        components["text_to_audio"] = t2a_loss.detach()
        weighted_terms.append(float(text_to_audio_config["weight"]) * t2a_loss)

    audio_to_audio_config = loss_config["audio_to_audio"]
    if audio_to_audio_config["weight"]:
        a2a_loss = audio_audio_supervised_contrastive(
            features["audio_tokens"],
            targets,
            temperature=float(audio_to_audio_config["temperature"]),
            ignore_index=ignore_index,
        )
        components["audio_to_audio"] = a2a_loss.detach()
        weighted_terms.append(float(audio_to_audio_config["weight"]) * a2a_loss)

    if not weighted_terms:
        raise ValueError("At least one loss weight must be non-zero")
    return sum(weighted_terms), components


def _loss_config(value: Any) -> dict[str, Any]:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise TypeError("loss must be a mapping or null")

    return {
        "name": str(value.get("name", "frame_label_cross_entropy")),
        "frame_label_weight": float(value.get("frame_label_weight", value.get("weight", 1.0))),
        "audio_to_text": _contrastive_term_config(value.get("audio_to_text"), default_weight=0.0),
        "text_to_audio": _contrastive_term_config(value.get("text_to_audio"), default_weight=0.0),
        "audio_to_audio": _contrastive_term_config(value.get("audio_to_audio"), default_weight=0.0),
    }


def _contrastive_term_config(value: Any, default_weight: float) -> dict[str, float | None]:
    if value in (None, False):
        value = {"weight": 0.0}
    elif value is True:
        value = {"weight": default_weight}
    elif isinstance(value, (int, float)):
        value = {"weight": float(value)}
    elif not isinstance(value, dict):
        raise TypeError("contrastive loss term config must be a mapping, number, boolean, or null")
    temperature = value.get("temperature", 0.07)
    return {
        "weight": float(value.get("weight", default_weight)),
        "temperature": None if temperature is None else float(temperature),
    }


def _has_auxiliary_losses(loss_config: dict[str, Any]) -> bool:
    return any(
        float(loss_config[name]["weight"]) != 0.0
        for name in ("audio_to_text", "text_to_audio", "audio_to_audio")
    )


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
