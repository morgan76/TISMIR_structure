from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

from tismir.decoding.segments import (
    decode_label_indices,
    merge_frame_labels,
    remove_short_segments,
    smooth_logits,
)
from tismir.data.annotations import (
    is_random_annotation_processing,
    validation_annotation_processing_choice,
)
from tismir.losses import (
    audio_audio_supervised_contrastive,
    audio_to_text_infonce,
    boundary_prediction_loss,
    cross_similarity_matching_loss,
    frame_label_cross_entropy,
    pairwise_probability_loss,
    pairwise_structure_relation_loss,
    token_uniformity_loss,
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
    segmentation_validation = _segmentation_validation_config(
        config.get("validation"),
        model_config=config.get("model", {}),
    )
    best_segmentation_checkpoint_path = (
        output_dir / str(segmentation_validation["checkpoint_name"])
        if validation_dataset is not None and segmentation_validation["enabled"]
        else None
    )

    opt_config = config.get("optimization", {})
    progress = _progress_config(opt_config.get("progress", "auto"))
    _print_model_summary(
        model=model,
        audio_shape=(1, *first.audio.shape),
        text_shape=first.text.shape,
        device=device,
        value=opt_config.get("model_summary", True),
    )
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
    best_segmentation_score: float | None = None
    best_segmentation_epoch: int | None = None
    epochs_without_improvement = 0
    stopped_early = False
    stop_reason: str | None = None

    for epoch in range(max_epochs):
        dataset.set_epoch(epoch)
        indices = list(range(len(dataset)))
        if opt_config.get("shuffle", True):
            random.shuffle(indices)

        epoch_loss = 0.0
        epoch_items = 0
        epoch_component_sums: dict[str, float] = {}
        optimizer_steps = 0
        model.train()
        microbatch_starts = list(range(0, len(indices), batch_size))
        train_iterator = _progress_iter(
            enumerate(microbatch_starts),
            enabled=progress,
            total=len(microbatch_starts),
            desc=f"epoch {epoch + 1}/{max_epochs} train",
        )
        optimizer.zero_grad(set_to_none=True)
        for microbatch_index, start in train_iterator:
            batch_indices = indices[start : start + batch_size]
            examples = [dataset[index] for index in batch_indices]
            batch = collate_training_examples(examples)
            audio = batch["audio"].to(device)
            text = batch["text"].to(device)
            targets = batch["targets"].to(device)
            base_targets = batch["base_targets"].to(device)
            segment_targets = batch["segment_targets"].to(device)
            mask = batch["mask"].to(device)

            loss, loss_components = _compute_loss(
                model=model,
                audio=audio,
                text=text,
                targets=targets,
                base_targets=base_targets,
                segment_targets=segment_targets,
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
            segmentation_improved = False
            val_loss, val_components = _evaluate_loss(
                model=model,
                dataset=validation_dataset,
                batch_size=batch_size,
                device=device,
                ignore_index=ignore_index,
                loss_config=loss_config,
                progress=progress,
                desc=f"epoch {epoch + 1}/{max_epochs} val-loss",
            )
            record["val_loss"] = val_loss
            message += f" val_loss={val_loss:.6f}"
            for name, value in sorted(val_components.items()):
                record[f"val_loss_{name}"] = value
                message += f" val_{name}={value:.6f}"
            if segmentation_validation["enabled"]:
                segmentation = _evaluate_segmentation(
                    model=model,
                    dataset=validation_dataset,
                    device=device,
                    ignore_index=ignore_index,
                    config=segmentation_validation,
                    progress=progress,
                    desc=f"epoch {epoch + 1}/{max_epochs} val-seg",
                )
                record["val_segmentation_num_tracks"] = float(segmentation["num_tracks"])
                for name, value in sorted(segmentation["metrics"].items()):
                    record[f"val_segmentation_{name}"] = value
                monitor = str(segmentation_validation["monitor"])
                monitor_score = segmentation["metrics"][monitor]
                for name in _segmentation_display_metrics(
                    config=segmentation_validation,
                    metrics=segmentation["metrics"],
                ):
                    message += f" val_{_short_metric_name(name)}={segmentation['metrics'][name]:.6f}"
                if _is_metric_improvement(
                    current=monitor_score,
                    best=best_segmentation_score,
                    mode=str(segmentation_validation["mode"]),
                    min_delta=float(segmentation_validation["min_delta"]),
                ):
                    segmentation_improved = True
                    best_segmentation_score = monitor_score
                    best_segmentation_epoch = epoch
                    torch.save(
                        _checkpoint_payload(
                            model=model,
                            config=config,
                            labels=first.labels,
                            audio_dim=first.audio.shape[1],
                            text_dim=first.text.shape[1],
                        ),
                        best_segmentation_checkpoint_path,
                    )
                    message += " best_segmentation"
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
            elif segmentation_improved and early_stopping["reset_on_segmentation_improvement"]:
                epochs_without_improvement = 0
            elif early_stopping["patience"] is not None:
                epochs_without_improvement += 1
                message += f" patience={epochs_without_improvement}/{early_stopping['patience']}"
                if epochs_without_improvement >= early_stopping["patience"]:
                    stopped_early = True
                    stop_reason = (
                        "neither validation loss nor monitored segmentation metric improved "
                        f"for {early_stopping['patience']} epochs"
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
        "best_segmentation_checkpoint": (
            None
            if best_segmentation_checkpoint_path is None
            or not best_segmentation_checkpoint_path.exists()
            else str(best_segmentation_checkpoint_path)
        ),
        "best_segmentation_metric": (
            None if not segmentation_validation["enabled"] else segmentation_validation["monitor"]
        ),
        "best_segmentation_score": best_segmentation_score,
        "best_segmentation_epoch": (
            None if best_segmentation_epoch is None else best_segmentation_epoch + 1
        ),
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


def _segmentation_validation_config(value: Any, model_config: dict[str, Any] | None = None) -> dict[str, Any]:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise TypeError("validation must be a mapping or null")
    segmentation = value.get("segmentation", False)
    if segmentation in (None, False):
        return {"enabled": False}
    if segmentation is True:
        segmentation = {}
    if not isinstance(segmentation, dict):
        raise TypeError("validation.segmentation must be a mapping, boolean, or null")
    if not bool(segmentation.get("enabled", True)):
        return {"enabled": False}

    monitor = str(segmentation.get("monitor", "F-measure@3.0"))
    metrics = tuple(
        segmentation.get(
            "metrics",
            (
                "F-measure@0.5",
                "F-measure@3.0",
                "Acc",
                "Balanced Acc",
                "Pairwise F-measure",
                "NCE F-measure",
            ),
        )
    )
    if monitor not in metrics:
        metrics = (*metrics, monitor)
    mode = str(segmentation.get("mode", "max")).lower()
    if mode not in {"max", "min"}:
        raise ValueError("validation.segmentation.mode must be one of: max, min")
    limit = segmentation.get("limit")
    if limit is not None:
        limit = int(limit)
        if limit < 1:
            raise ValueError("validation.segmentation.limit must be positive")
    smoothing_window = int(segmentation.get("smoothing_window", 7))
    if smoothing_window < 1:
        raise ValueError("validation.segmentation.smoothing_window must be positive")
    transition_penalty = float(segmentation.get("transition_penalty", 8.0))
    if transition_penalty < 0:
        raise ValueError("validation.segmentation.transition_penalty must be non-negative")
    min_segment_duration = float(segmentation.get("min_segment_duration", 0.0))
    if min_segment_duration < 0:
        raise ValueError("validation.segmentation.min_segment_duration must be non-negative")
    boundary_decoding = _boundary_decoding_config(
        segmentation.get("boundary_decoding"),
        model_config=model_config,
    )

    return {
        "enabled": True,
        "monitor": monitor,
        "mode": mode,
        "min_delta": float(segmentation.get("min_delta", 0.0)),
        "metrics": metrics,
        "display_metrics": tuple(
            segmentation.get(
                "display_metrics",
                (
                    "F-measure@0.5",
                    "F-measure@3.0",
                    "Acc",
                    "Balanced Acc",
                    "Pairwise F-measure",
                    "NCE F-measure",
                ),
            )
        ),
        "limit": limit,
        "smoothing_window": smoothing_window,
        "smoothing_mode": str(segmentation.get("smoothing_mode", "mean")),
        "decoder": str(segmentation.get("decoder", "viterbi")),
        "transition_penalty": transition_penalty,
        "boundary_decoding": boundary_decoding,
        "min_segment_duration": min_segment_duration,
        "trim": bool(segmentation.get("trim", True)),
        "checkpoint_name": str(segmentation.get("checkpoint_name", "best_segmentation_checkpoint.pt")),
    }


def _boundary_decoding_config(value: Any, model_config: dict[str, Any] | None) -> dict[str, Any]:
    head_enabled = _model_boundary_head_enabled(model_config or {})
    if value in (None, "auto"):
        value = {"enabled": head_enabled}
    elif value is True:
        value = {"enabled": True}
    elif value is False:
        value = {"enabled": False}
    elif not isinstance(value, dict):
        raise TypeError("validation.segmentation.boundary_decoding must be a mapping, boolean, 'auto', or null")

    enabled = bool(value.get("enabled", head_enabled)) and head_enabled
    weight = float(value.get("weight", 1.0))
    if weight < 0:
        raise ValueError("validation.segmentation.boundary_decoding.weight must be non-negative")
    eps = float(value.get("eps", 1e-4))
    if not 0.0 < eps < 0.5:
        raise ValueError("validation.segmentation.boundary_decoding.eps must be between 0 and 0.5")
    return {
        "enabled": enabled,
        "weight": weight,
        "eps": eps,
    }


def _model_boundary_head_enabled(model_config: dict[str, Any]) -> bool:
    update_blocks = model_config.get("update_blocks", model_config.get("cross_attention", {}))
    if not isinstance(update_blocks, dict):
        return False
    boundary_head = update_blocks.get("boundary_head", False)
    if isinstance(boundary_head, dict):
        return bool(boundary_head.get("enabled", True))
    return bool(boundary_head)


def _is_metric_improvement(
    current: float,
    best: float | None,
    mode: str,
    min_delta: float,
) -> bool:
    if best is None:
        return True
    if mode == "max":
        return current > best + min_delta
    if mode == "min":
        return current < best - min_delta
    raise ValueError("mode must be one of: max, min")


def _short_metric_name(metric: str) -> str:
    aliases = {
        "Pairwise F-measure": "PFC",
        "NCE F-measure": "NCE",
    }
    if metric in aliases:
        return aliases[metric]
    return metric.replace(" ", "_").replace("F-measure", "F").replace("@", "at")


def _segmentation_display_metrics(config: dict[str, Any], metrics: dict[str, float]) -> list[str]:
    names: list[str] = []
    for name in (*config.get("display_metrics", ()), str(config["monitor"])):
        if name in metrics and name not in names:
            names.append(name)
    return names


def _early_stopping_config(value: Any) -> dict[str, Any]:
    if value in (None, False):
        return {
            "patience": None,
            "min_delta": 0.0,
            "reset_on_segmentation_improvement": True,
        }
    if value is True:
        return {
            "patience": 10,
            "min_delta": 0.0,
            "reset_on_segmentation_improvement": True,
        }
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
    return {
        "patience": patience,
        "min_delta": min_delta,
        "reset_on_segmentation_improvement": bool(
            value.get("reset_on_segmentation_improvement", True)
        ),
    }


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


def _progress_config(value: Any) -> bool:
    if value == "auto":
        return sys.stderr.isatty()
    return bool(value)


def _progress_iter(iterable, enabled: bool, total: int | None = None, desc: str | None = None):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False, dynamic_ncols=True)


def _print_model_summary(
    model,
    audio_shape: tuple[int, ...],
    text_shape: tuple[int, ...],
    device,
    value: Any,
) -> None:
    config = _model_summary_config(value)
    if not config["enabled"]:
        return

    total_params = _num_parameters(model)
    trainable_params = _num_parameters(model, trainable_only=True)
    non_trainable_params = total_params - trainable_params
    param_memory_mb = total_params * 4 / 1024**2

    lines = [
        "model summary",
        f"  class: {model.__class__.__name__}",
        f"  device: {device}",
        f"  audio input: {list(audio_shape)}",
        f"  text input: {list(text_shape)}",
        f"  parameters: {_format_count(total_params)} total, "
        f"{_format_count(trainable_params)} trainable, "
        f"{_format_count(non_trainable_params)} frozen",
        f"  parameter memory: {param_memory_mb:.2f} MB (float32 estimate)",
    ]

    architecture_lines = _architecture_summary_lines(model)
    if architecture_lines:
        lines.append("  architecture:")
        lines.extend(architecture_lines)

    module_lines = _module_summary_lines(
        model,
        depth=int(config["depth"]),
        max_lines=int(config["max_lines"]),
    )
    if module_lines:
        lines.append("  modules:")
        lines.extend(module_lines)
    print("\n".join(lines))


def _model_summary_config(value: Any) -> dict[str, Any]:
    if value in (None, True):
        value = {}
    elif value is False:
        return {"enabled": False}
    elif not isinstance(value, dict):
        raise TypeError("optimization.model_summary must be a mapping, boolean, or null")

    depth = int(value.get("depth", 2))
    if depth < 1:
        raise ValueError("optimization.model_summary.depth must be positive")
    max_lines = int(value.get("max_lines", 80))
    if max_lines < 1:
        raise ValueError("optimization.model_summary.max_lines must be positive")
    return {
        "enabled": bool(value.get("enabled", True)),
        "depth": depth,
        "max_lines": max_lines,
    }


def _architecture_summary_lines(model) -> list[str]:
    lines: list[str] = []
    if bool(getattr(model, "bidirectional_cross_attention", False)):
        audio_adapter_layers = _transformer_layer_count(getattr(model, "audio_adapter", None))
        text_adapter_layers = _transformer_layer_count(getattr(model, "text_adapter", None))
        update_blocks = getattr(model, "cross_attention_blocks", None)
        update_count = 0 if update_blocks is None else len(update_blocks)
        update_section_layers = None
        update_frame_layers = None
        if update_count:
            first_block = update_blocks[0]
            update_section_layers = _transformer_layer_count(
                getattr(first_block, "section_branch", None)
            )
            update_frame_layers = _transformer_layer_count(
                getattr(first_block, "frame_branch", None)
            )
        lines.extend(
            [
                "    bidirectional section-conditioned audio model",
                f"    audio adapter self-attention layers: {audio_adapter_layers}",
                f"    text adapter self-attention layers: {text_adapter_layers}",
                "    init: sections <- audio",
                f"    update blocks: {update_count}",
                "    update order: audio <- sections, link-aware audio update, sections <- audio, section self-attention",
                "    link input channels: frame-label probability similarity and audio cosine",
                "    output: frame-label similarity from final audio and section tokens",
            ]
        )
        if update_section_layers is not None:
            lines.append(
                "    section self-attention layers per update block: "
                f"{update_section_layers}"
            )
        if update_frame_layers is not None:
            lines.append(
                "    audio relation/self-attention layers per update block: "
                f"{update_frame_layers}"
            )
    return lines


def _transformer_layer_count(module) -> int:
    layers = getattr(module, "layers", None)
    if layers is not None:
        return len(layers)
    return 0 if module is None else 1


def _module_summary_lines(model, depth: int, max_lines: int) -> list[str]:
    rows: list[tuple[str, str, int, int]] = []
    for name, module in model.named_modules():
        if not name:
            continue
        module_depth = name.count(".") + 1
        if module_depth > depth:
            continue
        rows.append(
            (
                name,
                module.__class__.__name__,
                _num_parameters(module),
                _num_parameters(module, trainable_only=True),
            )
        )
    shown_rows = rows[:max_lines]
    lines = [
        f"    {name}: {class_name} "
        f"({_format_count(total)} params, {_format_count(trainable)} trainable)"
        for name, class_name, total, trainable in shown_rows
    ]
    hidden = len(rows) - len(shown_rows)
    if hidden > 0:
        lines.append(f"    ... {hidden} more modules")
    return lines


def _num_parameters(module, trainable_only: bool = False) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )


def _format_count(value: int) -> str:
    return f"{value:,}"


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
    annotation_processing = data_config.get("annotation_processing")
    if is_random_annotation_processing(annotation_processing):
        assert isinstance(annotation_processing, dict)
        data_config["annotation_processing"] = validation_annotation_processing_choice(
            annotation_processing
        )
    return StructureEmbeddingDataset(**data_config)


def _evaluate_loss(
    model,
    dataset: StructureEmbeddingDataset,
    batch_size: int,
    device,
    ignore_index: int,
    loss_config: dict[str, Any],
    progress: bool = False,
    desc: str | None = None,
) -> tuple[float, dict[str, float]]:
    torch = _require_torch()
    model.eval()
    total_loss = 0.0
    total_items = 0
    component_sums: dict[str, float] = {}
    starts = list(range(0, len(dataset), batch_size))
    iterator = _progress_iter(starts, enabled=progress, total=len(starts), desc=desc)
    with torch.inference_mode():
        for start in iterator:
            examples = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
            batch = collate_training_examples(examples)
            audio = batch["audio"].to(device)
            text = batch["text"].to(device)
            targets = batch["targets"].to(device)
            base_targets = batch["base_targets"].to(device)
            segment_targets = batch["segment_targets"].to(device)
            mask = batch["mask"].to(device)
            loss, components = _compute_loss(
                model=model,
                audio=audio,
                text=text,
                targets=targets,
                base_targets=base_targets,
                segment_targets=segment_targets,
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


def _evaluate_segmentation(
    model,
    dataset: StructureEmbeddingDataset,
    device,
    ignore_index: int,
    config: dict[str, Any],
    progress: bool = False,
    desc: str | None = None,
) -> dict[str, Any]:
    try:
        import mir_eval
    except ImportError as exc:  # pragma: no cover - installed through JAMS
        raise ImportError("mir_eval is required for segmentation validation.") from exc

    torch = _require_torch()
    model.eval()
    metric_names = tuple(config["metrics"])
    limit = config["limit"]
    count = len(dataset) if limit is None else min(limit, len(dataset))
    metric_values = {name: [] for name in metric_names}
    iterator = _progress_iter(range(count), enabled=progress, total=count, desc=desc)

    with torch.inference_mode():
        for index in iterator:
            example = dataset[index]
            audio = torch.from_numpy(example.audio).unsqueeze(0).to(device)
            text = torch.from_numpy(example.text).to(device)
            boundary_probabilities = None
            if bool(config["boundary_decoding"]["enabled"]):
                if not hasattr(model, "extract_features"):
                    raise ValueError("boundary decoding requires a model with extract_features()")
                features = model.extract_features(audio, text)
                logits = features["logits"][0].detach().cpu().numpy()
                boundary_probabilities = _boundary_probabilities_from_features(features)
            else:
                logits = model(audio, text)[0].detach().cpu().numpy()
            decoded_logits = smooth_logits(
                logits,
                window=int(config["smoothing_window"]),
                mode=str(config["smoothing_mode"]),
            )
            label_indices = decode_label_indices(
                decoded_logits,
                strategy=str(config["decoder"]),
                transition_penalty=float(config["transition_penalty"]),
                boundary_probabilities=boundary_probabilities,
                boundary_weight=(
                    float(config["boundary_decoding"]["weight"])
                    if boundary_probabilities is not None
                    else 0.0
                ),
                boundary_eps=float(config["boundary_decoding"]["eps"]),
            )
            frame_scores = _frame_label_accuracy_scores(
                label_indices,
                targets=example.targets,
                ignore_index=ignore_index,
            )
            predicted_labels = [example.labels[int(label_index)] for label_index in label_indices]
            predicted_segments = merge_frame_labels(example.beat_intervals, predicted_labels)
            predicted_segments = remove_short_segments(
                predicted_segments,
                min_duration=float(config["min_segment_duration"]),
            )
            reference_segments = _reference_segments_from_targets(
                intervals=example.beat_intervals,
                targets=example.targets,
                labels=example.labels,
                ignore_index=ignore_index,
            )
            if not reference_segments or not predicted_segments:
                continue

            ref_intervals, ref_labels = _segments_to_arrays(reference_segments)
            pred_intervals, pred_labels = _segments_to_arrays(predicted_segments)
            duration = max(float(ref_intervals[-1, 1]), float(pred_intervals[-1, 1]))
            ref_intervals, ref_labels = mir_eval.util.adjust_intervals(
                ref_intervals,
                list(ref_labels),
                t_min=0.0,
                t_max=duration,
            )
            pred_intervals, pred_labels = mir_eval.util.adjust_intervals(
                pred_intervals,
                list(pred_labels),
                t_min=0.0,
                t_max=duration,
            )
            scores = mir_eval.segment.evaluate(
                ref_intervals,
                ref_labels,
                pred_intervals,
                pred_labels,
                trim=bool(config["trim"]),
            )
            for name in metric_names:
                if name in frame_scores:
                    metric_values[name].append(float(frame_scores[name]))
                else:
                    metric_values[name].append(float(scores[name]))

    model.train()
    if not any(metric_values.values()):
        raise ValueError("No validation tracks could be evaluated for segmentation metrics")
    return {
        "num_tracks": max(len(values) for values in metric_values.values()),
        "metrics": {
            name: float(np.nanmean(values)) if values else float("nan")
            for name, values in metric_values.items()
        },
    }


def _boundary_probabilities_from_features(features: dict[str, Any]) -> np.ndarray | None:
    boundary_logits = features.get("boundary_logits")
    if not boundary_logits:
        return None
    logits = boundary_logits[-1][0].detach().cpu().numpy().astype(np.float64)
    return 1.0 / (1.0 + np.exp(-logits))


def _frame_label_accuracy_scores(
    predictions: np.ndarray,
    targets: np.ndarray,
    ignore_index: int,
) -> dict[str, float]:
    targets = targets.astype(np.int64, copy=False)
    predictions = predictions.astype(np.int64, copy=False)
    valid = targets != ignore_index
    if not np.any(valid):
        return {"Acc": float("nan"), "Balanced Acc": float("nan")}

    valid_predictions = predictions[valid]
    valid_targets = targets[valid]
    per_label_acc = [
        float(np.mean(valid_predictions[valid_targets == label_index] == label_index))
        for label_index in np.unique(valid_targets)
    ]
    return {
        "Acc": float(np.mean(valid_predictions == valid_targets)),
        "Balanced Acc": float(np.mean(per_label_acc)) if per_label_acc else float("nan"),
    }


def _reference_segments_from_targets(
    intervals: list[tuple[float, float]],
    targets: np.ndarray,
    labels: list[str],
    ignore_index: int,
) -> list[tuple[float, float, str]]:
    segments: list[tuple[float, float, str]] = []
    current: tuple[float, float, str] | None = None
    for interval, target in zip(intervals, targets):
        target_index = int(target)
        if target_index == ignore_index:
            if current is not None:
                segments.append(current)
                current = None
            continue
        start, end = interval
        label = labels[target_index]
        if current is not None and current[2] == label and np.isclose(current[1], start):
            current = (current[0], end, label)
        else:
            if current is not None:
                segments.append(current)
            current = (start, end, label)
    if current is not None:
        segments.append(current)
    return segments


def _segments_to_arrays(
    segments: list[tuple[float, float, str]],
) -> tuple[np.ndarray, list[str]]:
    intervals = np.asarray([(start, end) for start, end, _ in segments], dtype=float)
    labels = [label for _, _, label in segments]
    return intervals, labels


def _compute_loss(
    model,
    audio,
    text,
    targets,
    base_targets,
    segment_targets,
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

    pairwise_probability_config = loss_config["pairwise_probability"]
    if pairwise_probability_config["weight"]:
        pair_loss = pairwise_probability_loss(
            logits,
            targets,
            ignore_index=ignore_index,
            balance=bool(pairwise_probability_config["balance"]),
        )
        components["pairwise_probability"] = pair_loss.detach()
        weighted_terms.append(float(pairwise_probability_config["weight"]) * pair_loss)

    structure_pairwise_config = loss_config["structure_pairwise"]
    if structure_pairwise_config["weight"]:
        structure_tokens = features.get("structure_tokens") if features is not None else None
        if structure_tokens is None:
            raise ValueError(
                "structure_pairwise loss requires model.structure_head.enabled=true"
            )
        if not hasattr(model, "structure_pair_logits"):
            raise ValueError("structure_pairwise loss requires a model with structure_pair_logits()")
        structure_loss = pairwise_structure_relation_loss(
            model.structure_pair_logits(structure_tokens),
            base_targets,
            segment_targets,
            ignore_index=ignore_index,
            balance=bool(structure_pairwise_config["balance"]),
        )
        components["structure_pairwise"] = structure_loss.detach()
        weighted_terms.append(float(structure_pairwise_config["weight"]) * structure_loss)

    link_relation_config = loss_config["link_relation"]
    if link_relation_config["weight"]:
        link_logits = features.get("link_logits") if features is not None else None
        if not link_logits:
            raise ValueError(
                "link_relation loss requires model.update_blocks.relation_attention.enabled=true"
            )
        link_losses = [
            pairwise_structure_relation_loss(
                block_logits,
                targets,
                segment_targets,
                ignore_index=ignore_index,
                balance=bool(link_relation_config["balance"]),
            )
            for block_logits in link_logits
        ]
        link_loss = sum(link_losses) / len(link_losses)
        components["link_relation"] = link_loss.detach()
        weighted_terms.append(float(link_relation_config["weight"]) * link_loss)

    boundary_config = loss_config["boundary"]
    if boundary_config["weight"]:
        boundary_logits = features.get("boundary_logits") if features is not None else None
        if not boundary_logits:
            raise ValueError(
                "boundary loss requires model.update_blocks.boundary_head.enabled=true"
            )
        boundary_losses = [
            boundary_prediction_loss(
                block_logits,
                segment_targets,
                ignore_index=ignore_index,
            )
            for block_logits in boundary_logits
        ]
        boundary_loss = sum(boundary_losses) / len(boundary_losses)
        components["boundary"] = boundary_loss.detach()
        weighted_terms.append(float(boundary_config["weight"]) * boundary_loss)

    intermediate_config = loss_config["intermediate_frame_label"]
    if intermediate_config["weight"]:
        intermediate_logits = features["intermediate_logits"]
        if not intermediate_logits:
            raise ValueError(
                "intermediate_frame_label loss requires model.cross_attention.intermediate_logits=true"
            )
        losses = [
            frame_label_cross_entropy(block_logits, targets, ignore_index=ignore_index)
            for block_logits in intermediate_logits
        ]
        intermediate_loss = sum(losses) / len(losses)
        components["intermediate_frame_label"] = intermediate_loss.detach()
        weighted_terms.append(float(intermediate_config["weight"]) * intermediate_loss)

    attention_alignment_config = loss_config["cross_attention_alignment"]
    if attention_alignment_config["weight"]:
        attention_maps = features["attention_maps"]
        if not attention_maps:
            raise ValueError(
                "cross_attention_alignment loss requires model.cross_attention.return_attention=true"
            )
        attention_loss = _cross_attention_alignment_loss(
            attention_maps=attention_maps,
            targets=targets,
            ignore_index=ignore_index,
            frame_to_text_weight=float(attention_alignment_config["frame_to_text_weight"]),
            label_to_frame_weight=float(attention_alignment_config["label_to_frame_weight"]),
            use_all_blocks=bool(attention_alignment_config["use_all_blocks"]),
        )
        components["cross_attention_alignment"] = attention_loss.detach()
        weighted_terms.append(float(attention_alignment_config["weight"]) * attention_loss)

    similarity_matching_config = loss_config["similarity_matching"]
    if similarity_matching_config["weight"]:
        match_loss = cross_similarity_matching_loss(
            features["similarity"],
            targets,
            ignore_index=ignore_index,
            positive_target=float(similarity_matching_config["positive_target"]),
            negative_target=float(similarity_matching_config["negative_target"]),
            balance=bool(similarity_matching_config["balance"]),
        )
        components["similarity_matching"] = match_loss.detach()
        weighted_terms.append(float(similarity_matching_config["weight"]) * match_loss)

    text_uniformity_config = loss_config["text_uniformity"]
    if text_uniformity_config["weight"]:
        text_loss = token_uniformity_loss(
            features["text_tokens"],
            alpha=float(text_uniformity_config["alpha"]),
        )
        components["text_uniformity"] = text_loss.detach()
        weighted_terms.append(float(text_uniformity_config["weight"]) * text_loss)

    audio_uniformity_config = loss_config["audio_uniformity"]
    if audio_uniformity_config["weight"]:
        valid_audio_mask = targets != ignore_index
        audio_loss = token_uniformity_loss(
            features["audio_tokens"],
            mask=valid_audio_mask,
            alpha=float(audio_uniformity_config["alpha"]),
        )
        components["audio_uniformity"] = audio_loss.detach()
        weighted_terms.append(float(audio_uniformity_config["weight"]) * audio_loss)

    if not weighted_terms:
        raise ValueError("At least one loss weight must be non-zero")
    return sum(weighted_terms), components


def _cross_attention_alignment_loss(
    attention_maps,
    targets,
    ignore_index: int,
    frame_to_text_weight: float,
    label_to_frame_weight: float,
    use_all_blocks: bool,
):
    if frame_to_text_weight < 0 or label_to_frame_weight < 0:
        raise ValueError("cross-attention alignment direction weights must be non-negative")
    selected_maps = attention_maps if use_all_blocks else attention_maps[-1:]
    losses = []
    for maps in selected_maps:
        if frame_to_text_weight:
            losses.append(frame_to_text_weight * _frame_to_text_attention_loss(
                maps["frame_to_text"],
                targets,
                ignore_index=ignore_index,
            ))
        if label_to_frame_weight:
            losses.append(label_to_frame_weight * _label_to_frame_attention_loss(
                maps["label_to_frame"],
                targets,
                ignore_index=ignore_index,
            ))
    if not losses:
        return targets.sum() * 0.0
    return sum(losses) / len(losses)


def _frame_to_text_attention_loss(attention, targets, ignore_index: int):
    valid = targets != ignore_index
    if not bool(valid.any()):
        return attention.sum() * 0.0
    correct = targets.clamp_min(0).unsqueeze(-1)
    probabilities = attention.gather(dim=-1, index=correct).squeeze(-1)
    return -probabilities[valid].clamp_min(1e-8).log().mean()


def _label_to_frame_attention_loss(attention, targets, ignore_index: int):
    losses = []
    for batch_index in range(targets.shape[0]):
        labels = targets[batch_index]
        valid = labels != ignore_index
        if not bool(valid.any()):
            continue
        label_count = attention.shape[1]
        for label_index in range(label_count):
            positives = valid & (labels == label_index)
            if bool(positives.any()):
                probabilities = attention[batch_index, label_index, positives]
                losses.append(-probabilities.clamp_min(1e-8).log().mean())
    if not losses:
        return attention.sum() * 0.0
    return sum(losses) / len(losses)


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
        "pairwise_probability": _pairwise_probability_config(value.get("pairwise_probability")),
        "structure_pairwise": _pairwise_probability_config(value.get("structure_pairwise")),
        "link_relation": _pairwise_probability_config(value.get("link_relation")),
        "boundary": _simple_weight_config(value.get("boundary")),
        "intermediate_frame_label": _simple_weight_config(value.get("intermediate_frame_label")),
        "cross_attention_alignment": _cross_attention_alignment_config(
            value.get("cross_attention_alignment")
        ),
        "similarity_matching": _similarity_matching_config(value.get("similarity_matching")),
        "text_uniformity": _uniformity_config(value.get("text_uniformity")),
        "audio_uniformity": _uniformity_config(value.get("audio_uniformity")),
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
        for name in (
            "audio_to_text",
            "text_to_audio",
            "audio_to_audio",
            "structure_pairwise",
            "link_relation",
            "boundary",
            "intermediate_frame_label",
            "cross_attention_alignment",
            "similarity_matching",
            "text_uniformity",
            "audio_uniformity",
        )
    )


def _pairwise_probability_config(value: Any) -> dict[str, float | int]:
    if value in (None, False):
        value = {"weight": 0.0}
    elif value is True:
        value = {"weight": 0.5}
    elif isinstance(value, (int, float)):
        value = {"weight": float(value)}
    elif not isinstance(value, dict):
        raise TypeError("pairwise_probability loss config must be a mapping, number, boolean, or null")
    return {
        "weight": float(value.get("weight", 0.0)),
        "balance": bool(value.get("balance", True)),
    }


def _simple_weight_config(value: Any) -> dict[str, float]:
    if value in (None, False):
        value = {"weight": 0.0}
    elif value is True:
        value = {"weight": 1.0}
    elif isinstance(value, (int, float)):
        value = {"weight": float(value)}
    elif not isinstance(value, dict):
        raise TypeError("loss term config must be a mapping, number, boolean, or null")
    return {"weight": float(value.get("weight", 0.0))}


def _cross_attention_alignment_config(value: Any) -> dict[str, float | bool]:
    if value in (None, False):
        value = {"weight": 0.0}
    elif value is True:
        value = {"weight": 0.05}
    elif isinstance(value, (int, float)):
        value = {"weight": float(value)}
    elif not isinstance(value, dict):
        raise TypeError(
            "cross_attention_alignment loss config must be a mapping, number, boolean, or null"
        )
    return {
        "weight": float(value.get("weight", 0.0)),
        "frame_to_text_weight": float(value.get("frame_to_text_weight", 1.0)),
        "label_to_frame_weight": float(value.get("label_to_frame_weight", 1.0)),
        "use_all_blocks": bool(value.get("use_all_blocks", True)),
    }


def _similarity_matching_config(value: Any) -> dict[str, float | bool]:
    if value in (None, False):
        value = {"weight": 0.0}
    elif value is True:
        value = {"weight": 0.2}
    elif isinstance(value, (int, float)):
        value = {"weight": float(value)}
    elif not isinstance(value, dict):
        raise TypeError("similarity_matching loss config must be a mapping, number, boolean, or null")
    return {
        "weight": float(value.get("weight", 0.0)),
        "positive_target": float(value.get("positive_target", 1.0)),
        "negative_target": float(value.get("negative_target", 0.0)),
        "balance": bool(value.get("balance", True)),
    }


def _uniformity_config(value: Any) -> dict[str, float]:
    if value in (None, False):
        value = {"weight": 0.0}
    elif value is True:
        value = {"weight": 0.01}
    elif isinstance(value, (int, float)):
        value = {"weight": float(value)}
    elif not isinstance(value, dict):
        raise TypeError("uniformity loss config must be a mapping, number, boolean, or null")
    alpha = float(value.get("alpha", 2.0))
    if alpha <= 0:
        raise ValueError("uniformity alpha must be positive")
    return {
        "weight": float(value.get("weight", 0.0)),
        "alpha": alpha,
    }


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
