from __future__ import annotations

from typing import Any

from tismir.models.adapters import TemporalTextAdapterBaseline
from tismir.models.baseline import ProjectionBaseline


def build_model(model_config: dict[str, Any], audio_dim: int, text_dim: int):
    """Build a model from config and inferred embedding dimensions."""

    name = model_config.get("name", "projection_baseline")
    if name == "projection_baseline":
        return _build_projection_baseline(model_config, audio_dim=audio_dim, text_dim=text_dim)
    if name == "temporal_text_adapter":
        return _build_temporal_text_adapter(model_config, audio_dim=audio_dim, text_dim=text_dim)
    raise ValueError(f"Unknown model: {name}")


def _build_projection_baseline(model_config: dict[str, Any], audio_dim: int, text_dim: int):
    audio_config = dict(model_config.get("audio", {}))
    text_config = dict(model_config.get("text", {}))
    sim_config = dict(model_config.get("similarity", {}))
    output_dim = int(audio_config.get("output_dim") or text_config.get("output_dim") or 128)
    return ProjectionBaseline(
        audio_dim=int(audio_config.get("input_dim") or audio_dim),
        text_dim=int(text_config.get("input_dim") or text_dim),
        audio_hidden_dim=int(audio_config.get("hidden_dim", 256)),
        text_hidden_dim=int(text_config.get("hidden_dim", 256)),
        output_dim=output_dim,
        temperature=float(sim_config.get("temperature", 0.07)),
        normalize=bool(sim_config.get("normalize", True)),
    )


def _build_temporal_text_adapter(model_config: dict[str, Any], audio_dim: int, text_dim: int):
    audio_config = dict(model_config.get("audio", {}))
    text_config = dict(model_config.get("text", {}))
    adapter_config = dict(model_config.get("adapter", {}))
    cross_config = dict(model_config.get("cross_attention", {}))
    sim_config = dict(model_config.get("similarity", {}))
    model_dim = int(adapter_config.get("model_dim", audio_config.get("output_dim") or text_config.get("output_dim") or 128))
    position_config = _position_config(audio_config.get("positional_encoding", True))
    return TemporalTextAdapterBaseline(
        audio_dim=int(audio_config.get("input_dim") or audio_dim),
        text_dim=int(text_config.get("input_dim") or text_dim),
        model_dim=model_dim,
        audio_hidden_dim=_optional_hidden_dim(audio_config),
        text_hidden_dim=_optional_hidden_dim(text_config),
        audio_layers=int(audio_config.get("num_layers", adapter_config.get("audio_layers", 2))),
        text_layers=int(text_config.get("num_layers", adapter_config.get("text_layers", 1))),
        num_heads=int(adapter_config.get("num_heads", 4)),
        feedforward_dim=int(adapter_config.get("feedforward_dim", model_dim * 4)),
        dropout=float(adapter_config.get("dropout", 0.1)),
        use_audio_positions=position_config["enabled"],
        audio_position_type=position_config["type"],
        rope_base=float(position_config.get("base", 10000.0)),
        cross_attention=bool(cross_config.get("enabled", False)),
        cross_attention_heads=(
            None
            if cross_config.get("num_heads") is None
            else int(cross_config["num_heads"])
        ),
        bidirectional_cross_attention=bool(cross_config.get("bidirectional", False)),
        cross_attention_layers=int(cross_config.get("num_layers", 1)),
        temperature=float(sim_config.get("temperature", 0.07)),
        normalize=bool(sim_config.get("normalize", True)),
    )


def _optional_hidden_dim(config: dict[str, Any]) -> int | None:
    if "hidden_dim" in config and config["hidden_dim"] is None:
        return None
    return int(config.get("hidden_dim", 256))


def _position_config(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        position_type = str(value.get("type", "sinusoidal")).lower()
        return {
            "enabled": position_type != "none",
            "type": position_type,
            "base": value.get("base", 10000.0),
        }
    if isinstance(value, str):
        position_type = value.lower()
        return {
            "enabled": position_type != "none",
            "type": position_type,
            "base": 10000.0,
        }
    enabled = bool(value)
    return {
        "enabled": enabled,
        "type": "sinusoidal" if enabled else "none",
        "base": 10000.0,
    }
