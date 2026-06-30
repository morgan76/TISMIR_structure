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
    audio_config = _named_config_block(model_config, preferred="audio_adapter", legacy="audio")
    text_config = _named_config_block(model_config, preferred="text_adapter", legacy="text")
    adapter_config = dict(model_config.get("adapter", {}))
    adapter_config.update(_top_level_adapter_config(model_config))
    cross_config = _named_config_block(
        model_config,
        preferred="update_blocks",
        legacy="cross_attention",
    )
    sim_config = dict(model_config.get("similarity", {}))
    structure_config = dict(model_config.get("structure_head", {}))
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
        cross_attention_layers=int(cross_config.get("num_blocks", cross_config.get("num_layers", 1))),
        return_intermediate_logits=bool(cross_config.get("intermediate_logits", False)),
        return_attention=bool(cross_config.get("return_attention", False)),
        attention_fusion_weight=_attention_fusion_weight(cross_config.get("attention_fusion")),
        relation_attention=_relation_attention_config(cross_config.get("relation_attention")),
        boundary_head=_boundary_head_config(cross_config.get("boundary_head")),
        structure_head_dim=_structure_head_dim(structure_config, model_dim),
        structure_head_hidden_dim=_structure_head_hidden_dim(structure_config, model_dim),
        structure_pair_hidden_dim=_structure_pair_hidden_dim(structure_config, model_dim),
        temperature=float(sim_config.get("temperature", 0.07)),
        normalize=bool(sim_config.get("normalize", True)),
    )


def _named_config_block(
    model_config: dict[str, Any],
    *,
    preferred: str,
    legacy: str,
) -> dict[str, Any]:
    config = dict(model_config.get(legacy, {}))
    config.update(dict(model_config.get(preferred, {})))
    return config


def _top_level_adapter_config(model_config: dict[str, Any]) -> dict[str, Any]:
    keys = ("model_dim", "num_heads", "feedforward_dim", "dropout")
    return {key: model_config[key] for key in keys if key in model_config}


def _optional_hidden_dim(config: dict[str, Any]) -> int | None:
    if "hidden_dim" in config and config["hidden_dim"] is None:
        return None
    return int(config.get("hidden_dim", 256))


def _attention_fusion_weight(value: Any) -> float:
    if value in (None, False):
        return 0.0
    if value is True:
        return 0.5
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, dict):
        raise TypeError("cross_attention.attention_fusion must be a mapping, number, boolean, or null")
    if not bool(value.get("enabled", True)):
        return 0.0
    return float(value.get("weight", value.get("alpha", 0.5)))


def _relation_attention_config(value: Any) -> dict[str, Any]:
    if value in (None, False):
        return {"enabled": False}
    if value is True:
        value = {}
    if not isinstance(value, dict):
        raise TypeError("update_blocks.relation_attention must be a mapping, boolean, or null")
    if not bool(value.get("enabled", True)):
        return {"enabled": False}
    link_cnn = dict(value.get("link_cnn", {}))
    return {
        "enabled": True,
        "edge_dim": int(value.get("edge_dim", 32)),
        "kernel_size": int(link_cnn.get("kernel_size", value.get("kernel_size", 5))),
        "dilations": tuple(
            int(dilation)
            for dilation in link_cnn.get("dilations", value.get("dilations", (1, 2, 4, 8, 16, 32, 64)))
        ),
        "dropout": float(link_cnn.get("dropout", value.get("dropout", 0.2))),
        "ema_factor": int(link_cnn.get("ema_factor", value.get("ema_factor", 4))),
        "gate_init": float(value.get("gate_init", -4.0)),
        "relative_distance": bool(value.get("relative_distance", True)),
        "max_distance": int(value.get("max_distance", 2048)),
        "pair_features": _relation_pair_features(value.get("pair_features")),
    }


def _boundary_head_config(value: Any) -> dict[str, Any]:
    if value in (None, False):
        return {"enabled": False, "hidden_dim": None}
    if value is True:
        value = {}
    if not isinstance(value, dict):
        raise TypeError("update_blocks.boundary_head must be a mapping, boolean, or null")
    if not bool(value.get("enabled", True)):
        return {"enabled": False, "hidden_dim": None}
    hidden_dim = value.get("hidden_dim")
    return {
        "enabled": True,
        "hidden_dim": None if hidden_dim is None else int(hidden_dim),
    }


def _structure_head_dim(config: dict[str, Any], model_dim: int) -> int | None:
    if not bool(config.get("enabled", False)):
        return None
    return int(config.get("output_dim", model_dim))


def _relation_pair_features(value: Any) -> tuple[str, ...]:
    if value is None:
        return ("probability", "cosine")
    if isinstance(value, str):
        return (value,)
    try:
        return tuple(str(feature) for feature in value)
    except TypeError as exc:
        raise TypeError("relation_attention.pair_features must be a string or sequence") from exc


def _structure_head_hidden_dim(config: dict[str, Any], model_dim: int) -> int | None:
    if not bool(config.get("enabled", False)):
        return None
    if "hidden_dim" in config and config["hidden_dim"] is None:
        return None
    return int(config.get("hidden_dim", model_dim * 2))


def _structure_pair_hidden_dim(config: dict[str, Any], model_dim: int) -> int | None:
    if not bool(config.get("enabled", False)):
        return None
    if "pair_hidden_dim" in config and config["pair_hidden_dim"] is None:
        return None
    return int(config.get("pair_hidden_dim", model_dim * 2))


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
