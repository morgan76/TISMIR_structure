import torch

from tismir.models import TemporalTextAdapterBaseline, build_model
from tismir.models.adapters import _apply_rope


def test_temporal_text_adapter_logits_shape_with_shared_labels():
    model = TemporalTextAdapterBaseline(
        audio_dim=4,
        text_dim=5,
        model_dim=8,
        audio_layers=1,
        text_layers=1,
        num_heads=2,
        feedforward_dim=16,
        dropout=0.0,
        cross_attention=False,
    )
    audio = torch.randn(2, 6, 4)
    text = torch.randn(3, 5)
    mask = torch.tensor(
        [
            [True, True, True, True, True, True],
            [True, True, True, True, False, False],
        ]
    )

    logits = model(audio, text, audio_mask=mask)

    assert logits.shape == (2, 6, 3)


def test_temporal_text_adapter_cross_attention_logits_shape():
    model = build_model(
        {
            "name": "temporal_text_adapter",
            "audio": {"num_layers": 1},
            "text": {"num_layers": 1},
            "adapter": {
                "model_dim": 8,
                "num_heads": 2,
                "feedforward_dim": 16,
                "dropout": 0.0,
            },
            "cross_attention": {"enabled": True},
            "similarity": {"temperature": 0.1, "normalize": True},
        },
        audio_dim=4,
        text_dim=5,
    )
    audio = torch.randn(2, 6, 4)
    text = torch.randn(3, 5)

    logits = model(audio, text)

    assert logits.shape == (2, 6, 3)


def test_temporal_text_adapter_bidirectional_cross_attention_logits_shape():
    model = build_model(
        {
            "name": "temporal_text_adapter",
            "audio": {"num_layers": 1},
            "text": {"num_layers": 1},
            "adapter": {
                "model_dim": 8,
                "num_heads": 2,
                "feedforward_dim": 16,
                "dropout": 0.0,
            },
            "cross_attention": {
                "enabled": True,
                "bidirectional": True,
                "num_layers": 2,
            },
            "similarity": {"temperature": 0.1, "normalize": True},
        },
        audio_dim=4,
        text_dim=5,
    )
    audio = torch.randn(2, 6, 4)
    text = torch.randn(3, 5)

    logits = model(audio, text)

    assert logits.shape == (2, 6, 3)


def test_temporal_text_adapter_bidirectional_uses_no_temporal_convolution():
    model = build_model(
        {
            "name": "temporal_text_adapter",
            "audio": {"num_layers": 1},
            "text": {"num_layers": 1},
            "adapter": {
                "model_dim": 8,
                "num_heads": 2,
                "feedforward_dim": 16,
                "dropout": 0.0,
            },
            "cross_attention": {
                "enabled": True,
                "bidirectional": True,
                "num_layers": 2,
            },
        },
        audio_dim=4,
        text_dim=5,
    )

    assert not any(isinstance(module, torch.nn.Conv1d) for module in model.modules())


def test_temporal_text_adapter_disables_nested_tensor_encoder_path():
    model = build_model(
        {
            "name": "temporal_text_adapter",
            "audio": {"num_layers": 1},
            "text": {"num_layers": 1},
            "adapter": {
                "model_dim": 8,
                "num_heads": 2,
                "feedforward_dim": 16,
                "dropout": 0.0,
            },
        },
        audio_dim=4,
        text_dim=5,
    )

    transformer_encoders = [
        module for module in model.modules() if isinstance(module, torch.nn.TransformerEncoder)
    ]
    assert transformer_encoders
    assert all(not module.enable_nested_tensor for module in transformer_encoders)


def test_temporal_text_adapter_bidirectional_rope_logits_shape():
    model = build_model(
        {
            "name": "temporal_text_adapter",
            "audio": {
                "num_layers": 1,
                "positional_encoding": {"type": "rope", "base": 10000.0},
            },
            "text": {"num_layers": 1},
            "adapter": {
                "model_dim": 8,
                "num_heads": 2,
                "feedforward_dim": 16,
                "dropout": 0.0,
            },
            "cross_attention": {
                "enabled": True,
                "bidirectional": True,
                "num_layers": 2,
            },
            "similarity": {"temperature": 0.1, "normalize": True},
        },
        audio_dim=4,
        text_dim=5,
    )
    audio = torch.randn(2, 6, 4)
    text = torch.randn(3, 5)

    logits = model(audio, text)

    assert logits.shape == (2, 6, 3)


def test_temporal_text_adapter_fact_optional_outputs_and_attention_fusion():
    model = build_model(
        {
            "name": "temporal_text_adapter",
            "audio": {"num_layers": 1},
            "text": {"num_layers": 1},
            "adapter": {
                "model_dim": 8,
                "num_heads": 2,
                "feedforward_dim": 16,
                "dropout": 0.0,
            },
            "cross_attention": {
                "enabled": True,
                "bidirectional": True,
                "num_layers": 2,
                "intermediate_logits": True,
                "return_attention": True,
                "attention_fusion": {"enabled": True, "weight": 0.25},
            },
            "similarity": {"temperature": 0.1, "normalize": True},
        },
        audio_dim=4,
        text_dim=5,
    )
    audio = torch.randn(2, 6, 4)
    text = torch.randn(2, 3, 5)
    mask = torch.tensor(
        [
            [True, True, True, True, True, True],
            [True, True, True, True, False, False],
        ]
    )

    features = model.extract_features(audio, text, audio_mask=mask)

    assert features["logits"].shape == (2, 6, 3)
    assert features["direct_logits"].shape == (2, 6, 3)
    assert not torch.allclose(features["logits"], features["direct_logits"])
    assert len(features["intermediate_logits"]) == 2
    assert all(logits.shape == (2, 6, 3) for logits in features["intermediate_logits"])
    assert len(features["attention_maps"]) == 2
    assert features["attention_maps"][0]["frame_to_text"].shape == (2, 6, 3)
    assert features["attention_maps"][0]["label_to_frame"].shape == (2, 3, 6)


def test_temporal_text_adapter_bidirectional_requires_cross_attention():
    try:
        TemporalTextAdapterBaseline(
            audio_dim=4,
            text_dim=5,
            model_dim=8,
            num_heads=2,
            feedforward_dim=16,
            cross_attention=False,
            bidirectional_cross_attention=True,
        )
    except ValueError as exc:
        assert "requires cross_attention=True" in str(exc)
    else:
        raise AssertionError("Expected disabled cross-attention to fail for bidirectional mode")


def test_temporal_text_adapter_rope_logits_shape():
    model = build_model(
        {
            "name": "temporal_text_adapter",
            "audio": {
                "num_layers": 1,
                "positional_encoding": {"type": "rope", "base": 10000.0},
            },
            "text": {"num_layers": 1},
            "adapter": {
                "model_dim": 8,
                "num_heads": 2,
                "feedforward_dim": 16,
                "dropout": 0.0,
            },
            "similarity": {"temperature": 0.1, "normalize": True},
        },
        audio_dim=4,
        text_dim=5,
    )
    audio = torch.randn(2, 6, 4)
    text = torch.randn(3, 5)

    logits = model(audio, text)

    assert logits.shape == (2, 6, 3)


def test_temporal_text_adapter_rope_requires_even_head_dim():
    try:
        build_model(
            {
                "name": "temporal_text_adapter",
                "audio": {
                    "num_layers": 1,
                    "positional_encoding": "rope",
                },
                "adapter": {
                    "model_dim": 6,
                    "num_heads": 2,
                    "feedforward_dim": 12,
                    "dropout": 0.0,
                },
            },
            audio_dim=4,
            text_dim=5,
        )
    except ValueError as exc:
        assert "even attention head dimension" in str(exc)
    else:
        raise AssertionError("Expected RoPE odd head dimension to fail")


def test_rope_rotation_preserves_pairwise_norms_and_position_zero():
    values = torch.randn(2, 3, 5, 8)

    rotated = _apply_rope(values, base=10000.0)

    original_pair_norms = values.reshape(*values.shape[:-1], -1, 2).norm(dim=-1)
    rotated_pair_norms = rotated.reshape(*rotated.shape[:-1], -1, 2).norm(dim=-1)
    assert torch.allclose(rotated_pair_norms, original_pair_norms, atol=1e-6)
    assert torch.allclose(rotated[:, :, 0], values[:, :, 0], atol=1e-6)
