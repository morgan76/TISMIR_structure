import torch

from tismir.models import TemporalTextAdapterBaseline, build_model
from tismir.models.adapters import TransformerIdentity, _apply_rope


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


def test_temporal_text_adapter_supports_named_architecture_blocks():
    model = build_model(
        {
            "name": "temporal_text_adapter",
            "model_dim": 8,
            "num_heads": 2,
            "feedforward_dim": 16,
            "dropout": 0.0,
            "audio_adapter": {"num_layers": 1},
            "text_adapter": {"num_layers": 1},
            "update_blocks": {
                "enabled": True,
                "bidirectional": True,
                "num_blocks": 2,
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
    assert len(model.cross_attention_blocks) == 2
    assert not isinstance(model.audio_adapter, TransformerIdentity)
    assert not isinstance(model.text_adapter, TransformerIdentity)
    assert not hasattr(model, "final_audio_from_text")
    assert hasattr(model.fact_input_block, "text_from_audio")
    assert hasattr(model.cross_attention_blocks[0], "section_branch")


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


def test_temporal_text_adapter_relation_attention_outputs_link_logits():
    model = build_model(
        {
            "name": "temporal_text_adapter",
            "model_dim": 8,
            "num_heads": 2,
            "feedforward_dim": 16,
            "dropout": 0.0,
            "audio_adapter": {"num_layers": 1},
            "text_adapter": {"num_layers": 1},
            "update_blocks": {
                "enabled": True,
                "bidirectional": True,
                "num_blocks": 2,
                "relation_attention": {
                    "enabled": True,
                    "edge_dim": 8,
                    "link_cnn": {
                        "kernel_size": 3,
                        "dilations": [1, 2],
                        "dropout": 0.0,
                        "ema_factor": 4,
                    },
                    "gate_init": -4.0,
                    "relative_distance": True,
                    "max_distance": 16,
                },
            },
            "similarity": {"temperature": 0.1, "normalize": True},
        },
        audio_dim=4,
        text_dim=5,
    )
    audio = torch.randn(2, 6, 4)
    text = torch.randn(2, 3, 5)

    features = model.extract_features(audio, text)

    assert features["logits"].shape == (2, 6, 3)
    assert len(features["link_logits"]) == 2
    assert all(logits.shape == (2, 6, 6, 3) for logits in features["link_logits"])
    assert model.cross_attention_blocks[0].frame_branch.link_cnn.input_bn.num_features == 2


def test_temporal_text_adapter_relation_attention_pair_features_are_configurable():
    for pair_features, expected_channels in [
        (["cosine"], 1),
        (["probability"], 1),
        (["cosine", "probability"], 2),
    ]:
        model = build_model(
            {
                "name": "temporal_text_adapter",
                "model_dim": 8,
                "num_heads": 2,
                "feedforward_dim": 16,
                "dropout": 0.0,
                "audio_adapter": {"num_layers": 1},
                "text_adapter": {"num_layers": 1},
                "update_blocks": {
                    "enabled": True,
                    "bidirectional": True,
                    "num_blocks": 1,
                    "relation_attention": {
                        "enabled": True,
                        "edge_dim": 8,
                        "pair_features": pair_features,
                        "link_cnn": {
                            "kernel_size": 3,
                            "dilations": [1],
                            "dropout": 0.0,
                            "ema_factor": 4,
                        },
                        "gate_init": -4.0,
                        "relative_distance": False,
                        "max_distance": 16,
                    },
                },
                "similarity": {"temperature": 0.1, "normalize": True},
            },
            audio_dim=4,
            text_dim=5,
        )
        audio = torch.randn(2, 6, 4)
        text = torch.randn(2, 3, 5)

        features = model.extract_features(audio, text)

        assert features["logits"].shape == (2, 6, 3)
        assert features["link_logits"][0].shape == (2, 6, 6, 3)
        assert (
            model.cross_attention_blocks[0].frame_branch.link_cnn.input_bn.num_features
            == expected_channels
        )


def test_temporal_text_adapter_relation_attention_outputs_boundary_logits():
    model = build_model(
        {
            "name": "temporal_text_adapter",
            "model_dim": 8,
            "num_heads": 2,
            "feedforward_dim": 16,
            "dropout": 0.0,
            "audio_adapter": {"num_layers": 1},
            "text_adapter": {"num_layers": 1},
            "update_blocks": {
                "enabled": True,
                "bidirectional": True,
                "num_blocks": 2,
                "boundary_head": {
                    "enabled": True,
                    "hidden_dim": 12,
                },
                "relation_attention": {
                    "enabled": True,
                    "edge_dim": 8,
                    "pair_features": ["cosine"],
                    "link_cnn": {
                        "kernel_size": 3,
                        "dilations": [1],
                        "dropout": 0.0,
                        "ema_factor": 4,
                    },
                    "gate_init": -4.0,
                    "relative_distance": False,
                    "max_distance": 16,
                },
            },
            "similarity": {"temperature": 0.1, "normalize": True},
        },
        audio_dim=4,
        text_dim=5,
    )
    audio = torch.randn(2, 6, 4)
    text = torch.randn(2, 3, 5)

    features = model.extract_features(audio, text)

    assert len(features["boundary_logits"]) == 2
    assert all(logits.shape == (2, 5) for logits in features["boundary_logits"])


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


def test_temporal_text_adapter_optional_structure_head_outputs_tokens():
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
            "structure_head": {
                "enabled": True,
                "hidden_dim": 12,
                "output_dim": 6,
            },
        },
        audio_dim=4,
        text_dim=5,
    )
    audio = torch.randn(2, 6, 4)
    text = torch.randn(3, 5)

    features = model.extract_features(audio, text)

    assert features["structure_tokens"].shape == (2, 6, 6)
    norms = torch.linalg.vector_norm(features["structure_tokens"], dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
    assert model.structure_pair_logits(features["structure_tokens"]).shape == (2, 6, 6, 3)


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
