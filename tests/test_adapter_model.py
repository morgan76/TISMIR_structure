import torch

from tismir.models import TemporalTextAdapterBaseline, build_model


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
