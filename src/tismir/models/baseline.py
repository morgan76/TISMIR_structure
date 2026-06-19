from __future__ import annotations


def require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError("Install the 'torch' extra to use model classes.") from exc
    return torch, nn


def build_mlp(input_dim: int, hidden_dim: int, output_dim: int):
    """Build a small projection MLP."""

    _, nn = require_torch()
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


class ProjectionBaseline:
    """Lazy wrapper for the initial open-vocabulary projection model.

    The full torch module will be expanded once training code lands; this class
    keeps the intended public API visible without making torch a hard dependency.
    """

    def __init__(self, audio_dim: int, text_dim: int, hidden_dim: int = 256, output_dim: int = 128):
        torch, nn = require_torch()

        class _Module(nn.Module):
            def __init__(self):
                super().__init__()
                self.audio_projection = build_mlp(audio_dim, hidden_dim, output_dim)
                self.text_projection = build_mlp(text_dim, hidden_dim, output_dim)

            def forward(self, audio, text):
                audio_z = torch.nn.functional.normalize(self.audio_projection(audio), dim=-1)
                text_z = torch.nn.functional.normalize(self.text_projection(text), dim=-1)
                return audio_z @ text_z.transpose(-1, -2)

        self.module = _Module()
