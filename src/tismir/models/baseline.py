from __future__ import annotations


def require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError("Install the 'torch' extra to use model classes.") from exc
    return torch, nn


torch, nn = require_torch()


def build_mlp(input_dim: int, hidden_dim: int, output_dim: int):
    """Build a small projection MLP."""

    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


class ProjectionBaseline(nn.Module):
    """Open-vocabulary frame-label projection baseline."""

    def __init__(
        self,
        audio_dim: int,
        text_dim: int,
        audio_hidden_dim: int = 256,
        text_hidden_dim: int = 256,
        output_dim: int = 128,
        temperature: float = 0.07,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.audio_projection = build_mlp(audio_dim, audio_hidden_dim, output_dim)
        self.text_projection = build_mlp(text_dim, text_hidden_dim, output_dim)
        self.temperature = temperature
        self.normalize = normalize

    def forward(self, audio, text, audio_mask=None):
        """Return frame-label logits."""

        audio_z = self.audio_projection(audio)
        text_z = self.text_projection(text)
        if self.normalize:
            audio_z = torch.nn.functional.normalize(audio_z, dim=-1)
            text_z = torch.nn.functional.normalize(text_z, dim=-1)

        if text_z.ndim == 2:
            logits = torch.einsum("btd,kd->btk", audio_z, text_z)
        elif text_z.ndim == 3:
            logits = torch.einsum("btd,bkd->btk", audio_z, text_z)
        else:
            raise ValueError("text must have shape [K, D] or [B, K, D]")
        return logits / self.temperature
