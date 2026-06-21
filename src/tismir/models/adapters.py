from __future__ import annotations

import math

from tismir.models.baseline import require_torch


torch, nn = require_torch()


class SinusoidalPositionalEncoding(nn.Module):
    """Add deterministic sinusoidal position information to frame sequences."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def forward(self, values):
        length = values.shape[1]
        device = values.device
        positions = torch.arange(length, device=device, dtype=values.dtype).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, self.dim, 2, device=device, dtype=values.dtype)
            * (-math.log(10000.0) / self.dim)
        )
        encoding = torch.zeros((length, self.dim), device=device, dtype=values.dtype)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        if self.dim > 1:
            encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
        return values + encoding.unsqueeze(0)


class TemporalTextAdapterBaseline(nn.Module):
    """Transformer-adapter baseline for open-vocabulary frame-label scoring."""

    def __init__(
        self,
        audio_dim: int,
        text_dim: int,
        model_dim: int = 128,
        audio_hidden_dim: int | None = 256,
        text_hidden_dim: int | None = 256,
        audio_layers: int = 2,
        text_layers: int = 1,
        num_heads: int = 4,
        feedforward_dim: int = 512,
        dropout: float = 0.1,
        use_audio_positions: bool = True,
        audio_position_type: str | None = None,
        rope_base: float = 10000.0,
        cross_attention: bool = False,
        cross_attention_heads: int | None = None,
        bidirectional_cross_attention: bool = False,
        cross_attention_layers: int = 1,
        temperature: float = 0.07,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if bidirectional_cross_attention and not cross_attention:
            raise ValueError("bidirectional_cross_attention requires cross_attention=True")
        if cross_attention_layers < 1:
            raise ValueError("cross_attention_layers must be positive")

        position_type = _resolve_position_type(use_audio_positions, audio_position_type)
        self.audio_projection = _build_projection(audio_dim, audio_hidden_dim, model_dim)
        self.text_projection = _build_projection(text_dim, text_hidden_dim, model_dim)
        self.audio_positions = (
            SinusoidalPositionalEncoding(model_dim)
            if position_type == "sinusoidal"
            else nn.Identity()
        )
        self.cross_attention_enabled = cross_attention
        self.bidirectional_cross_attention = bidirectional_cross_attention
        cross_heads = cross_attention_heads or num_heads
        if cross_attention and model_dim % cross_heads != 0:
            raise ValueError("model_dim must be divisible by cross_attention_heads")

        if bidirectional_cross_attention:
            self.audio_adapter = _build_audio_transformer_encoder(
                model_dim=model_dim,
                num_layers=audio_layers,
                num_heads=num_heads,
                feedforward_dim=feedforward_dim,
                dropout=dropout,
                position_type=position_type,
                rope_base=rope_base,
            )
            self.text_adapter = TransformerIdentity()
            self.fact_input_block = FactInputBlock(
                model_dim=model_dim,
                cross_attention_heads=cross_heads,
                action_heads=num_heads,
                feedforward_dim=feedforward_dim,
                action_layers=text_layers,
                dropout=dropout,
            )
        else:
            self.audio_adapter = _build_audio_transformer_encoder(
                model_dim=model_dim,
                num_layers=audio_layers,
                num_heads=num_heads,
                feedforward_dim=feedforward_dim,
                dropout=dropout,
                position_type=position_type,
                rope_base=rope_base,
            )
            self.text_adapter = _build_transformer_encoder(
                model_dim=model_dim,
                num_layers=text_layers,
                num_heads=num_heads,
                feedforward_dim=feedforward_dim,
                dropout=dropout,
            )
            self.fact_input_block = None

        if cross_attention:
            if bidirectional_cross_attention:
                self.cross_attention_blocks = nn.ModuleList(
                    [
                        FactUpdateBlock(
                            model_dim=model_dim,
                            cross_attention_heads=cross_heads,
                            frame_heads=num_heads,
                            action_heads=num_heads,
                            feedforward_dim=feedforward_dim,
                            frame_layers=audio_layers,
                            action_layers=text_layers,
                            dropout=dropout,
                            frame_position_type=position_type,
                            rope_base=rope_base,
                        )
                        for _ in range(cross_attention_layers)
                    ]
                )
                self.cross_attention = None
                self.cross_norm = None
                self.cross_dropout = None
            else:
                self.cross_attention_blocks = None
                self.cross_attention = nn.MultiheadAttention(
                    embed_dim=model_dim,
                    num_heads=cross_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                self.cross_norm = nn.LayerNorm(model_dim)
                self.cross_dropout = nn.Dropout(dropout)
        else:
            self.cross_attention_blocks = None
            self.cross_attention = None
            self.cross_norm = None
            self.cross_dropout = None

        self.temperature = temperature
        self.normalize = normalize

    def forward(self, audio, text, audio_mask=None):
        """Return frame-label logits."""

        return self.extract_features(audio, text, audio_mask=audio_mask)["logits"]

    def extract_features(self, audio, text, audio_mask=None):
        """Return final token embeddings and frame-label logits."""

        audio_z = self.audio_projection(audio)
        audio_z = self.audio_positions(audio_z)
        padding_mask = None if audio_mask is None else ~audio_mask.bool()

        if self.bidirectional_cross_attention:
            audio_z = self.audio_adapter(audio_z, src_key_padding_mask=padding_mask)
            text_z, shared_text = self._project_text(text, batch_size=audio_z.shape[0])
            text_z = self.fact_input_block(text_z, audio_z, audio_key_padding_mask=padding_mask)
            for block in self.cross_attention_blocks:
                audio_z, text_z = block(audio_z, text_z, audio_key_padding_mask=padding_mask)
            shared_text = False
        else:
            audio_z = self.audio_adapter(audio_z, src_key_padding_mask=padding_mask)
            text_z, shared_text = self._encode_text(text, batch_size=audio_z.shape[0])
            if self.cross_attention_enabled:
                attended_audio, _ = self.cross_attention(audio_z, text_z, text_z, need_weights=False)
                audio_z = self.cross_norm(audio_z + self.cross_dropout(attended_audio))

        if self.normalize:
            audio_z = torch.nn.functional.normalize(audio_z, dim=-1)
            text_z = torch.nn.functional.normalize(text_z, dim=-1)
            if shared_text:
                text_z = text_z[0]

        similarity = _frame_label_similarity(audio_z, text_z)
        return {
            "audio_tokens": audio_z,
            "text_tokens": text_z,
            "similarity": similarity,
            "logits": similarity / self.temperature,
        }

    def _encode_text(self, text, batch_size: int):
        if text.ndim == 2:
            text_z = self.text_projection(text).unsqueeze(0)
            text_z = self.text_adapter(text_z)
            if self.cross_attention_enabled:
                text_z = text_z.expand(batch_size, -1, -1)
            return text_z, True
        if text.ndim == 3:
            text_z = self.text_projection(text)
            return self.text_adapter(text_z), False
        raise ValueError("text must have shape [K, D] or [B, K, D]")

    def _project_text(self, text, batch_size: int):
        if text.ndim == 2:
            text_z = self.text_projection(text).unsqueeze(0)
            return text_z.expand(batch_size, -1, -1), True
        if text.ndim == 3:
            return self.text_projection(text), False
        raise ValueError("text must have shape [K, D] or [B, K, D]")


def _frame_label_similarity(audio_z, text_z):
    if text_z.ndim == 2:
        return torch.einsum("btd,kd->btk", audio_z, text_z)
    if text_z.ndim == 3:
        return torch.einsum("btd,bkd->btk", audio_z, text_z)
    raise ValueError("text must have shape [K, D] or [B, K, D]")


class FactInputBlock(nn.Module):
    """FACT-style input block: frame features initialize action/label tokens."""

    def __init__(
        self,
        model_dim: int,
        cross_attention_heads: int,
        action_heads: int,
        feedforward_dim: int,
        action_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.text_from_audio = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=cross_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.text_norm = nn.LayerNorm(model_dim)
        self.action_branch = _build_transformer_encoder(
            model_dim=model_dim,
            num_layers=action_layers,
            num_heads=action_heads,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )

    def forward(self, text, audio, audio_key_padding_mask=None):
        attended_text, _ = self.text_from_audio(
            text,
            audio,
            audio,
            key_padding_mask=audio_key_padding_mask,
            need_weights=False,
        )
        text = self.text_norm(text + self.dropout(attended_text))
        return self.action_branch(text)


class FactUpdateBlock(nn.Module):
    """FACT-style update block with self-attention and bidirectional cross-attention."""

    def __init__(
        self,
        model_dim: int,
        cross_attention_heads: int,
        frame_heads: int,
        action_heads: int,
        feedforward_dim: int,
        frame_layers: int,
        action_layers: int,
        dropout: float,
        frame_position_type: str,
        rope_base: float,
    ) -> None:
        super().__init__()
        self.text_from_audio = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=cross_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.action_branch = _build_transformer_encoder(
            model_dim=model_dim,
            num_layers=action_layers,
            num_heads=action_heads,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )
        self.audio_from_text = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=cross_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.text_norm = nn.LayerNorm(model_dim)
        self.audio_norm = nn.LayerNorm(model_dim)
        self.frame_branch = _build_audio_transformer_encoder(
            model_dim=model_dim,
            num_layers=frame_layers,
            num_heads=frame_heads,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
            position_type=frame_position_type,
            rope_base=rope_base,
        )

    def forward(self, audio, text, audio_key_padding_mask=None):
        attended_text, _ = self.text_from_audio(
            text,
            audio,
            audio,
            key_padding_mask=audio_key_padding_mask,
            need_weights=False,
        )
        text = self.text_norm(text + self.dropout(attended_text))
        text = self.action_branch(text)

        attended_audio, _ = self.audio_from_text(audio, text, text, need_weights=False)
        audio = self.audio_norm(audio + self.dropout(attended_audio))
        audio = self.frame_branch(audio, src_key_padding_mask=audio_key_padding_mask)
        return audio, text


def _build_transformer_encoder(
    model_dim: int,
    num_layers: int,
    num_heads: int,
    feedforward_dim: int,
    dropout: float,
):
    if num_layers <= 0:
        return TransformerIdentity()
    layer = nn.TransformerEncoderLayer(
        d_model=model_dim,
        nhead=num_heads,
        dim_feedforward=feedforward_dim,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=False,
    )
    return nn.TransformerEncoder(layer, num_layers=num_layers)


def _build_audio_transformer_encoder(
    model_dim: int,
    num_layers: int,
    num_heads: int,
    feedforward_dim: int,
    dropout: float,
    position_type: str,
    rope_base: float,
):
    if position_type == "rope":
        return RotaryTransformerEncoder(
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
            rope_base=rope_base,
        )
    return _build_transformer_encoder(
        model_dim=model_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        feedforward_dim=feedforward_dim,
        dropout=dropout,
    )


class RotaryTransformerEncoder(nn.Module):
    """Transformer encoder stack whose self-attention uses RoPE on Q/K."""

    def __init__(
        self,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
        rope_base: float,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            self.layers = nn.ModuleList()
        else:
            self.layers = nn.ModuleList(
                [
                    RotaryTransformerEncoderLayer(
                        model_dim=model_dim,
                        num_heads=num_heads,
                        feedforward_dim=feedforward_dim,
                        dropout=dropout,
                        rope_base=rope_base,
                    )
                    for _ in range(num_layers)
                ]
            )

    def forward(self, src, src_key_padding_mask=None):
        values = src
        for layer in self.layers:
            values = layer(values, src_key_padding_mask=src_key_padding_mask)
        return values


class RotaryTransformerEncoderLayer(nn.Module):
    """Post-norm transformer layer with rotary self-attention."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
        rope_base: float,
    ) -> None:
        super().__init__()
        self.self_attention = RotarySelfAttention(
            model_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            rope_base=rope_base,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, model_dim),
        )

    def forward(self, src, src_key_padding_mask=None):
        attended = self.self_attention(src, key_padding_mask=src_key_padding_mask)
        src = self.norm1(src + self.dropout(attended))
        return self.norm2(src + self.dropout(self.feed_forward(src)))


class RotarySelfAttention(nn.Module):
    """Multi-head self-attention with rotary positional embeddings on Q/K."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        dropout: float,
        rope_base: float,
    ) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        head_dim = model_dim // num_heads
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension")
        if rope_base <= 0:
            raise ValueError("rope_base must be positive")

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rope_base = rope_base
        self.qkv_projection = nn.Linear(model_dim, model_dim * 3)
        self.output_projection = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values, key_padding_mask=None):
        batch_size, length, model_dim = values.shape
        qkv = self.qkv_projection(values)
        qkv = qkv.view(batch_size, length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        queries = _apply_rope(queries, base=self.rope_base)
        keys = _apply_rope(keys, base=self.rope_base)

        scale = self.head_dim ** -0.5
        scores = torch.matmul(queries, keys.transpose(-2, -1)) * scale
        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :].bool()
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        context = torch.matmul(attention, values)
        context = context.transpose(1, 2).contiguous().view(batch_size, length, model_dim)
        return self.output_projection(context)


def _apply_rope(values, base: float):
    length = values.shape[-2]
    head_dim = values.shape[-1]
    positions = torch.arange(length, device=values.device, dtype=values.dtype)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=values.device, dtype=values.dtype) / head_dim)
    )
    angles = torch.einsum("t,d->td", positions, inv_freq)
    sin = angles.sin()[None, None, :, :]
    cos = angles.cos()[None, None, :, :]

    even = values[..., 0::2]
    odd = values[..., 1::2]
    rotated = torch.empty_like(values)
    rotated[..., 0::2] = even * cos - odd * sin
    rotated[..., 1::2] = even * sin + odd * cos
    return rotated


def _mask_sequence(values, key_padding_mask=None):
    if key_padding_mask is None:
        return values
    return values.masked_fill(key_padding_mask.unsqueeze(-1).bool(), 0.0)


class TransformerIdentity(nn.Module):
    """Identity module with the same call shape as TransformerEncoder."""

    def forward(self, src, src_key_padding_mask=None):
        return src


def _build_projection(input_dim: int, hidden_dim: int | None, output_dim: int):
    if hidden_dim is None or hidden_dim <= 0:
        return nn.Linear(input_dim, output_dim)
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


def _resolve_position_type(use_audio_positions: bool, audio_position_type: str | None) -> str:
    if audio_position_type is None:
        return "sinusoidal" if use_audio_positions else "none"
    position_type = audio_position_type.lower()
    if position_type not in {"none", "sinusoidal", "rope"}:
        raise ValueError("audio_position_type must be one of: none, sinusoidal, rope")
    return position_type
