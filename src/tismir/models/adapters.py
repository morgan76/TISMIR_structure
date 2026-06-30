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
        return_intermediate_logits: bool = False,
        return_attention: bool = False,
        attention_fusion_weight: float = 0.0,
        relation_attention: dict | None = None,
        boundary_head: dict | None = None,
        structure_head_dim: int | None = None,
        structure_head_hidden_dim: int | None = None,
        structure_pair_hidden_dim: int | None = None,
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
        if not 0.0 <= attention_fusion_weight <= 1.0:
            raise ValueError("attention_fusion_weight must be between 0 and 1")

        position_type = _resolve_position_type(use_audio_positions, audio_position_type)
        boundary_head_config = _boundary_head_config(boundary_head)
        self.audio_projection = _build_projection(audio_dim, audio_hidden_dim, model_dim)
        self.text_projection = _build_projection(text_dim, text_hidden_dim, model_dim)
        self.audio_positions = (
            SinusoidalPositionalEncoding(model_dim)
            if position_type == "sinusoidal"
            else nn.Identity()
        )
        self.cross_attention_enabled = cross_attention
        self.bidirectional_cross_attention = bidirectional_cross_attention
        self.return_intermediate_logits = return_intermediate_logits
        self.return_attention = return_attention
        self.attention_fusion_weight = attention_fusion_weight
        relation_attention_config = _relation_attention_config(relation_attention)
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
            self.text_adapter = _build_transformer_encoder(
                model_dim=model_dim,
                num_layers=text_layers,
                num_heads=num_heads,
                feedforward_dim=feedforward_dim,
                dropout=dropout,
            )
            self.fact_input_block = FactInputBlock(
                model_dim=model_dim,
                cross_attention_heads=cross_heads,
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
                            section_heads=num_heads,
                            feedforward_dim=feedforward_dim,
                            frame_layers=audio_layers,
                            section_layers=text_layers,
                            dropout=dropout,
                            frame_position_type=position_type,
                            rope_base=rope_base,
                            similarity_temperature=temperature,
                            relation_attention_config=relation_attention_config,
                            boundary_head_config=boundary_head_config,
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
                self.final_audio_from_text = None
                self.final_audio_norm = None
                self.final_audio_dropout = None
        else:
            self.cross_attention_blocks = None
            self.cross_attention = None
            self.cross_norm = None
            self.cross_dropout = None
            self.final_audio_from_text = None
            self.final_audio_norm = None
            self.final_audio_dropout = None

        self.temperature = temperature
        self.normalize = normalize
        self.structure_projection = (
            None
            if structure_head_dim is None
            else _build_projection(model_dim, structure_head_hidden_dim, structure_head_dim)
        )
        self.structure_pair_classifier = (
            None
            if structure_head_dim is None
            else _build_pair_classifier(
                input_dim=structure_head_dim * 2,
                hidden_dim=structure_pair_hidden_dim,
                output_dim=3,
            )
        )

    def forward(self, audio, text, audio_mask=None):
        """Return frame-label logits."""

        return self.extract_features(audio, text, audio_mask=audio_mask)["logits"]

    def extract_features(self, audio, text, audio_mask=None):
        """Return final token embeddings and frame-label logits."""

        audio_z = self.audio_projection(audio)
        audio_z = self.audio_positions(audio_z)
        padding_mask = None if audio_mask is None else ~audio_mask.bool()
        intermediate_logits = []
        attention_maps = []
        link_logits = []
        boundary_logits = []
        fusion_attention = None

        if self.bidirectional_cross_attention:
            audio_z = self.audio_adapter(audio_z, src_key_padding_mask=padding_mask)
            text_z, shared_text = self._encode_text(text, batch_size=audio_z.shape[0])
            text_z = self.fact_input_block(text_z, audio_z, audio_key_padding_mask=padding_mask)
            need_weights = self.return_attention or self.attention_fusion_weight > 0.0
            for block in self.cross_attention_blocks:
                block_output = block(
                    audio_z,
                    text_z,
                    audio_key_padding_mask=padding_mask,
                    need_weights=need_weights,
                )
                audio_z, text_z = block_output.audio, block_output.text
                if block_output.link_logits is not None:
                    link_logits.append(block_output.link_logits)
                if block_output.boundary_logits is not None:
                    boundary_logits.append(block_output.boundary_logits)
                if block_output.attention is not None:
                    attention_maps.append(block_output.attention)
                    fusion_attention = block_output.attention.get("frame_to_text")
                if self.return_intermediate_logits:
                    block_audio = torch.nn.functional.normalize(audio_z, dim=-1)
                    block_text = torch.nn.functional.normalize(text_z, dim=-1)
                    intermediate_logits.append(
                        _frame_label_similarity(block_audio, block_text) / self.temperature
                    )
            shared_text = False
        else:
            audio_z = self.audio_adapter(audio_z, src_key_padding_mask=padding_mask)
            text_z, shared_text = self._encode_text(text, batch_size=audio_z.shape[0])
            if self.cross_attention_enabled:
                need_weights = self.return_attention or self.attention_fusion_weight > 0.0
                attended_audio, weights = self.cross_attention(
                    audio_z,
                    text_z,
                    text_z,
                    need_weights=need_weights,
                    average_attn_weights=True,
                )
                audio_z = self.cross_norm(audio_z + self.cross_dropout(attended_audio))
                if need_weights:
                    fusion_attention = weights
                    attention_maps.append({"frame_to_text": weights})

        if self.normalize:
            audio_z = torch.nn.functional.normalize(audio_z, dim=-1)
            text_z = torch.nn.functional.normalize(text_z, dim=-1)
            if shared_text:
                text_z = text_z[0]

        structure_tokens = None
        if self.structure_projection is not None:
            structure_tokens = torch.nn.functional.normalize(
                self.structure_projection(audio_z),
                dim=-1,
            )

        similarity = _frame_label_similarity(audio_z, text_z)
        direct_logits = similarity / self.temperature
        logits = direct_logits
        if self.attention_fusion_weight > 0.0:
            if fusion_attention is None:
                raise ValueError("attention_fusion requires cross-attention weights")
            attention_logits = torch.log(fusion_attention.clamp_min(1e-8))
            logits = (
                (1.0 - self.attention_fusion_weight) * direct_logits
                + self.attention_fusion_weight * attention_logits
            )
        return {
            "audio_tokens": audio_z,
            "text_tokens": text_z,
            "similarity": similarity,
            "direct_logits": direct_logits,
            "logits": logits,
            "structure_tokens": structure_tokens,
            "intermediate_logits": intermediate_logits,
            "attention_maps": attention_maps,
            "link_logits": link_logits,
            "boundary_logits": boundary_logits,
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

    def structure_pair_logits(self, structure_tokens):
        """Return pair-relation logits from structure-head tokens."""

        if self.structure_pair_classifier is None:
            raise ValueError("structure pair classifier requires structure_head.enabled=true")
        left = structure_tokens.unsqueeze(2).expand(
            -1,
            -1,
            structure_tokens.shape[1],
            -1,
        )
        right = structure_tokens.unsqueeze(1).expand(
            -1,
            structure_tokens.shape[1],
            -1,
            -1,
        )
        pair_features = torch.cat([left, right], dim=-1)
        return self.structure_pair_classifier(pair_features)


def _frame_label_similarity(audio_z, text_z):
    if text_z.ndim == 2:
        return torch.einsum("btd,kd->btk", audio_z, text_z)
    if text_z.ndim == 3:
        return torch.einsum("btd,bkd->btk", audio_z, text_z)
    raise ValueError("text must have shape [K, D] or [B, K, D]")


class FactInputBlock(nn.Module):
    """Input block where section label tokens read from contextualized audio."""

    def __init__(
        self,
        model_dim: int,
        cross_attention_heads: int,
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

    def forward(self, text, audio, audio_key_padding_mask=None):
        attended_text, _ = self.text_from_audio(
            text,
            audio,
            audio,
            key_padding_mask=audio_key_padding_mask,
            need_weights=False,
        )
        return self.text_norm(text + self.dropout(attended_text))


class FactUpdateBlock(nn.Module):
    """Update audio from sections, refine audio relations, then update sections."""

    def __init__(
        self,
        model_dim: int,
        cross_attention_heads: int,
        frame_heads: int,
        section_heads: int,
        feedforward_dim: int,
        frame_layers: int,
        section_layers: int,
        dropout: float,
        frame_position_type: str,
        rope_base: float,
        similarity_temperature: float,
        relation_attention_config: dict,
        boundary_head_config: dict,
    ) -> None:
        super().__init__()
        self.text_from_audio = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=cross_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.section_branch = _build_transformer_encoder(
            model_dim=model_dim,
            num_layers=section_layers,
            num_heads=section_heads,
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
        self.relation_attention_enabled = bool(relation_attention_config["enabled"])
        if boundary_head_config["enabled"] and not self.relation_attention_enabled:
            raise ValueError("boundary_head requires relation_attention.enabled=true")
        if self.relation_attention_enabled:
            self.frame_branch = RelationAwareFrameBranch(
                model_dim=model_dim,
                num_heads=frame_heads,
                feedforward_dim=feedforward_dim,
                dropout=dropout,
                edge_dim=int(relation_attention_config["edge_dim"]),
                link_cnn_dropout=float(relation_attention_config["dropout"]),
                link_cnn_kernel_size=int(relation_attention_config["kernel_size"]),
                link_cnn_dilations=tuple(relation_attention_config["dilations"]),
                ema_factor=int(relation_attention_config["ema_factor"]),
                gate_init=float(relation_attention_config["gate_init"]),
                relative_distance=bool(relation_attention_config["relative_distance"]),
                max_distance=int(relation_attention_config["max_distance"]),
                position_type=frame_position_type,
                rope_base=rope_base,
                similarity_temperature=similarity_temperature,
                pair_features=tuple(relation_attention_config["pair_features"]),
                boundary_head_config=boundary_head_config,
            )
        else:
            self.frame_branch = _build_audio_transformer_encoder(
                model_dim=model_dim,
                num_layers=frame_layers,
                num_heads=frame_heads,
                feedforward_dim=feedforward_dim,
                dropout=dropout,
                position_type=frame_position_type,
                rope_base=rope_base,
            )

    def forward(self, audio, text, audio_key_padding_mask=None, need_weights: bool = False):
        attended_audio, frame_to_text = self.audio_from_text(
            audio,
            text,
            text,
            need_weights=need_weights,
            average_attn_weights=True,
        )
        audio = self.audio_norm(audio + self.dropout(attended_audio))
        link_logits = None
        boundary_logits = None
        if self.relation_attention_enabled:
            audio, link_logits, boundary_logits = self.frame_branch(
                audio,
                text,
                key_padding_mask=audio_key_padding_mask,
            )
        else:
            audio = self.frame_branch(audio, src_key_padding_mask=audio_key_padding_mask)

        attended_text, label_to_frame = self.text_from_audio(
            text,
            audio,
            audio,
            key_padding_mask=audio_key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=True,
        )
        text = self.text_norm(text + self.dropout(attended_text))
        text = self.section_branch(text)
        attention = None
        if need_weights:
            attention = {
                "label_to_frame": label_to_frame,
                "frame_to_text": frame_to_text,
            }
        return FactUpdateOutput(
            audio=audio,
            text=text,
            attention=attention,
            link_logits=link_logits,
            boundary_logits=boundary_logits,
        )


class FactUpdateOutput:
    """Container for one FACT update block output."""

    def __init__(self, audio, text, attention=None, link_logits=None, boundary_logits=None) -> None:
        self.audio = audio
        self.text = text
        self.attention = attention
        self.link_logits = link_logits
        self.boundary_logits = boundary_logits


class RelationAwareFrameBranch(nn.Module):
    """LinkSeg-style edge extractor plus relation-aware frame self-attention."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
        edge_dim: int,
        link_cnn_dropout: float,
        link_cnn_kernel_size: int,
        link_cnn_dilations: tuple[int, ...],
        ema_factor: int,
        gate_init: float,
        relative_distance: bool,
        max_distance: int,
        position_type: str,
        rope_base: float,
        similarity_temperature: float,
        pair_features: tuple[str, ...],
        boundary_head_config: dict,
    ) -> None:
        super().__init__()
        if similarity_temperature <= 0:
            raise ValueError("similarity_temperature must be positive")
        self.similarity_temperature = similarity_temperature
        self.pair_features = _normalize_relation_pair_features(pair_features)
        self.link_cnn = LinkSegConvNetSSM(
            input_channels=len(self.pair_features),
            output_channels=edge_dim,
            shape=link_cnn_kernel_size,
            dilations=link_cnn_dilations,
            dropout=link_cnn_dropout,
            ema_factor=ema_factor,
        )
        self.relative_distance = (
            nn.Embedding(max_distance + 1, edge_dim) if relative_distance else None
        )
        self.link_classifier = nn.Linear(edge_dim, 3)
        self.boundary_head = (
            None
            if not boundary_head_config["enabled"]
            else _build_pair_classifier(
                input_dim=model_dim * 2 + edge_dim,
                hidden_dim=boundary_head_config["hidden_dim"],
                output_dim=1,
            )
        )
        self.self_attention = RelationAwareSelfAttention(
            model_dim=model_dim,
            edge_dim=edge_dim,
            num_heads=num_heads,
            dropout=dropout,
            gate_init=gate_init,
            position_type=position_type,
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

    def forward(self, audio, text, key_padding_mask=None):
        edge_features = self._edge_features(audio, text)
        link_logits = self.link_classifier(edge_features)
        attended = self.self_attention(
            audio,
            edge_features=edge_features,
            key_padding_mask=key_padding_mask,
        )
        audio = self.norm1(audio + self.dropout(attended))
        audio = self.norm2(audio + self.dropout(self.feed_forward(audio)))
        boundary_logits = self._boundary_logits(audio, edge_features)
        return audio, link_logits, boundary_logits

    def _boundary_logits(self, audio, edge_features):
        if self.boundary_head is None:
            return None
        length = audio.shape[1]
        if length < 2:
            return audio.new_zeros((audio.shape[0], 0))
        positions = torch.arange(length - 1, device=audio.device)
        left = audio[:, :-1, :]
        right = audio[:, 1:, :]
        consecutive_edges = edge_features[:, positions, positions + 1, :]
        boundary_features = torch.cat([left, right, consecutive_edges], dim=-1)
        return self.boundary_head(boundary_features).squeeze(-1)

    def _edge_features(self, audio, text):
        normalized_audio = torch.nn.functional.normalize(audio, dim=-1)
        normalized_text = torch.nn.functional.normalize(text, dim=-1)
        pair_maps = []
        audio_similarity = None
        label_similarity = None
        if "cosine" in self.pair_features:
            audio_similarity = torch.einsum(
                "bid,bjd->bij",
                normalized_audio,
                normalized_audio,
            )
        if "probability" in self.pair_features:
            frame_label_logits = _frame_label_similarity(
                normalized_audio,
                normalized_text,
            ) / self.similarity_temperature
            frame_label_probs = torch.softmax(frame_label_logits, dim=-1)
            label_similarity = torch.einsum(
                "bik,bjk->bij",
                frame_label_probs,
                frame_label_probs,
            )
        for feature in self.pair_features:
            if feature == "cosine":
                assert audio_similarity is not None
                pair_maps.append(audio_similarity)
            elif feature == "probability":
                assert label_similarity is not None
                pair_maps.append(label_similarity)
            else:  # pragma: no cover - normalized at construction time
                raise ValueError(f"Unknown relation pair feature: {feature}")
        pair_map = torch.stack(pair_maps, dim=1)
        edge_features = self.link_cnn(pair_map).permute(0, 2, 3, 1).contiguous()
        if self.relative_distance is not None:
            length = audio.shape[1]
            positions = torch.arange(length, device=audio.device)
            distances = (positions[:, None] - positions[None, :]).abs()
            distances = distances.clamp_max(self.relative_distance.num_embeddings - 1)
            edge_features = edge_features + self.relative_distance(distances).unsqueeze(0)
        return edge_features


def _normalize_relation_pair_features(pair_features: tuple[str, ...]) -> tuple[str, ...]:
    aliases = {
        "audio": "cosine",
        "audio_cosine": "cosine",
        "cosine": "cosine",
        "label": "probability",
        "label_probability": "probability",
        "prob": "probability",
        "probability": "probability",
        "same_label_probability": "probability",
    }
    normalized = []
    for feature in pair_features:
        key = str(feature).strip().lower().replace("-", "_")
        if key not in aliases:
            raise ValueError(
                "relation_attention.pair_features must contain only "
                "'cosine' and/or 'probability'"
            )
        canonical = aliases[key]
        if canonical not in normalized:
            normalized.append(canonical)
    if not normalized:
        raise ValueError("relation_attention.pair_features must not be empty")
    return tuple(normalized)


class RelationAwareSelfAttention(nn.Module):
    """Self-attention whose logits are biased by LinkSeg edge features."""

    def __init__(
        self,
        model_dim: int,
        edge_dim: int,
        num_heads: int,
        dropout: float,
        gate_init: float,
        position_type: str,
        rope_base: float,
    ) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.position_type = position_type
        self.rope_base = rope_base
        if position_type == "rope" and self.head_dim % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension")
        self.qkv_projection = nn.Linear(model_dim, model_dim * 3)
        self.edge_bias = nn.Linear(edge_dim, num_heads)
        self.output_projection = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.edge_gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, values, edge_features, key_padding_mask=None):
        batch_size, length, model_dim = values.shape
        qkv = self.qkv_projection(values)
        qkv = qkv.view(batch_size, length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        if self.position_type == "rope":
            queries = _apply_rope(queries, base=self.rope_base)
            keys = _apply_rope(keys, base=self.rope_base)

        scores = torch.matmul(queries, keys.transpose(-2, -1)) * (self.head_dim ** -0.5)
        edge_bias = self.edge_bias(edge_features).permute(0, 3, 1, 2)
        scores = scores + torch.sigmoid(self.edge_gate) * edge_bias
        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :].bool()
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        context = torch.matmul(attention, values)
        context = context.transpose(1, 2).contiguous().view(batch_size, length, model_dim)
        return self.output_projection(context)


class LinkSegConvNetSSM(nn.Module):
    """LinkSeg ConvNetSSM link feature extractor."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        shape: int,
        dilations: tuple[int, ...],
        dropout: float,
        ema_factor: int,
    ) -> None:
        super().__init__()
        self.input_bn = nn.BatchNorm2d(input_channels, affine=False, track_running_stats=False)
        layers = []
        channels = input_channels
        for dilation in dilations:
            layers.append(
                LinkSegConv2d(
                    input_channels=channels,
                    output_channels=output_channels,
                    shape=shape,
                    dilation=dilation,
                    dropout=dropout,
                    ema_factor=ema_factor,
                )
            )
            channels = output_channels
        self.layers = nn.ModuleList(layers)

    def forward(self, values):
        values = self.input_bn(values)
        for layer in self.layers:
            values = layer(values)
        return values


class LinkSegConv2d(nn.Module):
    """LinkSeg 2D convolutional layer with EMA attention and residual 1x1 path."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        shape: int,
        dilation: int,
        dropout: float,
        ema_factor: int,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            input_channels,
            output_channels,
            shape,
            padding="same",
            dilation=dilation,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(output_channels, affine=False, track_running_stats=False)
        self.conv_1x1 = nn.Conv2d(
            input_channels,
            output_channels,
            1,
            padding="same",
            bias=True,
        )
        self.bn_1x1 = nn.BatchNorm2d(output_channels, affine=False, track_running_stats=False)
        self.activation = nn.ELU()
        self.attention = LinkSegEMA(output_channels, factor=ema_factor)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values):
        residual = values
        values = self.conv(values)
        values = self.attention(values)
        values = self.bn(values)
        values = self.bn_1x1(self.conv_1x1(residual)) + values
        values = self.activation(values)
        return self.dropout(values)


class LinkSegEMA(nn.Module):
    """EMA spatial/channel attention module used by LinkSeg."""

    def __init__(self, channels: int, factor: int = 4) -> None:
        super().__init__()
        if factor < 1:
            raise ValueError("ema_factor must be positive")
        if channels // factor <= 0 or channels % factor != 0:
            raise ValueError("edge_dim must be positive and divisible by ema_factor")
        self.groups = factor
        group_channels = channels // self.groups
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(group_channels, group_channels)
        self.conv1x1 = nn.Conv2d(group_channels, group_channels, kernel_size=1)
        self.conv3x3 = nn.Conv2d(group_channels, group_channels, kernel_size=3, padding=1)

    def forward(self, values):
        batch, channels, height, width = values.shape
        group_values = values.reshape(batch * self.groups, -1, height, width)
        pooled_h = self.pool_h(group_values)
        pooled_w = self.pool_w(group_values).permute(0, 1, 3, 2)
        height_width = self.conv1x1(torch.cat([pooled_h, pooled_w], dim=2))
        gate_h, gate_w = torch.split(height_width, [height, width], dim=2)
        values_1 = self.gn(
            group_values
            * gate_h.sigmoid()
            * gate_w.permute(0, 1, 3, 2).sigmoid()
        )
        values_2 = self.conv3x3(group_values)
        weights_11 = self.softmax(
            self.agp(values_1).reshape(batch * self.groups, -1, 1).permute(0, 2, 1)
        )
        weights_12 = values_2.reshape(batch * self.groups, channels // self.groups, -1)
        weights_21 = self.softmax(
            self.agp(values_2).reshape(batch * self.groups, -1, 1).permute(0, 2, 1)
        )
        weights_22 = values_1.reshape(batch * self.groups, channels // self.groups, -1)
        weights = (
            torch.matmul(weights_11, weights_12)
            + torch.matmul(weights_21, weights_22)
        ).reshape(batch * self.groups, 1, height, width)
        return (group_values * weights.sigmoid()).reshape(batch, channels, height, width)


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
    return nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)


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


def _build_pair_classifier(input_dim: int, hidden_dim: int | None, output_dim: int):
    if hidden_dim is None or hidden_dim <= 0:
        return nn.Linear(input_dim, output_dim)
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


def _relation_attention_config(value: dict | None) -> dict:
    if not value:
        return {"enabled": False}
    config = dict(value)
    if not bool(config.get("enabled", False)):
        return {"enabled": False}
    link_cnn = dict(config.get("link_cnn", {}))
    pair_features = config.get("pair_features", ("probability", "cosine"))
    if isinstance(pair_features, str):
        pair_features = (pair_features,)
    else:
        pair_features = tuple(pair_features)
    return {
        "enabled": True,
        "edge_dim": int(config.get("edge_dim", 32)),
        "kernel_size": int(link_cnn.get("kernel_size", config.get("kernel_size", 5))),
        "dilations": tuple(
            int(dilation)
            for dilation in link_cnn.get(
                "dilations",
                config.get("dilations", (1, 2, 4, 8, 16, 32, 64)),
            )
        ),
        "dropout": float(link_cnn.get("dropout", config.get("dropout", 0.2))),
        "ema_factor": int(link_cnn.get("ema_factor", config.get("ema_factor", 4))),
        "gate_init": float(config.get("gate_init", -4.0)),
        "relative_distance": bool(config.get("relative_distance", True)),
        "max_distance": int(config.get("max_distance", 2048)),
        "pair_features": _normalize_relation_pair_features(pair_features),
    }


def _boundary_head_config(value: dict | None) -> dict:
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


def _resolve_position_type(use_audio_positions: bool, audio_position_type: str | None) -> str:
    if audio_position_type is None:
        return "sinusoidal" if use_audio_positions else "none"
    position_type = audio_position_type.lower()
    if position_type not in {"none", "sinusoidal", "rope"}:
        raise ValueError("audio_position_type must be one of: none, sinusoidal, rope")
    return position_type
