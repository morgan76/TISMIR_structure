from __future__ import annotations


def audio_to_text_infonce(logits, targets, ignore_index: int = -100):
    """InfoNCE from each audio frame to candidate text labels."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError("Install the 'torch' extra to use losses.") from exc

    num_labels = logits.shape[-1]
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, num_labels),
        targets.reshape(-1),
        ignore_index=ignore_index,
    )


def text_to_audio_infonce(audio_tokens, text_tokens, targets, temperature: float = 0.07, ignore_index: int = -100):
    """Multi-positive InfoNCE from each label token to its matching audio frames."""

    torch = _require_torch()
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    text_tokens = _batched_text_tokens(text_tokens, batch_size=audio_tokens.shape[0])
    audio_tokens = torch.nn.functional.normalize(audio_tokens, dim=-1)
    text_tokens = torch.nn.functional.normalize(text_tokens, dim=-1)

    losses = []
    for batch_index in range(audio_tokens.shape[0]):
        valid = targets[batch_index] != ignore_index
        if not bool(valid.any()):
            continue
        audio = audio_tokens[batch_index, valid]
        labels = targets[batch_index, valid]
        text = text_tokens[batch_index]
        scores = text @ audio.transpose(0, 1) / temperature
        present_labels = labels.unique()
        for label in present_labels:
            label_index = int(label.detach().cpu())
            if label_index < 0 or label_index >= text.shape[0]:
                continue
            positive = labels == label
            if not bool(positive.any()):
                continue
            numerator = torch.logsumexp(scores[label_index, positive], dim=0)
            denominator = torch.logsumexp(scores[label_index], dim=0)
            losses.append(-(numerator - denominator))
    return _mean_or_zero(losses, reference=audio_tokens)


def audio_audio_supervised_contrastive(
    audio_tokens,
    targets,
    temperature: float = 0.07,
    ignore_index: int = -100,
):
    """Supervised contrastive loss over audio frames with same-label positives."""

    torch = _require_torch()
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    audio_tokens = torch.nn.functional.normalize(audio_tokens, dim=-1)

    losses = []
    for batch_index in range(audio_tokens.shape[0]):
        valid = targets[batch_index] != ignore_index
        if int(valid.sum().detach().cpu()) < 2:
            continue
        audio = audio_tokens[batch_index, valid]
        labels = targets[batch_index, valid]
        scores = audio @ audio.transpose(0, 1) / temperature
        self_mask = torch.eye(len(audio), dtype=torch.bool, device=audio.device)
        scores = scores.masked_fill(self_mask, torch.finfo(scores.dtype).min)
        same_label = labels[:, None] == labels[None, :]
        positive_mask = same_label & ~self_mask
        valid_anchors = positive_mask.any(dim=1)
        if not bool(valid_anchors.any()):
            continue
        numerator = torch.logsumexp(scores.masked_fill(~positive_mask, torch.finfo(scores.dtype).min), dim=1)
        denominator = torch.logsumexp(scores, dim=1)
        losses.append(-(numerator[valid_anchors] - denominator[valid_anchors]).mean())
    return _mean_or_zero(losses, reference=audio_tokens)


def _batched_text_tokens(text_tokens, batch_size: int):
    if text_tokens.ndim == 2:
        return text_tokens.unsqueeze(0).expand(batch_size, -1, -1)
    if text_tokens.ndim == 3:
        return text_tokens
    raise ValueError("text_tokens must have shape [K, D] or [B, K, D]")


def _mean_or_zero(losses, reference):
    if not losses:
        return reference.sum() * 0.0
    torch = _require_torch()
    return torch.stack(losses).mean()


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install the 'torch' extra to use losses.") from exc
    return torch
