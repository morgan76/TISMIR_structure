from __future__ import annotations


def frame_label_cross_entropy(logits, targets, ignore_index: int = -100):
    """Cross-entropy over candidate text labels for each audio frame/beat."""

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


def pairwise_probability_loss(
    logits,
    targets,
    ignore_index: int = -100,
    balance: bool = True,
    eps: float = 1e-6,
):
    """BCE loss on whether frame-label probabilities imply same-label pairs."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError("Install the 'torch' extra to use losses.") from exc

    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, time, labels]")
    if targets.ndim != 2:
        raise ValueError("targets must have shape [batch, time]")
    if not 0.0 < eps < 0.5:
        raise ValueError("eps must be between 0 and 0.5")

    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    losses = []
    for batch_index in range(logits.shape[0]):
        valid = targets[batch_index] != ignore_index
        if int(valid.sum().detach().cpu()) < 2:
            continue
        probs = probabilities[batch_index, valid]
        labels = targets[batch_index, valid]
        same = labels[:, None] == labels[None, :]
        not_self = ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
        same_probabilities = probs @ probs.transpose(0, 1)
        same_probabilities = same_probabilities.clamp(min=eps, max=1.0 - eps)
        target_values = same.to(dtype=same_probabilities.dtype)
        weights = None
        if balance:
            weights = _balanced_pair_weights(target_values, not_self)
        losses.append(
            torch.nn.functional.binary_cross_entropy(
                same_probabilities[not_self],
                target_values[not_self],
                weight=None if weights is None else weights[not_self],
            )
        )

    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def cross_similarity_matching_loss(
    similarity,
    targets,
    ignore_index: int = -100,
    positive_target: float = 1.0,
    negative_target: float = 0.0,
    balance: bool = True,
):
    """MSE between audio-text similarities and soft frame-label targets."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError("Install the 'torch' extra to use losses.") from exc

    if similarity.ndim != 3:
        raise ValueError("similarity must have shape [batch, time, labels]")
    if targets.ndim != 2:
        raise ValueError("targets must have shape [batch, time]")

    losses = []
    for batch_index in range(similarity.shape[0]):
        valid = targets[batch_index] != ignore_index
        if not bool(valid.any()):
            continue
        scores = similarity[batch_index, valid]
        labels = targets[batch_index, valid].long()
        target_scores = scores.new_full(scores.shape, float(negative_target))
        target_scores[torch.arange(len(labels), device=labels.device), labels] = float(positive_target)
        squared_error = (scores - target_scores).pow(2)
        if balance:
            weights = scores.new_full(scores.shape, 1.0 / max(scores.shape[1] - 1, 1))
            weights[torch.arange(len(labels), device=labels.device), labels] = 1.0
            squared_error = squared_error * weights
        losses.append(squared_error.mean())

    if not losses:
        return similarity.sum() * 0.0
    return torch.stack(losses).mean()


def token_uniformity_loss(tokens, mask=None, alpha: float = 2.0):
    """Wang-Isola uniformity loss over normalized tokens."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError("Install the 'torch' extra to use losses.") from exc

    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if tokens.ndim == 2:
        tokens = tokens.unsqueeze(0)
    if tokens.ndim != 3:
        raise ValueError("tokens must have shape [items, dim] or [batch, items, dim]")
    if mask is not None and mask.shape != tokens.shape[:2]:
        raise ValueError("mask must have shape [batch, items]")

    losses = []
    normalized = torch.nn.functional.normalize(tokens, dim=-1)
    for batch_index in range(normalized.shape[0]):
        if mask is None:
            values = normalized[batch_index]
        else:
            values = normalized[batch_index, mask[batch_index].bool()]
        if int(values.shape[0]) < 2:
            continue
        distances = torch.pdist(values, p=2).pow(2)
        losses.append(torch.log(torch.exp(-float(alpha) * distances).mean().clamp_min(1e-12)))

    if not losses:
        return tokens.sum() * 0.0
    return torch.stack(losses).mean()


def _balanced_pair_weights(targets, mask):
    positive = targets == 1
    negative = targets == 0
    positive_count = (positive & mask).sum()
    negative_count = (negative & mask).sum()
    total = positive_count + negative_count
    if int(positive_count.detach().cpu()) == 0 or int(negative_count.detach().cpu()) == 0:
        return None
    weights = targets.new_zeros(targets.shape)
    weights[positive] = total / (2.0 * positive_count)
    weights[negative] = total / (2.0 * negative_count)
    return weights
