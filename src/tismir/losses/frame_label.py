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


def pairwise_structure_loss(
    tokens,
    targets,
    ignore_index: int = -100,
    balance: bool = True,
    eps: float = 1e-6,
):
    """BCE loss on whether projected audio tokens imply same-label pairs."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError("Install the 'torch' extra to use losses.") from exc

    if tokens.ndim != 3:
        raise ValueError("tokens must have shape [batch, time, dim]")
    if targets.ndim != 2:
        raise ValueError("targets must have shape [batch, time]")
    if not 0.0 < eps < 0.5:
        raise ValueError("eps must be between 0 and 0.5")

    normalized = torch.nn.functional.normalize(tokens, dim=-1)
    losses = []
    for batch_index in range(tokens.shape[0]):
        valid = targets[batch_index] != ignore_index
        if int(valid.sum().detach().cpu()) < 2:
            continue
        values = normalized[batch_index, valid]
        labels = targets[batch_index, valid]
        same = labels[:, None] == labels[None, :]
        not_self = ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
        similarities = values @ values.transpose(0, 1)
        probabilities = ((similarities + 1.0) * 0.5).clamp(min=eps, max=1.0 - eps)
        target_values = same.to(dtype=probabilities.dtype)
        weights = None
        if balance:
            weights = _balanced_pair_weights(target_values, not_self)
        losses.append(
            torch.nn.functional.binary_cross_entropy(
                probabilities[not_self],
                target_values[not_self],
                weight=None if weights is None else weights[not_self],
            )
        )

    if not losses:
        return tokens.sum() * 0.0
    return torch.stack(losses).mean()


def pairwise_structure_relation_loss(
    pair_logits,
    base_targets,
    segment_targets,
    ignore_index: int = -100,
    balance: bool = True,
):
    """Cross-entropy over different-section/same-section/same-segment pairs."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError("Install the 'torch' extra to use losses.") from exc

    if pair_logits.ndim != 4 or pair_logits.shape[-1] != 3:
        raise ValueError("pair_logits must have shape [batch, time, time, 3]")
    if base_targets.ndim != 2 or segment_targets.ndim != 2:
        raise ValueError("base_targets and segment_targets must have shape [batch, time]")
    if pair_logits.shape[:3] != (
        base_targets.shape[0],
        base_targets.shape[1],
        base_targets.shape[1],
    ):
        raise ValueError("pair_logits and targets have incompatible shapes")

    losses = []
    for batch_index in range(pair_logits.shape[0]):
        valid = (
            (base_targets[batch_index] != ignore_index)
            & (segment_targets[batch_index] != ignore_index)
        )
        if int(valid.sum().detach().cpu()) < 2:
            continue
        logits = pair_logits[batch_index, valid][:, valid]
        bases = base_targets[batch_index, valid]
        segments = segment_targets[batch_index, valid]
        not_self = ~torch.eye(len(bases), dtype=torch.bool, device=bases.device)
        same_segment = segments[:, None] == segments[None, :]
        same_section = bases[:, None] == bases[None, :]
        relation_targets = torch.zeros((len(bases), len(bases)), dtype=torch.long, device=bases.device)
        relation_targets[same_section] = 1
        relation_targets[same_segment] = 2
        logits = logits[not_self]
        relation_targets = relation_targets[not_self]
        weights = None
        if balance:
            counts = torch.bincount(relation_targets, minlength=3).to(dtype=logits.dtype)
            if bool((counts > 0).all()):
                weights = counts.sum() / (3.0 * counts)
        losses.append(
            torch.nn.functional.cross_entropy(
                logits,
                relation_targets,
                weight=weights,
            )
        )

    if not losses:
        return pair_logits.sum() * 0.0
    return torch.stack(losses).mean()


def boundary_prediction_loss(
    boundary_logits,
    segment_targets,
    ignore_index: int = -100,
    eps: float = 1e-8,
):
    """LinkSeg-style Dice loss for boundaries between consecutive segments."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError("Install the 'torch' extra to use losses.") from exc

    if boundary_logits.ndim != 2:
        raise ValueError("boundary_logits must have shape [batch, time - 1]")
    if segment_targets.ndim != 2:
        raise ValueError("segment_targets must have shape [batch, time]")
    if boundary_logits.shape[0] != segment_targets.shape[0] or boundary_logits.shape[1] != max(
        segment_targets.shape[1] - 1,
        0,
    ):
        raise ValueError("boundary_logits and segment_targets have incompatible shapes")
    if eps <= 0:
        raise ValueError("eps must be positive")

    losses = []
    for batch_index in range(boundary_logits.shape[0]):
        left = segment_targets[batch_index, :-1]
        right = segment_targets[batch_index, 1:]
        valid = (left != ignore_index) & (right != ignore_index)
        if not bool(valid.any()):
            continue
        predictions = torch.sigmoid(boundary_logits[batch_index, valid])
        boundary_targets = (left[valid] != right[valid]).to(dtype=predictions.dtype)
        numerator = 2.0 * (predictions * boundary_targets).sum()
        denominator = predictions.pow(2).sum() + boundary_targets.pow(2).sum()
        losses.append(1.0 - numerator / denominator.clamp_min(eps))

    if not losses:
        return boundary_logits.sum() * 0.0
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
