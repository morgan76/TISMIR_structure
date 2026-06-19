from __future__ import annotations


def frame_label_cross_entropy(logits, targets):
    """Cross-entropy over candidate text labels for each audio frame/beat."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError("Install the 'torch' extra to use losses.") from exc

    return torch.nn.functional.cross_entropy(logits, targets)
