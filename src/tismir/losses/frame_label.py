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
