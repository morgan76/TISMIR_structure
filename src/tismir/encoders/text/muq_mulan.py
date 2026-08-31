from __future__ import annotations

import numpy as np

from tismir.encoders.text import text_encoders


class MuqMulanTextEncoder:
    """Text encoder backed by OpenMuQ MuQ-MuLan."""

    name = "muq_mulan"

    def __init__(
        self,
        checkpoint: str = "OpenMuQ/MuQ-MuLan-large",
        device: str | None = None,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        **_: object,
    ) -> None:
        try:
            import torch
            from muq import MuQMuLan
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "MuQ-MuLan text encoding requires the `muq` package. "
                "Install it with `python -m pip install muq`."
            ) from exc

        self.checkpoint = checkpoint
        self.device = device or "cpu"
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self._torch = torch
        self._model = MuQMuLan.from_pretrained(checkpoint).to(self.device)
        self._model.eval()
        self.output_dim = int(getattr(self._model, "embed_dim", 0) or 512)

    def encode(self, labels: list[str]) -> np.ndarray:
        if not labels:
            return np.zeros((0, self.output_dim), dtype=np.float32)

        embeddings = []
        for start in range(0, len(labels), self.batch_size):
            batch = labels[start : start + self.batch_size]
            with self._torch.inference_mode():
                features = self._model(texts=batch)
                features = _tensor_from_output(features)
                if self.normalize_embeddings:
                    features = self._torch.nn.functional.normalize(features, dim=-1)
            embeddings.append(features.detach().cpu().numpy())

        values = np.concatenate(embeddings, axis=0).astype(np.float32, copy=False)
        self.output_dim = int(values.shape[1])
        return values


def _tensor_from_output(output):
    if isinstance(output, dict):
        for key in ("text_embeds", "embeds", "last_hidden_state"):
            if key in output:
                return output[key]
    if hasattr(output, "text_embeds") and output.text_embeds is not None:
        return output.text_embeds
    return output


text_encoders.register("muq_mulan", MuqMulanTextEncoder)
