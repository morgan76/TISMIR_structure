from __future__ import annotations

import numpy as np

from tismir.encoders.text import text_encoders


class ClapTextEncoder:
    """Text encoder backed by Hugging Face CLAP models."""

    name = "clap"

    def __init__(
        self,
        checkpoint: str = "laion/larger_clap_music_and_speech",
        device: str | None = None,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        max_length: int | None = None,
        **_: object,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "CLAP text encoding requires torch and transformers."
            ) from exc

        self.checkpoint = checkpoint
        self.device = device or "cpu"
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.max_length = max_length
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self._model = AutoModel.from_pretrained(checkpoint).to(self.device)
        self._model.eval()
        self.output_dim = int(getattr(self._model.config, "projection_dim", 0) or 512)

    def encode(self, labels: list[str]) -> np.ndarray:
        if not labels:
            return np.zeros((0, self.output_dim), dtype=np.float32)

        embeddings = []
        for start in range(0, len(labels), self.batch_size):
            batch = labels[start : start + self.batch_size]
            tokenized = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tokenized = {key: value.to(self.device) for key, value in tokenized.items()}
            with self._torch.inference_mode():
                if not hasattr(self._model, "get_text_features"):
                    raise TypeError(
                        f"Model {self.checkpoint!r} does not expose get_text_features()."
                    )
                features = self._model.get_text_features(**tokenized)
                features = _tensor_from_text_features(features)
                if self.normalize_embeddings:
                    features = self._torch.nn.functional.normalize(features, dim=-1)
            embeddings.append(features.detach().cpu().numpy())

        values = np.concatenate(embeddings, axis=0).astype(np.float32, copy=False)
        self.output_dim = int(values.shape[1])
        return values


text_encoders.register("clap", ClapTextEncoder)


def _tensor_from_text_features(features):
    if hasattr(features, "text_embeds") and features.text_embeds is not None:
        return features.text_embeds
    if hasattr(features, "pooler_output") and features.pooler_output is not None:
        return features.pooler_output
    return features
