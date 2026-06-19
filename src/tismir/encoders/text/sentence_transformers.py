from __future__ import annotations

import numpy as np

from tismir.encoders.text import text_encoders


class SentenceTransformersTextEncoder:
    """Text encoder backed by the sentence-transformers package."""

    name = "sentence_transformers"

    def __init__(
        self,
        checkpoint: str = "intfloat/e5-base-v2",
        device: str | None = None,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        **_: object,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "sentence-transformers is not installed. Install with "
                "`python -m pip install -e '.[text]'`."
            ) from exc

        self.checkpoint = checkpoint
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self._model = SentenceTransformer(checkpoint, device=device)
        if hasattr(self._model, "get_embedding_dimension"):
            self.output_dim = int(self._model.get_embedding_dimension())
        else:
            self.output_dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, labels: list[str]) -> np.ndarray:
        if not labels:
            return np.zeros((0, self.output_dim), dtype=np.float32)
        embeddings = self._model.encode(
            labels,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)


text_encoders.register("sentence_transformers", SentenceTransformersTextEncoder)
