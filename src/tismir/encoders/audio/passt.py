from __future__ import annotations

from pathlib import Path

import numpy as np

from tismir.encoders.audio import audio_encoders
from tismir.encoders.base import EmbeddingSequence


class PasstAudioEncoder:
    """PaSST audio encoder using the ``hear21passt`` package.

    Uses ``get_timestamp_embeddings`` from the HEAR API, which returns
    framewise embeddings (1295-d HEAR timestamp embedding) together with
    per-frame timestamps at ~20 Hz, so PaSST maps directly onto the
    dense-embedding contract. The ``variant`` selects the maximum supported
    input length (10/20/30 seconds).
    """

    name = "passt"

    def __init__(
        self,
        variant: str = "base",
        device: str = "cpu",
        embedding_dim: int = 1295,
        max_seconds: float | None = None,
        **_: object,
    ) -> None:
        try:
            if variant == "base30sec":
                from hear21passt.base30sec import load_model
            elif variant == "base20sec":
                from hear21passt.base20sec import load_model
            else:
                from hear21passt.base import load_model
            from hear21passt.base import get_timestamp_embeddings
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "PaSST requires hear21passt. Install with "
                "`python -m pip install -e '.[passt]'`."
            ) from exc

        self.variant = variant
        self.device = device
        self.output_dim = embedding_dim
        self.max_seconds = max_seconds
        self.sampling_rate = 32000  # PaSST is trained at 32 kHz
        self._torch = torch
        self._get_timestamp_embeddings = get_timestamp_embeddings
        self._model = load_model().to(device)
        self._model.eval()

    def encode(self, audio_path: str | Path) -> EmbeddingSequence:
        waveform, sample_rate = _load_audio_mono(audio_path, target_sr=self.sampling_rate)
        if self.max_seconds is not None:
            waveform = waveform[: int(self.max_seconds * sample_rate)]
        duration = len(waveform) / float(sample_rate)

        audio = self._torch.from_numpy(waveform).reshape(1, -1).to(self.device)
        with self._torch.inference_mode():
            embeddings, timestamps = self._get_timestamp_embeddings(audio, self._model)

        embeddings = embeddings.squeeze(0).detach().cpu().numpy().astype(np.float32)
        # hear21passt returns timestamps in milliseconds.
        times = (timestamps.squeeze(0).detach().cpu().numpy() / 1000.0).astype(np.float32)
        self.output_dim = int(embeddings.shape[1])

        return EmbeddingSequence(
            embeddings=embeddings,
            times=times,
            metadata={
                "encoder": self.name,
                "variant": self.variant,
                "device": self.device,
                "sampling_rate": sample_rate,
                "duration": duration,
                "output_dim": self.output_dim,
                "num_frames": int(len(embeddings)),
                "time_axis_note": "timestamps reported by hear21passt (converted ms -> s)",
            },
        )


def _load_audio_mono(audio_path: str | Path, target_sr: int) -> tuple[np.ndarray, int]:
    try:
        import librosa
    except ImportError:
        librosa = None

    if librosa is not None:
        waveform, sample_rate = librosa.load(audio_path, sr=target_sr, mono=True)
        return waveform.astype(np.float32), int(sample_rate)

    try:
        import soundfile as sf
        import soxr
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install librosa, or soundfile and soxr, to load audio for PaSST.") from exc

    waveform, sample_rate = sf.read(str(audio_path), always_2d=True)
    waveform = waveform.mean(axis=1).astype(np.float32)
    if sample_rate != target_sr:
        waveform = soxr.resample(waveform, sample_rate, target_sr).astype(np.float32)
        sample_rate = target_sr
    return waveform, int(sample_rate)


audio_encoders.register("passt", PasstAudioEncoder)
