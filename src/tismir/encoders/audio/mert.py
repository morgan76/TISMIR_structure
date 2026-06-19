from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from tismir.encoders.audio import audio_encoders
from tismir.encoders.base import EmbeddingSequence


class MertAudioEncoder:
    """MERT audio encoder using Hugging Face transformers."""

    name = "mert"

    def __init__(
        self,
        checkpoint: str = "m-a-p/MERT-v1-95M",
        layer: int | Literal["mean"] = -1,
        device: str = "cpu",
        max_seconds: float | None = None,
        trust_remote_code: bool = True,
        **_: object,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, Wav2Vec2FeatureExtractor
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "MERT requires torch and transformers. Install with "
                "`python -m pip install -e '.[hf-audio]'`."
            ) from exc

        self.checkpoint = checkpoint
        self.layer = layer
        self.device = device
        self.max_seconds = max_seconds
        self.trust_remote_code = trust_remote_code
        self._torch = torch
        self._feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            checkpoint,
            trust_remote_code=trust_remote_code,
        )
        self._model = AutoModel.from_pretrained(
            checkpoint,
            trust_remote_code=trust_remote_code,
        ).to(device)
        self._model.eval()
        self.output_dim = int(self._model.config.hidden_size)
        self.sampling_rate = int(self._feature_extractor.sampling_rate)

    def encode(self, audio_path: str | Path) -> EmbeddingSequence:
        waveform, sample_rate = _load_audio_mono(audio_path, target_sr=self.sampling_rate)
        duration = len(waveform) / float(sample_rate)
        if self.max_seconds is not None:
            max_samples = int(self.max_seconds * sample_rate)
            waveform = waveform[:max_samples]
            duration = len(waveform) / float(sample_rate)

        inputs = self._feature_extractor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with self._torch.inference_mode():
            outputs = self._model(**inputs, output_hidden_states=True)

        hidden = self._select_hidden_state(outputs.hidden_states)
        embeddings = hidden.squeeze(0).detach().cpu().numpy().astype(np.float32)
        times = _uniform_times(num_frames=len(embeddings), duration=duration)

        return EmbeddingSequence(
            embeddings=embeddings,
            times=times,
            metadata={
                "encoder": self.name,
                "checkpoint": self.checkpoint,
                "layer": self.layer,
                "device": self.device,
                "sampling_rate": sample_rate,
                "duration": duration,
                "output_dim": self.output_dim,
                "num_frames": int(len(embeddings)),
                "time_axis_note": "uniformly spaced over loaded audio duration",
            },
        )

    def _select_hidden_state(self, hidden_states):
        if self.layer == "mean":
            return self._torch.stack(hidden_states, dim=0).mean(dim=0)
        return hidden_states[int(self.layer)]


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
        raise ImportError("Install librosa, or soundfile and soxr, to load audio for MERT.") from exc

    waveform, sample_rate = sf.read(str(audio_path), always_2d=True)
    waveform = waveform.mean(axis=1).astype(np.float32)
    if sample_rate != target_sr:
        waveform = soxr.resample(waveform, sample_rate, target_sr).astype(np.float32)
        sample_rate = target_sr
    return waveform, int(sample_rate)


def _uniform_times(num_frames: int, duration: float) -> np.ndarray:
    if num_frames <= 0:
        return np.asarray([], dtype=np.float32)
    if num_frames == 1:
        return np.asarray([0.0], dtype=np.float32)
    return np.linspace(0.0, duration, num=num_frames, endpoint=False, dtype=np.float32)


audio_encoders.register("mert", MertAudioEncoder)
