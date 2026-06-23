from __future__ import annotations

from pathlib import Path

import numpy as np

from tismir.encoders.audio import audio_encoders
from tismir.encoders.base import EmbeddingSequence


class EncodecAudioEncoder:
    """EnCodec audio encoder using Hugging Face transformers.

    Uses the continuous latent sequence produced by the EnCodec encoder
    (the representation *before* residual vector quantization) as dense
    frame embeddings. EnCodec is a neural codec rather than a semantic
    foundation model, so these features are acoustic/reconstructive in
    nature; they are included for comparison with foundation-model encoders.
    """

    name = "encodec"

    def __init__(
        self,
        checkpoint: str = "facebook/encodec_24khz",
        device: str = "cpu",
        max_seconds: float | None = None,
        **_: object,
    ) -> None:
        try:
            import torch
            from transformers import AutoProcessor, EncodecModel
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "EnCodec requires torch and transformers. Install with "
                "`python -m pip install -e '.[encodec]'`."
            ) from exc

        self.checkpoint = checkpoint
        self.device = device
        self.max_seconds = max_seconds
        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(checkpoint)
        self._model = EncodecModel.from_pretrained(checkpoint).to(device)
        self._model.eval()
        self.sampling_rate = int(self._processor.sampling_rate)

        # Probe the encoder once to learn the latent dimension without
        # relying on a specific config attribute name.
        channels = int(getattr(self._model.config, "audio_channels", 1))
        probe = self._torch.zeros(1, channels, self.sampling_rate)
        with self._torch.inference_mode():
            latents = self._model.encoder(probe.to(device))
        self.output_dim = int(latents.shape[1])

    def encode(self, audio_path: str | Path) -> EmbeddingSequence:
        waveform, sample_rate = _load_audio_mono(audio_path, target_sr=self.sampling_rate)
        duration = len(waveform) / float(sample_rate)
        if self.max_seconds is not None:
            waveform = waveform[: int(self.max_seconds * sample_rate)]
            duration = len(waveform) / float(sample_rate)

        inputs = self._processor(
            raw_audio=waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        input_values = inputs["input_values"].to(self.device)

        with self._torch.inference_mode():
            latents = self._model.encoder(input_values)

        # latents: [batch, dim, num_frames] -> [num_frames, dim]
        embeddings = latents.squeeze(0).transpose(0, 1).detach().cpu().numpy().astype(np.float32)
        times = _uniform_times(num_frames=len(embeddings), duration=duration)

        return EmbeddingSequence(
            embeddings=embeddings,
            times=times,
            metadata={
                "encoder": self.name,
                "checkpoint": self.checkpoint,
                "device": self.device,
                "sampling_rate": sample_rate,
                "duration": duration,
                "output_dim": self.output_dim,
                "num_frames": int(len(embeddings)),
                "representation": "encoder_latent_pre_quantization",
                "time_axis_note": "uniformly spaced over loaded audio duration",
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
        raise ImportError("Install librosa, or soundfile and soxr, to load audio for EnCodec.") from exc

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


audio_encoders.register("encodec", EncodecAudioEncoder)
