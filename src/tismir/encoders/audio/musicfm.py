from __future__ import annotations

from pathlib import Path

import numpy as np

from tismir.encoders.audio import audio_encoders
from tismir.encoders.base import EmbeddingSequence


class MusicFmAudioEncoder:
    """MusicFM audio encoder (Won et al., 2024).

    Wraps the reference implementation at https://github.com/minzwon/musicfm,
    which is not published on PyPI. To use this backend:

    1. Clone the repo and make it importable (e.g. add it to ``PYTHONPATH``
       or ``pip install -e .`` from the clone) so ``musicfm`` imports.
    2. Download the statistics JSON and pretrained checkpoint (MSD or FMA)
       and pass their paths via ``stat_path`` / ``model_path``.

    MusicFM consumes 24 kHz mono audio and produces a 25 Hz latent sequence;
    layer 7 is the authors' recommended representation for frozen features.
    """

    name = "musicfm"

    def __init__(
        self,
        stat_path: str | None = None,
        model_path: str | None = None,
        layer: int = 7,
        device: str = "cpu",
        is_flash: bool = False,
        max_seconds: float | None = None,
        **_: object,
    ) -> None:
        if stat_path is None or model_path is None:
            raise ValueError(
                "MusicFM requires `stat_path` and `model_path` pointing to the "
                "downloaded statistics JSON and pretrained checkpoint."
            )
        try:
            import torch
            from musicfm.model.musicfm_25hz import MusicFM25Hz
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "MusicFM is not importable. Clone https://github.com/minzwon/musicfm "
                "and put it on PYTHONPATH, then install the [musicfm] extra "
                "(`python -m pip install -e '.[musicfm]'`)."
            ) from exc

        self.stat_path = stat_path
        self.model_path = model_path
        self.layer = layer
        self.device = device
        self.is_flash = is_flash
        self.max_seconds = max_seconds
        self.output_dim = 1024  # MusicFM-25Hz hidden size; refined after first encode
        self.sampling_rate = 24000
        self._torch = torch
        self._model = MusicFM25Hz(
            is_flash=is_flash,
            stat_path=stat_path,
            model_path=model_path,
        ).to(device)
        self._model.eval()

    def encode(self, audio_path: str | Path) -> EmbeddingSequence:
        waveform, sample_rate = _load_audio_mono(audio_path, target_sr=self.sampling_rate)
        if self.max_seconds is not None:
            waveform = waveform[: int(self.max_seconds * sample_rate)]
        duration = len(waveform) / float(sample_rate)

        wav = self._torch.from_numpy(waveform).reshape(1, -1).to(self.device)
        with self._torch.inference_mode():
            latent = self._model.get_latent(wav, layer_ix=self.layer)

        # latent: [batch, num_frames, dim] -> [num_frames, dim]
        embeddings = latent.squeeze(0).detach().cpu().numpy().astype(np.float32)
        self.output_dim = int(embeddings.shape[1])
        times = _uniform_times(num_frames=len(embeddings), duration=duration)

        return EmbeddingSequence(
            embeddings=embeddings,
            times=times,
            metadata={
                "encoder": self.name,
                "model_path": self.model_path,
                "stat_path": self.stat_path,
                "layer": self.layer,
                "device": self.device,
                "sampling_rate": sample_rate,
                "duration": duration,
                "output_dim": self.output_dim,
                "num_frames": int(len(embeddings)),
                "time_axis_note": "uniformly spaced over loaded audio duration (~25 Hz)",
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
        raise ImportError("Install librosa, or soundfile and soxr, to load audio for MusicFM.") from exc

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


audio_encoders.register("musicfm", MusicFmAudioEncoder)
