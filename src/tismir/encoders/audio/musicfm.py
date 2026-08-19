from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

from tismir.encoders.audio import audio_encoders
from tismir.encoders.base import EmbeddingSequence


class MusicFmAudioEncoder:
    """MusicFM audio encoder.

    The reference implementation lives at https://github.com/minzwon/musicfm
    rather than on PyPI. To use this backend, clone/install MusicFM so the
    ``musicfm`` package is importable, then pass the downloaded statistics JSON
    and checkpoint through ``stat_path`` and ``model_path``.

    MusicFM consumes 24 kHz mono audio and returns a roughly 25 Hz latent
    sequence. Layer 7 is the representation recommended by the MusicFM authors
    for frozen downstream features.
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
            MusicFM25Hz = _import_musicfm_25hz()
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "MusicFM is not importable. Clone https://github.com/minzwon/musicfm "
                "and put either its parent directory or the clone itself on PYTHONPATH, "
                "then install optional runtime deps with "
                "`python -m pip install -e '.[musicfm]'`."
            ) from exc

        self.stat_path = stat_path
        self.model_path = model_path
        self.layer = layer
        self.device = device
        self.is_flash = is_flash
        self.max_seconds = max_seconds
        self.output_dim = 1024
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

        audio = self._torch.from_numpy(waveform).reshape(1, -1).to(self.device)
        with self._torch.inference_mode():
            latent = self._model.get_latent(audio, layer_ix=self.layer)

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


def _import_musicfm_25hz():
    try:
        from musicfm.model.musicfm_25hz import MusicFM25Hz

        return MusicFM25Hz
    except ImportError:
        _ensure_musicfm_parent_on_path()
        from musicfm.model.musicfm_25hz import MusicFM25Hz

        return MusicFM25Hz


def _ensure_musicfm_parent_on_path() -> None:
    """Support PYTHONPATH entries that point directly at the MusicFM clone."""

    for entry in list(sys.path):
        if not entry:
            continue
        path = Path(entry).expanduser()
        if path.name != "musicfm":
            continue
        if (path / "model" / "musicfm_25hz.py").exists():
            parent = str(path.parent)
            if parent not in sys.path:
                sys.path.insert(0, parent)
            return


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
