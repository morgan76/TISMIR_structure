from __future__ import annotations

from pathlib import Path

import numpy as np

from tismir.encoders.audio import audio_encoders
from tismir.encoders.base import EmbeddingSequence


class PannsAudioEncoder:
    """PANNs (Cnn14) audio encoder using the ``panns_inference`` package.

    PANNs produce a single clip-level embedding (2048-d for Cnn14) rather than
    a frame sequence. To fit the dense-embedding contract used by the rest of
    the pipeline, the audio is split into fixed-length windows and each window
    is encoded independently, yielding one embedding per window. Use
    ``window_seconds`` / ``hop_seconds`` to control the resulting frame rate.
    """

    name = "panns"

    def __init__(
        self,
        checkpoint_path: str | None = None,
        device: str = "cpu",
        window_seconds: float = 1.0,
        hop_seconds: float = 1.0,
        embedding_dim: int = 2048,
        max_seconds: float | None = None,
        **_: object,
    ) -> None:
        try:
            from panns_inference import AudioTagging
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "PANNs requires panns-inference. Install with "
                "`python -m pip install -e '.[panns]'`."
            ) from exc

        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if hop_seconds <= 0:
            raise ValueError("hop_seconds must be positive")

        self.checkpoint_path = checkpoint_path
        self.device = device
        self.window_seconds = window_seconds
        self.hop_seconds = hop_seconds
        self.output_dim = embedding_dim
        self.max_seconds = max_seconds
        self.sampling_rate = 32000  # PANNs are trained at 32 kHz
        self._tagger = AudioTagging(checkpoint_path=checkpoint_path, device=device)

    def encode(self, audio_path: str | Path) -> EmbeddingSequence:
        waveform, sample_rate = _load_audio_mono(audio_path, target_sr=self.sampling_rate)
        if self.max_seconds is not None:
            waveform = waveform[: int(self.max_seconds * sample_rate)]
        duration = len(waveform) / float(sample_rate)

        window = max(int(self.window_seconds * sample_rate), 1)
        hop = max(int(self.hop_seconds * sample_rate), 1)
        starts = list(range(0, max(len(waveform) - window + 1, 1), hop))

        embeddings = []
        for start in starts:
            chunk = waveform[start : start + window]
            if len(chunk) < window:
                chunk = np.pad(chunk, (0, window - len(chunk)))
            _, embedding = self._tagger.inference(chunk[None, :])
            embeddings.append(np.asarray(embedding).reshape(-1))

        embeddings = np.stack(embeddings, axis=0).astype(np.float32)
        self.output_dim = int(embeddings.shape[1])
        times = np.asarray([start / float(sample_rate) for start in starts], dtype=np.float32)

        return EmbeddingSequence(
            embeddings=embeddings,
            times=times,
            metadata={
                "encoder": self.name,
                "checkpoint_path": self.checkpoint_path,
                "device": self.device,
                "sampling_rate": sample_rate,
                "duration": duration,
                "output_dim": self.output_dim,
                "num_frames": int(len(embeddings)),
                "window_seconds": self.window_seconds,
                "hop_seconds": self.hop_seconds,
                "time_axis_note": "window start times; clip-level embedding per window",
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
        raise ImportError("Install librosa, or soundfile and soxr, to load audio for PANNs.") from exc

    waveform, sample_rate = sf.read(str(audio_path), always_2d=True)
    waveform = waveform.mean(axis=1).astype(np.float32)
    if sample_rate != target_sr:
        waveform = soxr.resample(waveform, sample_rate, target_sr).astype(np.float32)
        sample_rate = target_sr
    return waveform, int(sample_rate)


audio_encoders.register("panns", PannsAudioEncoder)
