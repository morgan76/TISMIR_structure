from __future__ import annotations

from pathlib import Path

import numpy as np

from tismir.encoders.audio import audio_encoders
from tismir.encoders.base import EmbeddingSequence


class SongFormerSslAudioEncoder:
    """SongFormer SSL input-stack encoder (ASLP-lab/SongFormer).

    SongFormer is a music-structure-analysis model, not a feature extractor:
    it predicts segment labels. This backend does *not* run the SongFormer
    transformer; instead it reproduces the multi-resolution SSL feature stack
    that SongFormer consumes as input, so the rest of this pipeline can use
    those features as dense frame embeddings.

    Following ``src/SongFormer/infer/infer.py``, four 25 Hz feature streams are
    concatenated along the feature axis (in this order)::

        [ MusicFM(30s windows), MuQ(30s windows),
          MusicFM(long window),  MuQ(long window) ]

    The "30s" streams chunk the audio into 30-second segments (the SSL models'
    native context) and concatenate over time; the "long" streams feed up to
    ``long_window_seconds`` at once. All four use hidden-state ``layer`` and
    are truncated to a common length before concatenation, giving a
    ``2 * (musicfm_dim + muq_dim)``-d embedding (4096 for the default 1024-d
    MusicFM-MSD + 1024-d MuQ-large).

    Requirements (mirrors the SongFormer setup):
      * MuQ from PyPI (``muq``), checkpoint ``OpenMuQ/MuQ-large-msd-iter``.
      * MusicFM from source (github.com/minzwon/musicfm) on PYTHONPATH, with
        downloaded ``musicfm_stat_path`` / ``musicfm_model_path``.
    """

    name = "songformer_ssl"

    def __init__(
        self,
        musicfm_stat_path: str | None = None,
        musicfm_model_path: str | None = None,
        muq_checkpoint: str = "OpenMuQ/MuQ-large-msd-iter",
        layer: int = 10,
        device: str = "cpu",
        chunk_seconds: float = 30.0,
        long_window_seconds: float = 420.0,
        is_flash: bool = False,
        max_seconds: float | None = None,
        **_: object,
    ) -> None:
        if musicfm_stat_path is None or musicfm_model_path is None:
            raise ValueError(
                "SongFormer SSL stack requires `musicfm_stat_path` and "
                "`musicfm_model_path` for the MusicFM component."
            )
        try:
            import torch
            from muq import MuQ
            from musicfm.model.musicfm_25hz import MusicFM25Hz
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "SongFormer SSL stack requires torch, muq (`pip install muq`), and "
                "MusicFM from source (github.com/minzwon/musicfm on PYTHONPATH). "
                "Install runtime deps with `python -m pip install -e '.[songformer]'`."
            ) from exc

        self.musicfm_stat_path = musicfm_stat_path
        self.musicfm_model_path = musicfm_model_path
        self.muq_checkpoint = muq_checkpoint
        self.layer = layer
        self.device = device
        self.chunk_seconds = chunk_seconds
        self.long_window_seconds = long_window_seconds
        self.max_seconds = max_seconds
        self.sampling_rate = 24000  # SongFormer INPUT_SAMPLING_RATE
        self.output_dim = 4096  # refined after first encode

        self._torch = torch
        self._musicfm = MusicFM25Hz(
            is_flash=is_flash,
            stat_path=musicfm_stat_path,
            model_path=musicfm_model_path,
        ).to(device)
        self._musicfm.eval()
        self._muq = MuQ.from_pretrained(muq_checkpoint).to(device)
        self._muq.eval()

    def _musicfm_features(self, segment):
        _, hidden_states = self._musicfm.get_predictions(segment.unsqueeze(0))
        return hidden_states[self.layer]  # [1, frames, dim]

    def _muq_features(self, segment):
        output = self._muq(segment.unsqueeze(0), output_hidden_states=True)
        return output["hidden_states"][self.layer]  # [1, frames, dim]

    def _chunked(self, audio, feature_fn, chunk_samples):
        chunk_samples = max(chunk_samples, 1)
        parts = []
        for start in range(0, audio.shape[0], chunk_samples):
            segment = audio[start : start + chunk_samples]
            if segment.shape[0] == 0:
                continue
            parts.append(feature_fn(segment))
        return self._torch.cat(parts, dim=1)  # concat over time

    def encode(self, audio_path: str | Path) -> EmbeddingSequence:
        torch = self._torch
        waveform, sample_rate = _load_audio_mono(audio_path, target_sr=self.sampling_rate)
        if self.max_seconds is not None:
            waveform = waveform[: int(self.max_seconds * sample_rate)]
        duration = len(waveform) / float(sample_rate)

        audio = torch.from_numpy(waveform).to(self.device)
        chunk_samples = int(self.chunk_seconds * sample_rate)
        long_samples = int(self.long_window_seconds * sample_rate)

        with torch.inference_mode():
            streams = [
                self._chunked(audio, self._musicfm_features, chunk_samples),
                self._chunked(audio, self._muq_features, chunk_samples),
                self._chunked(audio, self._musicfm_features, long_samples),
                self._chunked(audio, self._muq_features, long_samples),
            ]
            min_len = min(stream.shape[1] for stream in streams)
            streams = [stream[:, :min_len, :] for stream in streams]
            embd = torch.concatenate(streams, axis=-1)

        embeddings = embd.squeeze(0).detach().cpu().numpy().astype(np.float32)
        self.output_dim = int(embeddings.shape[1])
        times = _uniform_times(num_frames=len(embeddings), duration=duration)

        return EmbeddingSequence(
            embeddings=embeddings,
            times=times,
            metadata={
                "encoder": self.name,
                "muq_checkpoint": self.muq_checkpoint,
                "musicfm_model_path": self.musicfm_model_path,
                "layer": self.layer,
                "device": self.device,
                "sampling_rate": sample_rate,
                "duration": duration,
                "output_dim": self.output_dim,
                "num_frames": int(len(embeddings)),
                "stream_order": ["musicfm_30s", "muq_30s", "musicfm_long", "muq_long"],
                "chunk_seconds": self.chunk_seconds,
                "long_window_seconds": self.long_window_seconds,
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
        raise ImportError("Install librosa, or soundfile and soxr, to load audio for SongFormer.") from exc

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


audio_encoders.register("songformer_ssl", SongFormerSslAudioEncoder)
