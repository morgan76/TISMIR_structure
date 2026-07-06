from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from tismir.encoders.audio import audio_encoders
from tismir.encoders.base import EmbeddingSequence


class AudioMaeAudioEncoder:
    """AudioMAE encoder ("Masked Autoencoders that Listen", Huang et al. 2022).

    Loads the timm-compatible AudioMAE ViT weights published by ``gaunernst``
    on the Hugging Face Hub and uses the encoder backbone as a frozen feature
    extractor. The official repo ships only research/training code, so this
    backend reproduces the AudioMAE input recipe (128-bin Kaldi fbank at
    16 kHz, 10 ms hop, the AudioSet normalization) and maps the ViT patch grid
    back to a time sequence.

    Each 16x16 patch covers 16 spectrogram frames (~0.16 s) x 16 mel bins.
    Tokens are reshaped to (time_patches, freq_patches, dim) and mean-pooled
    over the frequency axis, yielding one ~6.25 Hz frame embedding per time
    patch. Audio longer than ``target_length`` frames is processed in
    consecutive non-overlapping windows.

    NOTE: this is the least turnkey backend here -- it depends on the exact
    patch layout of the gaunernst checkpoints; verify shapes against the
    chosen ``model_name`` before relying on the features.
    """

    name = "audiomae"

    # AudioSet fbank normalization used by AudioMAE/AST.
    NORM_MEAN = -4.2677393
    NORM_STD = 4.5689974

    def __init__(
        self,
        model_name: str = "hf_hub:gaunernst/vit_base_patch16_1024_128.audiomae_as2m",
        device: str = "cpu",
        num_mel_bins: int = 128,
        target_length: int = 1024,
        patch_size: int = 16,
        max_seconds: float | None = None,
        **_: object,
    ) -> None:
        try:
            import timm
            import torch
            import torchaudio
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "AudioMAE requires torch, timm and torchaudio. Install with "
                "`python -m pip install -e '.[audiomae]'`."
            ) from exc

        if target_length % patch_size != 0 or num_mel_bins % patch_size != 0:
            raise ValueError("target_length and num_mel_bins must be divisible by patch_size")

        self.model_name = model_name
        self.device = device
        self.num_mel_bins = num_mel_bins
        self.target_length = target_length
        self.patch_size = patch_size
        self.max_seconds = max_seconds
        self.sampling_rate = 16000
        self._torch = torch
        self._torchaudio = torchaudio

        self._model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            in_chans=1,
            img_size=(target_length, num_mel_bins),
        ).to(device)
        self._model.eval()

        self._num_prefix = int(getattr(self._model, "num_prefix_tokens", 1))
        self._time_patches = target_length // patch_size
        self._freq_patches = num_mel_bins // patch_size
        self.output_dim = int(getattr(self._model, "num_features", 768))
        # Each time patch spans patch_size fbank frames at a 10 ms hop.
        self._frame_seconds = patch_size * 0.010

    def _fbank(self, waveform: np.ndarray):
        torch = self._torch
        signal = torch.from_numpy(waveform).reshape(1, -1)
        signal = signal - signal.mean()
        fbank = self._torchaudio.compliance.kaldi.fbank(
            signal,
            htk_compat=True,
            sample_frequency=self.sampling_rate,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=self.num_mel_bins,
            dither=0.0,
            frame_shift=10,
        )
        return (fbank - self.NORM_MEAN) / (self.NORM_STD * 2.0)

    def encode(self, audio_path: str | Path) -> EmbeddingSequence:
        torch = self._torch
        waveform, sample_rate = _load_audio_mono(audio_path, target_sr=self.sampling_rate)
        if self.max_seconds is not None:
            waveform = waveform[: int(self.max_seconds * sample_rate)]
        duration = len(waveform) / float(sample_rate)

        fbank = self._fbank(waveform)  # [num_frames, num_mel_bins]
        num_frames = int(fbank.shape[0])
        num_chunks = max(1, math.ceil(num_frames / self.target_length))

        chunk_embeddings = []
        with torch.inference_mode():
            for index in range(num_chunks):
                start = index * self.target_length
                segment = fbank[start : start + self.target_length]
                if segment.shape[0] < self.target_length:
                    pad = self.target_length - segment.shape[0]
                    segment = torch.nn.functional.pad(segment, (0, 0, 0, pad))
                x = segment.unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, T, F]
                features = self._model.forward_features(x)
                patches = features[:, self._num_prefix :, :]
                grid = patches.reshape(1, self._time_patches, self._freq_patches, -1)
                pooled = grid.mean(dim=2).squeeze(0)  # [time_patches, dim]
                chunk_embeddings.append(pooled.detach().cpu())

        embeddings = torch.cat(chunk_embeddings, dim=0).numpy().astype(np.float32)
        # Trim padding-only time patches beyond the real audio.
        valid = max(1, math.ceil(num_frames / self.patch_size))
        embeddings = embeddings[:valid]
        self.output_dim = int(embeddings.shape[1])
        times = (np.arange(len(embeddings), dtype=np.float32) * self._frame_seconds).astype(np.float32)

        return EmbeddingSequence(
            embeddings=embeddings,
            times=times,
            metadata={
                "encoder": self.name,
                "model_name": self.model_name,
                "device": self.device,
                "sampling_rate": sample_rate,
                "duration": duration,
                "output_dim": self.output_dim,
                "num_frames": int(len(embeddings)),
                "num_mel_bins": self.num_mel_bins,
                "target_length": self.target_length,
                "patch_size": self.patch_size,
                "time_axis_note": "one frame per ViT time patch (freq-pooled), ~6.25 Hz",
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
        raise ImportError("Install librosa, or soundfile and soxr, to load audio for AudioMAE.") from exc

    waveform, sample_rate = sf.read(str(audio_path), always_2d=True)
    waveform = waveform.mean(axis=1).astype(np.float32)
    if sample_rate != target_sr:
        waveform = soxr.resample(waveform, sample_rate, target_sr).astype(np.float32)
        sample_rate = target_sr
    return waveform, int(sample_rate)


audio_encoders.register("audiomae", AudioMaeAudioEncoder)
