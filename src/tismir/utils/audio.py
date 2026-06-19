from __future__ import annotations

import contextlib
import wave
from pathlib import Path


def get_wav_duration(audio_path: str | Path) -> float:
    """Return WAV duration in seconds using the standard library."""

    audio_path = Path(audio_path)
    with contextlib.closing(wave.open(str(audio_path), "rb")) as handle:
        sample_rate = handle.getframerate()
        num_frames = handle.getnframes()
    if sample_rate <= 0:
        raise ValueError(f"Invalid sample rate in {audio_path}: {sample_rate}")
    return num_frames / float(sample_rate)
