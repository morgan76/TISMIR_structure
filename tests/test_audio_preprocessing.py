import json
import wave
from pathlib import Path

import numpy as np

from tismir.data.schemas import Track
from tismir.encoders.beats.base import BeatTrackingResult
from tismir.preprocessing.audio import _trim_beats_to_duration, preprocess_track_audio


def test_preprocess_track_audio_writes_expected_arrays(tmp_path):
    audio_path = tmp_path / "audio.wav"
    _write_silent_wav(audio_path, duration=2.0, sample_rate=8000)
    jams_path = tmp_path / "audio.jams"
    jams_path.write_text("{}", encoding="utf-8")
    track = Track(
        track_id="track",
        audio_path=audio_path,
        jams_path=jams_path,
        dataset="dataset",
        split="train",
    )

    result = preprocess_track_audio(
        track=track,
        output_root=tmp_path / "embeddings",
        audio_encoder_name="placeholder",
        audio_encoder_params={"output_dim": 4, "frame_rate": 4.0},
        beat_tracker_name="uniform",
        beat_tracker_params={"beat_period": 0.5, "estimate_downbeats": True},
        pooling={"method": "mean", "keep_dense": True},
    )

    output_dir = Path(result.output_dir)
    assert result.dense_shape == (8, 4)
    assert result.beat_sync_shape == (4, 4)
    assert np.load(output_dir / "beat_sync.npy").shape == (4, 4)
    assert np.load(output_dir / "beats.npy").tolist() == [0.0, 0.5, 1.0, 1.5]

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["audio_encoder"]["encoder"] == "placeholder"
    assert metadata["beat_tracker"]["beat_tracker"] == "uniform"
    assert metadata["pooling"]["method"] == "mean"


def test_preprocess_track_audio_trims_beats_to_embedding_duration(tmp_path):
    trimmed = _trim_beats_to_duration(
        BeatTrackingResult(
            beats=np.asarray([0.0, 0.5, 1.0, 1.5, 2.0]),
            downbeats=np.asarray([0.0, 2.0]),
        ),
        duration=1.6,
    )

    np.testing.assert_allclose(trimmed.beats, [0.0, 0.5, 1.0, 1.5])
    np.testing.assert_allclose(trimmed.downbeats, [0.0])
    assert trimmed.metadata["original_num_beats"] == 5
    assert trimmed.metadata["num_beats_after_duration_trim"] == 4


def _write_silent_wav(path: Path, duration: float, sample_rate: int) -> None:
    num_frames = int(duration * sample_rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * num_frames)
