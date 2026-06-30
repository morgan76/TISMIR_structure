from pathlib import Path

import numpy as np

from tismir.encoders.beats import beat_trackers


def test_builtin_beat_trackers_are_registered():
    names = beat_trackers.names()

    assert "uniform" in names
    assert "beat_this" in names
    assert "madmom" in names
    assert "precomputed" in names


def test_precomputed_beat_tracker_loads_track_estimates(tmp_path):
    track_dir = tmp_path / "dataset" / "track"
    track_dir.mkdir(parents=True)
    np.save(track_dir / "beats.npy", np.asarray([0.0, 0.5, 1.0], dtype=np.float32))
    np.save(track_dir / "downbeats.npy", np.asarray([0.0], dtype=np.float32))

    tracker = beat_trackers.build("precomputed", root=tmp_path, dataset="dataset")
    result = tracker.track(Path("audio") / "track.wav")

    np.testing.assert_array_equal(result.beats, [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(result.downbeats, [0.0])
    assert result.metadata["beat_tracker"] == "precomputed"
