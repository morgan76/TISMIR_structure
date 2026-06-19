import numpy as np

from tismir.preprocessing.beat_sync import build_beat_intervals, mean_pool_to_intervals


def test_build_beat_intervals_uses_track_duration_for_final_boundary():
    beats = np.asarray([0.0, 1.0, 2.0])
    intervals = build_beat_intervals(beats, track_duration=2.75)

    assert intervals == [(0.0, 1.0), (1.0, 2.0), (2.0, 2.75)]


def test_mean_pool_to_intervals():
    embeddings = np.asarray(
        [
            [0.0, 0.0],
            [2.0, 2.0],
            [10.0, 10.0],
            [14.0, 14.0],
        ]
    )
    times = np.asarray([0.1, 0.9, 1.1, 1.9])
    intervals = [(0.0, 1.0), (1.0, 2.0)]

    pooled = mean_pool_to_intervals(embeddings, times, intervals)

    np.testing.assert_allclose(pooled, [[1.0, 1.0], [12.0, 12.0]])
