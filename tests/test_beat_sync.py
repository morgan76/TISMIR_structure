import numpy as np
import pytest

from tismir.preprocessing.beat_sync import (
    MULTI_STAT_CHOICES,
    POOLING_METHODS,
    build_beat_intervals,
    mean_pool_to_intervals,
    pool_frames_to_intervals,
)


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


# --- shared fixtures for per-method checks -------------------------------------------------

_EMBEDDINGS = np.asarray(
    [
        [0.0, 0.0],
        [2.0, 2.0],
        [10.0, 10.0],
        [14.0, 14.0],
    ]
)
_TIMES = np.asarray([0.1, 0.9, 1.1, 1.9])
_INTERVALS = [(0.0, 1.0), (1.0, 2.0)]


def test_mean_matches_dispatcher_and_wrapper():
    dispatched = pool_frames_to_intervals(_EMBEDDINGS, _TIMES, _INTERVALS, method="mean")
    wrapped = mean_pool_to_intervals(_EMBEDDINGS, _TIMES, _INTERVALS)
    np.testing.assert_allclose(dispatched, wrapped)


def test_max_pool_takes_elementwise_maximum():
    pooled = pool_frames_to_intervals(_EMBEDDINGS, _TIMES, _INTERVALS, method="max")
    np.testing.assert_allclose(pooled, [[2.0, 2.0], [14.0, 14.0]])


def test_first_pool_takes_earliest_frame_in_interval():
    pooled = pool_frames_to_intervals(_EMBEDDINGS, _TIMES, _INTERVALS, method="first")
    np.testing.assert_allclose(pooled, [[0.0, 0.0], [10.0, 10.0]])


def test_last_pool_takes_latest_frame_in_interval():
    pooled = pool_frames_to_intervals(_EMBEDDINGS, _TIMES, _INTERVALS, method="last")
    np.testing.assert_allclose(pooled, [[2.0, 2.0], [14.0, 14.0]])


def test_energy_weighted_reduces_to_mean_when_norms_are_equal():
    # Both frames have L2 norm 5, so softmax weights are equal -> plain mean.
    embeddings = np.asarray([[3.0, 4.0], [4.0, 3.0]])
    times = np.asarray([0.1, 0.9])
    pooled = pool_frames_to_intervals(embeddings, times, [(0.0, 1.0)], method="energy_weighted")
    np.testing.assert_allclose(pooled, [[3.5, 3.5]])


def test_energy_weighted_favours_high_norm_frames():
    pooled = pool_frames_to_intervals(_EMBEDDINGS, _TIMES, _INTERVALS, method="energy_weighted")
    mean = pool_frames_to_intervals(_EMBEDDINGS, _TIMES, _INTERVALS, method="mean")
    # The higher-norm frame in each interval pulls the pooled vector above the mean.
    assert np.all(pooled >= mean - 1e-6)
    assert pooled[0, 0] > mean[0, 0]


def test_energy_weighted_temperature_flattens_towards_mean():
    hot = pool_frames_to_intervals(
        _EMBEDDINGS, _TIMES, _INTERVALS, method="energy_weighted", temperature=0.1
    )
    cold = pool_frames_to_intervals(
        _EMBEDDINGS, _TIMES, _INTERVALS, method="energy_weighted", temperature=100.0
    )
    mean = pool_frames_to_intervals(_EMBEDDINGS, _TIMES, _INTERVALS, method="mean")
    maximum = pool_frames_to_intervals(_EMBEDDINGS, _TIMES, _INTERVALS, method="max")
    # Low temperature sharpens towards the max frame; high temperature -> mean.
    np.testing.assert_allclose(cold, mean, atol=0.1)
    assert np.all(np.abs(hot - maximum) < np.abs(cold - maximum) + 1e-6)


def test_energy_weighted_rejects_non_positive_temperature():
    with pytest.raises(ValueError):
        pool_frames_to_intervals(
            _EMBEDDINGS, _TIMES, _INTERVALS, method="energy_weighted", temperature=0.0
        )


def test_multi_stat_default_concatenates_mean_max_std():
    pooled = pool_frames_to_intervals(_EMBEDDINGS, _TIMES, _INTERVALS, method="multi_stat")
    assert pooled.shape == (2, 6)
    # interval 0 frames: [[0,0],[2,2]] -> mean [1,1], max [2,2], std [1,1]
    np.testing.assert_allclose(pooled[0], [1.0, 1.0, 2.0, 2.0, 1.0, 1.0])
    # interval 1 frames: [[10,10],[14,14]] -> mean [12,12], max [14,14], std [2,2]
    np.testing.assert_allclose(pooled[1], [12.0, 12.0, 14.0, 14.0, 2.0, 2.0])


def test_multi_stat_respects_custom_stats_order_and_dim():
    pooled = pool_frames_to_intervals(
        _EMBEDDINGS, _TIMES, _INTERVALS, method="multi_stat", stats=["min", "max"]
    )
    assert pooled.shape == (2, 4)
    np.testing.assert_allclose(pooled[0], [0.0, 0.0, 2.0, 2.0])
    np.testing.assert_allclose(pooled[1], [10.0, 10.0, 14.0, 14.0])


def test_multi_stat_rejects_unknown_statistic():
    with pytest.raises(ValueError):
        pool_frames_to_intervals(
            _EMBEDDINGS, _TIMES, _INTERVALS, method="multi_stat", stats=["mean", "median"]
        )


def test_multi_stat_rejects_empty_stats():
    with pytest.raises(ValueError):
        pool_frames_to_intervals(_EMBEDDINGS, _TIMES, _INTERVALS, method="multi_stat", stats=[])


# --- empty-interval policy per method ------------------------------------------------------

_EMPTY_EMBEDDINGS = np.asarray([[0.0, 0.0], [2.0, 2.0]])
_EMPTY_TIMES = np.asarray([0.1, 0.9])
_EMPTY_INTERVALS = [(0.0, 1.0), (1.0, 2.0)]  # second interval has no frames


@pytest.mark.parametrize("method", POOLING_METHODS)
def test_empty_zeros_policy_emits_zero_vector_with_method_output_dim(method):
    pooled = pool_frames_to_intervals(
        _EMPTY_EMBEDDINGS, _EMPTY_TIMES, _EMPTY_INTERVALS, method=method, empty="zeros"
    )
    expected_dim = 2 * (3 if method == "multi_stat" else 1)
    assert pooled.shape == (2, expected_dim)
    np.testing.assert_allclose(pooled[1], np.zeros(expected_dim))


def test_empty_nearest_policy_uses_closest_frame():
    # Second interval centre is 1.5; nearest frame is t=0.9 -> [2, 2].
    pooled = pool_frames_to_intervals(
        _EMPTY_EMBEDDINGS, _EMPTY_TIMES, _EMPTY_INTERVALS, method="mean", empty="nearest"
    )
    np.testing.assert_allclose(pooled[1], [2.0, 2.0])


def test_empty_nearest_policy_multi_stat_single_frame_has_zero_std():
    pooled = pool_frames_to_intervals(
        _EMPTY_EMBEDDINGS, _EMPTY_TIMES, _EMPTY_INTERVALS, method="multi_stat", empty="nearest"
    )
    # single frame [2,2] -> mean [2,2], max [2,2], std [0,0]
    np.testing.assert_allclose(pooled[1], [2.0, 2.0, 2.0, 2.0, 0.0, 0.0])


def test_empty_raise_policy_raises():
    with pytest.raises(ValueError):
        pool_frames_to_intervals(
            _EMPTY_EMBEDDINGS, _EMPTY_TIMES, _EMPTY_INTERVALS, method="mean", empty="raise"
        )


# --- validation ----------------------------------------------------------------------------


def test_rejects_unknown_method():
    with pytest.raises(ValueError):
        pool_frames_to_intervals(_EMBEDDINGS, _TIMES, _INTERVALS, method="median")


def test_rejects_unknown_empty_policy():
    with pytest.raises(ValueError):
        pool_frames_to_intervals(_EMBEDDINGS, _TIMES, _INTERVALS, empty="interpolate")


def test_frame_on_interval_end_belongs_to_following_interval():
    # Frame at exactly t=1.0 must land in [1,2), not [0,1).
    embeddings = np.asarray([[5.0, 5.0]])
    times = np.asarray([1.0])
    pooled = pool_frames_to_intervals(
        embeddings, times, [(0.0, 1.0), (1.0, 2.0)], method="mean", empty="zeros"
    )
    np.testing.assert_allclose(pooled[0], [0.0, 0.0])
    np.testing.assert_allclose(pooled[1], [5.0, 5.0])


def test_exported_constants_are_consistent():
    assert set(MULTI_STAT_CHOICES) == {"mean", "max", "std", "min"}
    assert "mean" in POOLING_METHODS and "multi_stat" in POOLING_METHODS
