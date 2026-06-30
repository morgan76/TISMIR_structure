import numpy as np

from tismir.decoding.segments import (
    decode_label_indices,
    merge_frame_labels_with_boundary_scores,
    remove_short_segments,
    smooth_logits,
)


def test_smooth_logits_mean():
    logits = np.asarray([[0.0, 1.0], [10.0, 0.0], [0.0, 1.0]])

    smoothed = smooth_logits(logits, window=3, mode="mean")

    np.testing.assert_allclose(smoothed[1], [10.0 / 3.0, 2.0 / 3.0])


def test_viterbi_decode_matches_argmax_without_transition_penalty():
    logits = np.asarray(
        [
            [3.0, 0.0],
            [0.0, 2.0],
            [4.0, 0.0],
        ]
    )

    decoded = decode_label_indices(logits, strategy="viterbi", transition_penalty=0.0)

    np.testing.assert_array_equal(decoded, [0, 1, 0])


def test_viterbi_decode_penalizes_short_label_changes():
    logits = np.asarray(
        [
            [3.0, 0.0],
            [0.0, 2.0],
            [3.0, 0.0],
        ]
    )

    decoded = decode_label_indices(logits, strategy="viterbi", transition_penalty=2.0)

    np.testing.assert_array_equal(decoded, [0, 0, 0])


def test_viterbi_decode_uses_high_boundary_probability_to_encourage_switches():
    logits = np.asarray(
        [
            [2.0, 0.0],
            [0.0, 2.0],
            [2.0, 0.0],
        ]
    )

    decoded = decode_label_indices(
        logits,
        strategy="viterbi",
        transition_penalty=3.0,
        boundary_probabilities=np.asarray([0.95, 0.95]),
        boundary_weight=3.0,
    )

    np.testing.assert_array_equal(decoded, [0, 1, 0])


def test_viterbi_decode_uses_low_boundary_probability_to_encourage_stays():
    logits = np.asarray(
        [
            [2.0, 0.0],
            [0.0, 2.0],
            [2.0, 0.0],
        ]
    )

    decoded = decode_label_indices(
        logits,
        strategy="viterbi",
        transition_penalty=0.0,
        boundary_probabilities=np.asarray([0.05, 0.05]),
        boundary_weight=3.0,
    )

    np.testing.assert_array_equal(decoded, [0, 0, 0])


def test_remove_short_segments_merges_into_neighbor():
    segments = [
        (0.0, 5.0, "verse"),
        (5.0, 5.5, "bridge"),
        (5.5, 10.0, "chorus"),
    ]

    cleaned = remove_short_segments(segments, min_duration=1.0)

    assert cleaned == [(0.0, 5.5, "verse"), (5.5, 10.0, "chorus")]


def test_remove_short_segments_collapses_matching_neighbors():
    segments = [
        (0.0, 5.0, "verse"),
        (5.0, 5.5, "bridge"),
        (5.5, 10.0, "verse"),
    ]

    cleaned = remove_short_segments(segments, min_duration=1.0)

    assert cleaned == [(0.0, 10.0, "verse")]


def test_remove_short_segments_can_preserve_matching_neighbors():
    segments = [
        (0.0, 5.0, "verse"),
        (5.0, 10.0, "verse"),
    ]

    cleaned = remove_short_segments(
        segments,
        min_duration=1.0,
        merge_same_label_neighbors=False,
    )

    assert cleaned == segments


def test_merge_frame_labels_with_boundary_scores_splits_same_label_peak():
    intervals = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)]
    labels = ["verse", "verse", "verse", "verse"]
    probabilities = np.asarray([0.1, 0.9, 0.1])

    segments = merge_frame_labels_with_boundary_scores(
        intervals,
        labels,
        boundary_probabilities=probabilities,
        threshold=0.5,
        mode="peaks",
    )

    assert segments == [(0.0, 1.5, "verse"), (1.5, 4.0, "verse")]


def test_merge_frame_labels_with_boundary_scores_keeps_label_change_behavior():
    intervals = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
    labels = ["verse", "chorus", "chorus"]
    probabilities = np.asarray([0.9, 0.1])

    segments = merge_frame_labels_with_boundary_scores(
        intervals,
        labels,
        boundary_probabilities=probabilities,
        threshold=0.5,
        mode="peaks",
    )

    assert segments == [(0.0, 1.0, "verse"), (1.0, 3.0, "chorus")]
