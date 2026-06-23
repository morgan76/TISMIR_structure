import numpy as np

from tismir.decoding.segments import decode_label_indices, remove_short_segments, smooth_logits


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
