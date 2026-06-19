import numpy as np

from tismir.data.annotations import assign_intervals_to_adjusted_timeline, assign_intervals_to_grid
from tismir.data.schemas import Section


def test_assign_intervals_to_grid_uses_maximum_overlap():
    sections = [
        Section(start=0.0, end=2.0, label="intro"),
        Section(start=2.0, end=5.0, label="verse"),
    ]
    intervals = [(0.5, 1.5), (1.5, 2.5), (3.0, 4.0)]
    labels = ["intro", "verse"]

    targets = assign_intervals_to_grid(intervals, sections, labels=labels)

    np.testing.assert_array_equal(targets, [0, 0, 1])


def test_assign_intervals_to_grid_returns_labels_without_candidate_set():
    sections = [Section(start=0.0, end=1.0, label="chorus")]

    targets = assign_intervals_to_grid([(0.0, 1.0)], sections)

    assert targets.tolist() == ["chorus"]


def test_assign_intervals_to_grid_can_ignore_unannotated_intervals():
    sections = [Section(start=0.0, end=1.0, label="chorus")]

    targets = assign_intervals_to_grid(
        [(0.0, 1.0), (1.0, 2.0)],
        sections,
        labels=["chorus"],
        no_overlap_value=-100,
    )

    np.testing.assert_array_equal(targets, [0, -100])


def test_adjusted_timeline_maps_tail_to_silence_like_label():
    sections = [Section(start=1.0, end=2.0, label="verse")]
    intervals = [(0.0, 0.5), (1.0, 1.5), (2.5, 3.0)]

    targets = assign_intervals_to_adjusted_timeline(
        intervals,
        sections,
        duration=3.0,
        labels=["Silence", "verse"],
        no_overlap_value=-100,
    )

    np.testing.assert_array_equal(targets, [0, 1, 0])


def test_adjusted_timeline_ignores_synthetic_boundary_without_silence_label():
    sections = [Section(start=1.0, end=2.0, label="verse")]
    intervals = [(0.0, 0.5), (1.0, 1.5), (2.5, 3.0)]

    targets = assign_intervals_to_adjusted_timeline(
        intervals,
        sections,
        duration=3.0,
        labels=["verse"],
        no_overlap_value=-100,
    )

    np.testing.assert_array_equal(targets, [-100, 0, -100])
