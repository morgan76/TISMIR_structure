import numpy as np

from tismir.data.annotations import assign_intervals_to_grid
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
