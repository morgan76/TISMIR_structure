import numpy as np

from tismir.data.annotations import (
    assign_intervals_to_adjusted_timeline,
    assign_intervals_to_grid,
    process_sections,
)
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


def test_process_sections_merges_consecutive_same_labels():
    sections = [
        Section(start=0.0, end=1.0, label="verse"),
        Section(start=1.0, end=2.0, label="verse"),
        Section(start=2.0, end=3.0, label="chorus"),
    ]

    processed = process_sections(sections, {"policy": "merge"})

    assert [(section.start, section.end, section.label) for section in processed] == [
        (0.0, 2.0, "verse"),
        (2.0, 3.0, "chorus"),
    ]


def test_process_sections_enumerates_all_repeated_labels():
    sections = [
        Section(start=0.0, end=1.0, label="verse"),
        Section(start=1.0, end=2.0, label="chorus"),
        Section(start=2.0, end=3.0, label="verse"),
        Section(start=3.0, end=4.0, label="verse"),
        Section(start=4.0, end=5.0, label="outro"),
    ]

    processed = process_sections(sections, {"policy": "enumerate_all_occurrences"})

    assert [section.label for section in processed] == [
        "verse 1",
        "chorus",
        "verse 2",
        "verse 3",
        "outro",
    ]


def test_process_sections_enumerates_labels_with_consecutive_repeats_only():
    sections = [
        Section(start=0.0, end=1.0, label="verse"),
        Section(start=1.0, end=2.0, label="chorus"),
        Section(start=2.0, end=3.0, label="verse"),
        Section(start=3.0, end=4.0, label="verse"),
        Section(start=4.0, end=5.0, label="chorus"),
    ]

    processed = process_sections(sections, {"policy": "enumerate_consecutive_repeats"})

    assert [section.label for section in processed] == [
        "verse 1",
        "chorus",
        "verse 2",
        "verse 3",
        "chorus",
    ]


def test_process_sections_enumerates_base_occurrences():
    sections = [
        Section(start=0.0, end=1.0, label="verse2"),
        Section(start=1.0, end=2.0, label="chorus"),
        Section(start=2.0, end=3.0, label="verse 5"),
        Section(start=3.0, end=4.0, label="verse2"),
        Section(start=4.0, end=5.0, label="bridge1a"),
        Section(start=5.0, end=6.0, label="bridge 2"),
    ]

    processed = process_sections(sections, {"policy": "enumerate_base_occurrences"})

    assert [section.label for section in processed] == [
        "verse 1",
        "chorus",
        "verse 2",
        "verse 3",
        "bridge 1",
        "bridge 2",
    ]
