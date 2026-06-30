import numpy as np

from tismir.data.annotations import (
    assign_intervals_to_adjusted_timeline,
    assign_intervals_to_grid,
    project_lower_sections_to_function_labels,
    process_sections,
)
from tismir.data.jams import load_processed_structure_sections
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


def test_process_sections_collapses_to_base_labels_and_merges_neighbors():
    sections = [
        Section(start=0.0, end=1.0, label="verse1"),
        Section(start=1.0, end=2.0, label="verse 2"),
        Section(start=2.0, end=3.0, label="chorus1"),
        Section(start=3.0, end=4.0, label="verse3"),
    ]

    processed = process_sections(sections, {"policy": "base_labels"})

    assert [(section.start, section.end, section.label) for section in processed] == [
        (0.0, 2.0, "verse"),
        (2.0, 3.0, "chorus"),
        (3.0, 4.0, "verse"),
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


def test_process_sections_enumerates_any_repeated_base_label():
    sections = [
        Section(start=0.0, end=1.0, label="prechorus"),
        Section(start=1.0, end=2.0, label="verse"),
        Section(start=2.0, end=3.0, label="prechorus2"),
        Section(start=3.0, end=4.0, label="instrumental"),
        Section(start=4.0, end=5.0, label="instrumental3"),
        Section(start=5.0, end=6.0, label="outro"),
    ]

    processed = process_sections(sections, {"policy": "enumerate_base_occurrences"})

    assert [section.label for section in processed] == [
        "prechorus 1",
        "verse",
        "prechorus 2",
        "instrumental 1",
        "instrumental 2",
        "outro",
    ]


def test_process_sections_can_replace_salami_no_function_with_previous_label():
    sections = [
        Section(start=0.0, end=1.0, label="silence"),
        Section(start=1.0, end=2.0, label="intro"),
        Section(start=2.0, end=3.0, label="no_function"),
        Section(start=3.0, end=4.0, label="verse"),
        Section(start=4.0, end=5.0, label="no_function"),
    ]

    processed = process_sections(
        sections,
        {"policy": "keep", "replace_no_function": True},
    )

    assert [section.label for section in processed] == [
        "silence",
        "intro",
        "intro",
        "verse",
        "verse",
    ]


def test_process_sections_can_replace_leading_salami_no_function_with_next_label():
    sections = [
        Section(start=0.0, end=1.0, label="no_function"),
        Section(start=1.0, end=2.0, label="verse"),
        Section(start=2.0, end=3.0, label="no_function"),
    ]

    processed = process_sections(
        sections,
        {"policy": "merge", "replace_no_function": True},
    )

    assert [(section.start, section.end, section.label) for section in processed] == [
        (0.0, 3.0, "verse"),
    ]


def test_salami_function_merge_policy_replaces_no_function_and_merges():
    sections = [
        Section(start=0.0, end=1.0, label="intro"),
        Section(start=1.0, end=2.0, label="no_function"),
        Section(start=2.0, end=3.0, label="verse"),
    ]

    processed = process_sections(sections, {"policy": "salami_function_merge"})

    assert [(section.start, section.end, section.label) for section in processed] == [
        (0.0, 2.0, "intro"),
        (2.0, 3.0, "verse"),
    ]


def test_salami_function_occurrences_policy_enumerates_repeated_functions():
    sections = [
        Section(start=0.0, end=1.0, label="silence"),
        Section(start=1.0, end=2.0, label="verse"),
        Section(start=2.0, end=3.0, label="no_function"),
        Section(start=3.0, end=4.0, label="chorus"),
        Section(start=4.0, end=5.0, label="verse"),
        Section(start=5.0, end=6.0, label="silence"),
    ]

    processed = process_sections(sections, {"policy": "salami_function_occurrences"})

    assert [section.label for section in processed] == [
        "silence",
        "verse 1",
        "chorus",
        "verse 2",
        "silence",
    ]


def test_project_lower_sections_to_function_labels_uses_maximum_overlap():
    function_sections = [
        Section(start=0.0, end=4.0, label="verse"),
        Section(start=4.0, end=8.0, label="chorus"),
    ]
    lower_sections = [
        Section(start=0.0, end=2.0, label="a"),
        Section(start=2.0, end=5.0, label="b"),
        Section(start=5.0, end=8.0, label="a"),
    ]

    processed = project_lower_sections_to_function_labels(
        function_sections=function_sections,
        lower_sections=lower_sections,
    )

    assert [section.label for section in processed] == [
        "verse subsegment a",
        "verse subsegment b",
        "chorus subsegment a",
    ]


def test_load_processed_structure_sections_supports_salami_projected_lower(tmp_path):
    jams_path = tmp_path / "track.jams"
    _write_salami_jams(jams_path)

    processed = load_processed_structure_sections(
        jams_path,
        annotation_processing={"policy": "salami_function_projected_lower"},
    )

    assert [(section.start, section.end, section.label) for section in processed] == [
        (0.0, 2.0, "silence"),
        (2.0, 4.0, "verse subsegment a"),
        (4.0, 6.0, "verse subsegment b"),
        (6.0, 8.0, "chorus subsegment a"),
    ]


def test_salami_projected_lower_can_use_occurrence_function_labels(tmp_path):
    jams_path = tmp_path / "track.jams"
    _write_salami_repeated_function_jams(jams_path)

    merged = load_processed_structure_sections(
        jams_path,
        annotation_processing={
            "policy": "salami_function_projected_lower",
            "projected_function_policy": "salami_function_merge",
        },
    )
    occurrences = load_processed_structure_sections(
        jams_path,
        annotation_processing={
            "policy": "salami_function_projected_lower",
            "projected_function_policy": "salami_function_occurrences",
        },
    )

    assert [section.label for section in merged] == [
        "verse subsegment a",
        "chorus subsegment b",
        "verse subsegment c",
    ]
    assert [section.label for section in occurrences] == [
        "verse 1 subsegment a",
        "chorus subsegment b",
        "verse 2 subsegment c",
    ]


def test_salami_function_policy_can_select_richest_annotator(tmp_path):
    jams_path = tmp_path / "track.jams"
    _write_salami_multi_annotator_jams(jams_path)

    first = load_processed_structure_sections(
        jams_path,
        annotation_processing={"policy": "salami_function_merge"},
    )
    richest = load_processed_structure_sections(
        jams_path,
        annotation_processing={
            "policy": "salami_function_merge",
            "annotation_selection": "richest_function",
        },
    )

    assert [(section.start, section.end, section.label) for section in first] == [
        (0.0, 8.0, "silence"),
    ]
    assert [(section.start, section.end, section.label) for section in richest] == [
        (0.0, 2.0, "silence"),
        (2.0, 4.0, "intro"),
        (4.0, 6.0, "verse"),
        (6.0, 8.0, "outro"),
    ]


def _write_salami_jams(path):
    import jams

    jam = jams.JAMS()
    jam.file_metadata.duration = 8.0
    function = jams.Annotation(namespace="segment_salami_function")
    for start, end, label in [
        (0.0, 2.0, "silence"),
        (2.0, 4.0, "verse"),
        (4.0, 6.0, "no_function"),
        (6.0, 8.0, "chorus"),
    ]:
        function.append(time=start, duration=end - start, value=label)
    lower = jams.Annotation(namespace="segment_salami_lower")
    for start, end, label in [
        (0.0, 2.0, "Silence"),
        (2.0, 4.0, "a"),
        (4.0, 6.0, "b"),
        (6.0, 8.0, "a"),
    ]:
        lower.append(time=start, duration=end - start, value=label)
    jam.annotations.extend([function, lower])
    jam.save(str(path))


def _write_salami_repeated_function_jams(path):
    import jams

    jam = jams.JAMS()
    jam.file_metadata.duration = 6.0
    function = jams.Annotation(namespace="segment_salami_function")
    for start, end, label in [
        (0.0, 2.0, "verse"),
        (2.0, 4.0, "chorus"),
        (4.0, 6.0, "verse"),
    ]:
        function.append(time=start, duration=end - start, value=label)
    lower = jams.Annotation(namespace="segment_salami_lower")
    for start, end, label in [
        (0.0, 2.0, "a"),
        (2.0, 4.0, "b"),
        (4.0, 6.0, "c"),
    ]:
        lower.append(time=start, duration=end - start, value=label)
    jam.annotations.extend([function, lower])
    jam.save(str(path))


def _write_salami_multi_annotator_jams(path):
    import jams

    jam = jams.JAMS()
    jam.file_metadata.duration = 8.0
    poor = jams.Annotation(namespace="segment_salami_function")
    for start, end, label in [
        (0.0, 2.0, "silence"),
        (2.0, 4.0, "no_function"),
        (4.0, 6.0, "no_function"),
        (6.0, 8.0, "silence"),
    ]:
        poor.append(time=start, duration=end - start, value=label)
    rich = jams.Annotation(namespace="segment_salami_function")
    for start, end, label in [
        (0.0, 2.0, "silence"),
        (2.0, 4.0, "intro"),
        (4.0, 6.0, "verse"),
        (6.0, 8.0, "outro"),
    ]:
        rich.append(time=start, duration=end - start, value=label)
    jam.annotations.extend([poor, rich])
    jam.save(str(path))
