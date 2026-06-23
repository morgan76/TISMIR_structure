import json
from pathlib import Path

import jams
import numpy as np

from tismir.data.schemas import Track
from tismir.preprocessing.text import preprocess_dataset_text


def test_preprocess_dataset_text_writes_label_embeddings(tmp_path):
    jams_path = tmp_path / "track.jams"
    _write_jams(jams_path)
    track = Track(
        track_id="track",
        audio_path=tmp_path / "track.wav",
        jams_path=jams_path,
        dataset="dataset",
    )

    results = preprocess_dataset_text(
        tracks=[track],
        output_root=tmp_path / "text",
        text_encoder_name="placeholder",
        text_encoder_params={"output_dim": 8},
        prompt={"template": "music section: {label}", "normalize_whitespace": True},
        label_normalization={"name": "harmonix", "disambiguate_duplicates": True},
    )

    assert len(results) == 1
    output_dir = Path(results[0].output_dir)
    assert np.load(output_dir / "embeddings.npy").shape == (4, 8)

    labels = json.loads((output_dir / "labels.json").read_text(encoding="utf-8"))
    assert labels["labels"] == ["fadeout", "verseinst", "instrumentalverse", "inst6"]
    assert labels["text_labels"] == [
        "fade out",
        "instrumental verse",
        "instrumental verse",
        "instrumental 6",
    ]
    assert labels["prompt_labels"] == [
        "fade out",
        "instrumental verse (verseinst)",
        "instrumental verse (instrumentalverse)",
        "instrumental 6",
    ]
    assert labels["prompts"] == [
        "music section: fade out",
        "music section: instrumental verse (verseinst)",
        "music section: instrumental verse (instrumentalverse)",
        "music section: instrumental 6",
    ]


def test_preprocess_dataset_text_uses_annotation_processing(tmp_path):
    jams_path = tmp_path / "track.jams"
    _write_repeated_jams(jams_path)
    track = Track(
        track_id="track",
        audio_path=tmp_path / "track.wav",
        jams_path=jams_path,
        dataset="dataset",
    )

    results = preprocess_dataset_text(
        tracks=[track],
        output_root=tmp_path / "text",
        text_encoder_name="placeholder",
        text_encoder_params={"output_dim": 4},
        annotation_processing={"policy": "enumerate_consecutive_repeats"},
    )

    labels = json.loads((Path(results[0].output_dir) / "labels.json").read_text(encoding="utf-8"))
    assert labels["labels"] == ["verse 1", "verse 2", "chorus", "verse 3"]


def test_preprocess_dataset_text_can_enumerate_base_occurrences(tmp_path):
    jams_path = tmp_path / "track.jams"
    _write_numbered_repeated_jams(jams_path)
    track = Track(
        track_id="track",
        audio_path=tmp_path / "track.wav",
        jams_path=jams_path,
        dataset="dataset",
    )

    results = preprocess_dataset_text(
        tracks=[track],
        output_root=tmp_path / "text",
        text_encoder_name="placeholder",
        text_encoder_params={"output_dim": 4},
        annotation_processing={"policy": "enumerate_base_occurrences"},
    )

    labels = json.loads((Path(results[0].output_dir) / "labels.json").read_text(encoding="utf-8"))
    assert labels["labels"] == ["verse 1", "chorus", "verse 2", "verse 3"]


def test_preprocess_dataset_text_random_annotation_processing_encodes_union(tmp_path):
    jams_path = tmp_path / "track.jams"
    _write_repeated_jams(jams_path)
    track = Track(
        track_id="track",
        audio_path=tmp_path / "track.wav",
        jams_path=jams_path,
        dataset="dataset",
    )

    results = preprocess_dataset_text(
        tracks=[track],
        output_root=tmp_path / "text",
        text_encoder_name="placeholder",
        text_encoder_params={"output_dim": 4},
        annotation_processing={
            "policy": "random",
            "choices": ["merge", "enumerate_consecutive_repeats"],
        },
    )

    labels = json.loads((Path(results[0].output_dir) / "labels.json").read_text(encoding="utf-8"))
    assert labels["labels"] == ["verse", "chorus", "verse 1", "verse 2", "verse 3"]


def test_preprocess_dataset_text_can_use_descriptive_music_structure_prompts(tmp_path):
    jams_path = tmp_path / "track.jams"
    _write_numbered_repeated_jams(jams_path)
    track = Track(
        track_id="track",
        audio_path=tmp_path / "track.wav",
        jams_path=jams_path,
        dataset="dataset",
    )

    results = preprocess_dataset_text(
        tracks=[track],
        output_root=tmp_path / "text",
        text_encoder_name="placeholder",
        text_encoder_params={"output_dim": 4},
        prompt={"mode": "descriptive", "normalize_whitespace": True},
        label_normalization={"name": "harmonix", "normalize_whitespace": True},
        annotation_processing={"policy": "enumerate_base_occurrences"},
    )

    labels = json.loads((Path(results[0].output_dir) / "labels.json").read_text(encoding="utf-8"))
    assert labels["prompts"][0] == (
        "Music structure label: verse 1. Base type: verse. "
        "Occurrence: the 1st verse section in chronological order. "
        "Meaning: a recurring lyrical section, usually distinct from the chorus. "
        "Use this label for frames belonging to this section."
    )


def test_preprocess_dataset_text_uses_bare_prompt_mode_by_default(tmp_path):
    jams_path = tmp_path / "track.jams"
    _write_numbered_repeated_jams(jams_path)
    track = Track(
        track_id="track",
        audio_path=tmp_path / "track.wav",
        jams_path=jams_path,
        dataset="dataset",
    )

    results = preprocess_dataset_text(
        tracks=[track],
        output_root=tmp_path / "text",
        text_encoder_name="placeholder",
        text_encoder_params={"output_dim": 4},
        label_normalization={"name": "harmonix", "normalize_whitespace": True},
        annotation_processing={"policy": "enumerate_base_occurrences"},
    )

    labels = json.loads((Path(results[0].output_dir) / "labels.json").read_text(encoding="utf-8"))
    metadata = json.loads((Path(results[0].output_dir) / "metadata.json").read_text(encoding="utf-8"))
    assert labels["prompts"][:2] == ["verse 1", "chorus"]
    assert metadata["prompt"]["mode"] == "bare"


def test_preprocess_dataset_text_can_use_compact_prompt_mode(tmp_path):
    jams_path = tmp_path / "track.jams"
    _write_numbered_repeated_jams(jams_path)
    track = Track(
        track_id="track",
        audio_path=tmp_path / "track.wav",
        jams_path=jams_path,
        dataset="dataset",
    )

    results = preprocess_dataset_text(
        tracks=[track],
        output_root=tmp_path / "text",
        text_encoder_name="placeholder",
        text_encoder_params={"output_dim": 4},
        prompt={"mode": "compact"},
        label_normalization={"name": "harmonix", "normalize_whitespace": True},
        annotation_processing={"policy": "enumerate_base_occurrences"},
    )

    labels = json.loads((Path(results[0].output_dir) / "labels.json").read_text(encoding="utf-8"))
    assert labels["prompts"][:2] == [
        "Music structure label: verse 1",
        "Music structure label: chorus",
    ]


def test_preprocess_dataset_text_can_use_occurrence_descriptive_prompt_mode(tmp_path):
    jams_path = tmp_path / "track.jams"
    _write_numbered_repeated_jams(jams_path)
    track = Track(
        track_id="track",
        audio_path=tmp_path / "track.wav",
        jams_path=jams_path,
        dataset="dataset",
    )

    results = preprocess_dataset_text(
        tracks=[track],
        output_root=tmp_path / "text",
        text_encoder_name="placeholder",
        text_encoder_params={"output_dim": 4},
        prompt={"mode": "occurrence_descriptive"},
        label_normalization={"name": "harmonix", "normalize_whitespace": True},
        annotation_processing={"policy": "enumerate_base_occurrences"},
    )

    labels = json.loads((Path(results[0].output_dir) / "labels.json").read_text(encoding="utf-8"))
    assert labels["prompts"][:2] == [
        (
            "Music structure label: verse 1. Meaning: the first occurrence of "
            "the verse section in this song. Use this label for frames belonging "
            "to the first occurrence of the verse section in this song."
        ),
        (
            "Music structure label: chorus. Meaning: the chorus section. "
            "Use this label for frames belonging to the chorus section."
        ),
    ]


def _write_jams(path: Path) -> None:
    jam = jams.JAMS()
    jam.file_metadata.duration = 4.0
    annotation = jams.Annotation(namespace="segment_open")
    annotation.append(time=0.0, duration=1.0, value="fadeout")
    annotation.append(time=1.0, duration=1.0, value="verseinst")
    annotation.append(time=2.0, duration=1.0, value="instrumentalverse")
    annotation.append(time=3.0, duration=1.0, value="inst6")
    jam.annotations.append(annotation)
    jam.save(str(path))


def _write_repeated_jams(path: Path) -> None:
    jam = jams.JAMS()
    jam.file_metadata.duration = 4.0
    annotation = jams.Annotation(namespace="segment_open")
    annotation.append(time=0.0, duration=1.0, value="verse")
    annotation.append(time=1.0, duration=1.0, value="verse")
    annotation.append(time=2.0, duration=1.0, value="chorus")
    annotation.append(time=3.0, duration=1.0, value="verse")
    jam.annotations.append(annotation)
    jam.save(str(path))


def _write_numbered_repeated_jams(path: Path) -> None:
    jam = jams.JAMS()
    jam.file_metadata.duration = 4.0
    annotation = jams.Annotation(namespace="segment_open")
    annotation.append(time=0.0, duration=1.0, value="verse2")
    annotation.append(time=1.0, duration=1.0, value="chorus")
    annotation.append(time=2.0, duration=1.0, value="verse5")
    annotation.append(time=3.0, duration=1.0, value="verse2")
    jam.annotations.append(annotation)
    jam.save(str(path))
