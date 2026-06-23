import json
import wave
from pathlib import Path

import jams
import numpy as np

from tismir.data.manifest import save_manifest
from tismir.data.schemas import Track
from tismir.preprocessing.audio import preprocess_track_audio
from tismir.preprocessing.text import preprocess_dataset_text
from tismir.training.data import StructureEmbeddingDataset, collate_training_examples


def test_structure_embedding_dataset_loads_targets(tmp_path):
    audio_path = tmp_path / "audio.wav"
    jams_path = tmp_path / "audio.jams"
    manifest_path = tmp_path / "manifest.jsonl"
    _write_silent_wav(audio_path, duration=2.0, sample_rate=8000)
    _write_jams(jams_path)
    track = Track(
        track_id="track",
        audio_path=audio_path,
        jams_path=jams_path,
        dataset="dataset",
    )
    save_manifest(manifest_path, [track])

    preprocess_track_audio(
        track=track,
        output_root=tmp_path / "audio_embeddings",
        audio_encoder_name="placeholder",
        audio_encoder_params={"output_dim": 4, "frame_rate": 4.0},
        beat_tracker_name="uniform",
        beat_tracker_params={"beat_period": 0.5},
        pooling={"method": "mean", "keep_dense": True},
    )
    preprocess_dataset_text(
        tracks=[track],
        output_root=tmp_path / "text_embeddings",
        text_encoder_name="placeholder",
        text_encoder_params={"output_dim": 3},
    )

    dataset = StructureEmbeddingDataset(
        manifest=manifest_path,
        audio_embedding_root=tmp_path / "audio_embeddings",
        audio_encoder="placeholder",
        text_embedding_root=tmp_path / "text_embeddings",
        text_encoder="placeholder",
    )
    example = dataset[0]

    assert example.audio.shape == (4, 4)
    assert example.text.shape == (2, 3)
    assert example.labels == ["intro", "verse"]
    np.testing.assert_array_equal(example.targets, [0, 0, 1, 1])

    batch = collate_training_examples([example])
    assert tuple(batch["audio"].shape) == (1, 4, 4)
    assert tuple(batch["text"].shape) == (2, 3)
    assert tuple(batch["targets"].shape) == (1, 4)
    assert batch["mask"].all()


def test_structure_embedding_dataset_can_use_track_label_candidates(tmp_path):
    audio_path = tmp_path / "audio.wav"
    jams_path = tmp_path / "audio.jams"
    extra_jams_path = tmp_path / "extra.jams"
    manifest_path = tmp_path / "manifest.jsonl"
    _write_silent_wav(audio_path, duration=2.0, sample_rate=8000)
    _write_jams(jams_path)
    _write_extra_jams(extra_jams_path)
    track = Track(
        track_id="track",
        audio_path=audio_path,
        jams_path=jams_path,
        dataset="dataset",
    )
    extra_track = Track(
        track_id="extra",
        audio_path=tmp_path / "extra.wav",
        jams_path=extra_jams_path,
        dataset="dataset",
    )
    save_manifest(manifest_path, [track])

    preprocess_track_audio(
        track=track,
        output_root=tmp_path / "audio_embeddings",
        audio_encoder_name="placeholder",
        audio_encoder_params={"output_dim": 4, "frame_rate": 4.0},
        beat_tracker_name="uniform",
        beat_tracker_params={"beat_period": 0.5},
        pooling={"method": "mean", "keep_dense": True},
    )
    preprocess_dataset_text(
        tracks=[track, extra_track],
        output_root=tmp_path / "text_embeddings",
        text_encoder_name="placeholder",
        text_encoder_params={"output_dim": 3},
    )

    dataset = StructureEmbeddingDataset(
        manifest=manifest_path,
        audio_embedding_root=tmp_path / "audio_embeddings",
        audio_encoder="placeholder",
        text_embedding_root=tmp_path / "text_embeddings",
        text_encoder="placeholder",
        candidate_label_strategy="track_labels",
    )
    example = dataset[0]

    assert example.labels == ["intro", "verse"]
    assert example.text.shape == (2, 3)
    np.testing.assert_array_equal(example.targets, [0, 0, 1, 1])


def test_structure_embedding_dataset_can_merge_consecutive_same_labels(tmp_path):
    audio_path = tmp_path / "audio.wav"
    jams_path = tmp_path / "audio.jams"
    manifest_path = tmp_path / "manifest.jsonl"
    _write_silent_wav(audio_path, duration=2.0, sample_rate=8000)
    _write_repeated_jams(jams_path)
    track = Track(
        track_id="track",
        audio_path=audio_path,
        jams_path=jams_path,
        dataset="dataset",
    )
    save_manifest(manifest_path, [track])

    preprocess_track_audio(
        track=track,
        output_root=tmp_path / "audio_embeddings",
        audio_encoder_name="placeholder",
        audio_encoder_params={"output_dim": 4, "frame_rate": 4.0},
        beat_tracker_name="uniform",
        beat_tracker_params={"beat_period": 0.5},
        pooling={"method": "mean", "keep_dense": True},
    )
    preprocess_dataset_text(
        tracks=[track],
        output_root=tmp_path / "text_embeddings",
        text_encoder_name="placeholder",
        text_encoder_params={"output_dim": 3},
        annotation_processing={"policy": "merge"},
    )

    dataset = StructureEmbeddingDataset(
        manifest=manifest_path,
        audio_embedding_root=tmp_path / "audio_embeddings",
        audio_encoder="placeholder",
        text_embedding_root=tmp_path / "text_embeddings",
        text_encoder="placeholder",
        candidate_label_strategy="track_labels",
        annotation_processing={"policy": "merge"},
    )
    example = dataset[0]

    assert example.labels == ["verse"]
    np.testing.assert_array_equal(example.targets, [0, 0, 0, 0])


def test_structure_embedding_dataset_can_enumerate_consecutive_same_labels(tmp_path):
    audio_path = tmp_path / "audio.wav"
    jams_path = tmp_path / "audio.jams"
    manifest_path = tmp_path / "manifest.jsonl"
    _write_silent_wav(audio_path, duration=2.0, sample_rate=8000)
    _write_repeated_jams(jams_path)
    track = Track(
        track_id="track",
        audio_path=audio_path,
        jams_path=jams_path,
        dataset="dataset",
    )
    save_manifest(manifest_path, [track])

    preprocess_track_audio(
        track=track,
        output_root=tmp_path / "audio_embeddings",
        audio_encoder_name="placeholder",
        audio_encoder_params={"output_dim": 4, "frame_rate": 4.0},
        beat_tracker_name="uniform",
        beat_tracker_params={"beat_period": 0.5},
        pooling={"method": "mean", "keep_dense": True},
    )
    preprocess_dataset_text(
        tracks=[track],
        output_root=tmp_path / "text_embeddings",
        text_encoder_name="placeholder",
        text_encoder_params={"output_dim": 3},
        annotation_processing={"policy": "enumerate_consecutive_repeats"},
    )

    dataset = StructureEmbeddingDataset(
        manifest=manifest_path,
        audio_embedding_root=tmp_path / "audio_embeddings",
        audio_encoder="placeholder",
        text_embedding_root=tmp_path / "text_embeddings",
        text_encoder="placeholder",
        candidate_label_strategy="track_labels",
        annotation_processing={"policy": "enumerate_consecutive_repeats"},
    )
    example = dataset[0]

    assert example.labels == ["verse 1", "verse 2"]
    np.testing.assert_array_equal(example.targets, [0, 0, 1, 1])


def test_structure_embedding_dataset_can_enumerate_base_occurrences(tmp_path):
    audio_path = tmp_path / "audio.wav"
    jams_path = tmp_path / "audio.jams"
    manifest_path = tmp_path / "manifest.jsonl"
    _write_silent_wav(audio_path, duration=2.0, sample_rate=8000)
    _write_numbered_repeated_jams(jams_path)
    track = Track(
        track_id="track",
        audio_path=audio_path,
        jams_path=jams_path,
        dataset="dataset",
    )
    save_manifest(manifest_path, [track])

    preprocess_track_audio(
        track=track,
        output_root=tmp_path / "audio_embeddings",
        audio_encoder_name="placeholder",
        audio_encoder_params={"output_dim": 4, "frame_rate": 4.0},
        beat_tracker_name="uniform",
        beat_tracker_params={"beat_period": 0.5},
        pooling={"method": "mean", "keep_dense": True},
    )
    preprocess_dataset_text(
        tracks=[track],
        output_root=tmp_path / "text_embeddings",
        text_encoder_name="placeholder",
        text_encoder_params={"output_dim": 3},
        annotation_processing={"policy": "enumerate_base_occurrences"},
    )

    dataset = StructureEmbeddingDataset(
        manifest=manifest_path,
        audio_embedding_root=tmp_path / "audio_embeddings",
        audio_encoder="placeholder",
        text_embedding_root=tmp_path / "text_embeddings",
        text_encoder="placeholder",
        candidate_label_strategy="track_labels",
        annotation_processing={"policy": "enumerate_base_occurrences"},
    )
    example = dataset[0]

    assert example.labels == ["verse 1", "verse 2"]
    np.testing.assert_array_equal(example.targets, [0, 0, 1, 1])


def test_structure_embedding_dataset_samples_annotation_processing_by_epoch(tmp_path):
    audio_path = tmp_path / "audio.wav"
    jams_path = tmp_path / "audio.jams"
    manifest_path = tmp_path / "manifest.jsonl"
    _write_silent_wav(audio_path, duration=2.0, sample_rate=8000)
    _write_repeated_jams(jams_path)
    track = Track(
        track_id="track",
        audio_path=audio_path,
        jams_path=jams_path,
        dataset="dataset",
    )
    save_manifest(manifest_path, [track])

    preprocess_track_audio(
        track=track,
        output_root=tmp_path / "audio_embeddings",
        audio_encoder_name="placeholder",
        audio_encoder_params={"output_dim": 4, "frame_rate": 4.0},
        beat_tracker_name="uniform",
        beat_tracker_params={"beat_period": 0.5},
        pooling={"method": "mean", "keep_dense": True},
    )
    _write_text_embedding_cache(
        tmp_path / "text_embeddings" / "placeholder" / "dataset",
        labels=["verse", "verse 1", "verse 2"],
        output_dim=3,
    )

    dataset = StructureEmbeddingDataset(
        manifest=manifest_path,
        audio_embedding_root=tmp_path / "audio_embeddings",
        audio_encoder="placeholder",
        text_embedding_root=tmp_path / "text_embeddings",
        text_encoder="placeholder",
        candidate_label_strategy="track_labels",
        annotation_processing={
            "policy": "random",
            "seed": 0,
            "choices": ["merge", "enumerate_consecutive_repeats"],
        },
    )

    seen_labels = set()
    for epoch in range(64):
        dataset.set_epoch(epoch)
        first = dataset[0]
        second = dataset[0]
        assert second.labels == first.labels
        np.testing.assert_array_equal(second.targets, first.targets)
        seen_labels.add(tuple(first.labels))
        if len(seen_labels) == 2:
            break

    assert ("verse",) in seen_labels
    assert ("verse 1", "verse 2") in seen_labels


def _write_silent_wav(path: Path, duration: float, sample_rate: int) -> None:
    num_frames = int(duration * sample_rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * num_frames)


def _write_jams(path: Path) -> None:
    jam = jams.JAMS()
    jam.file_metadata.duration = 2.0
    annotation = jams.Annotation(namespace="segment_open")
    annotation.append(time=0.0, duration=1.0, value="intro")
    annotation.append(time=1.0, duration=1.0, value="verse")
    jam.annotations.append(annotation)
    jam.save(str(path))


def _write_extra_jams(path: Path) -> None:
    jam = jams.JAMS()
    jam.file_metadata.duration = 2.0
    annotation = jams.Annotation(namespace="segment_open")
    annotation.append(time=0.0, duration=1.0, value="chorus")
    annotation.append(time=1.0, duration=1.0, value="outro")
    jam.annotations.append(annotation)
    jam.save(str(path))


def _write_repeated_jams(path: Path) -> None:
    jam = jams.JAMS()
    jam.file_metadata.duration = 2.0
    annotation = jams.Annotation(namespace="segment_open")
    annotation.append(time=0.0, duration=1.0, value="verse")
    annotation.append(time=1.0, duration=1.0, value="verse")
    jam.annotations.append(annotation)
    jam.save(str(path))


def _write_numbered_repeated_jams(path: Path) -> None:
    jam = jams.JAMS()
    jam.file_metadata.duration = 2.0
    annotation = jams.Annotation(namespace="segment_open")
    annotation.append(time=0.0, duration=1.0, value="verse2")
    annotation.append(time=1.0, duration=1.0, value="verse5")
    jam.annotations.append(annotation)
    jam.save(str(path))


def _write_text_embedding_cache(path: Path, labels: list[str], output_dim: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "labels.json").write_text(json.dumps({"labels": labels}), encoding="utf-8")
    embeddings = np.arange(len(labels) * output_dim, dtype=np.float32).reshape(
        len(labels),
        output_dim,
    )
    np.save(path / "embeddings.npy", embeddings)
