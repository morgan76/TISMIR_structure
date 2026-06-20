import wave
from pathlib import Path

import jams

from tismir.data.manifest import save_manifest
from tismir.data.schemas import Track
from tismir.preprocessing.audio import preprocess_track_audio
from tismir.preprocessing.text import preprocess_dataset_text
from tismir.training.loop import train_projection_baseline


def test_train_projection_baseline_smoke(tmp_path):
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
        text_encoder_params={"output_dim": 4},
    )

    metrics = train_projection_baseline(
        {
            "seed": 0,
            "device": "cpu",
            "output_dir": str(tmp_path / "train"),
            "model": {
                "audio": {"hidden_dim": 8, "output_dim": 4},
                "text": {"hidden_dim": 8, "output_dim": 4},
                "similarity": {"temperature": 0.1, "normalize": True},
            },
            "data": {
                "manifest": str(manifest_path),
                "audio_embedding_root": str(tmp_path / "audio_embeddings"),
                "audio_encoder": "placeholder",
                "text_embedding_root": str(tmp_path / "text_embeddings"),
                "text_encoder": "placeholder",
                "audio_embedding_key": "beat_sync",
                "namespace": "segment_open",
                "candidate_label_strategy": "track_labels",
                "ignore_index": -100,
            },
            "optimization": {
                "batch_size": 1,
                "gradient_accumulation_steps": 2,
                "max_epochs": 2,
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "shuffle": False,
                "lr_scheduler": {
                    "name": "reduce_on_plateau",
                    "patience": 1,
                    "factor": 0.5,
                    "min_lr": 1e-5,
                },
            },
            "validation": {
                "manifest": str(manifest_path),
            },
        }
    )

    assert (tmp_path / "train" / "checkpoint.pt").exists()
    assert (tmp_path / "train" / "best_checkpoint.pt").exists()
    assert (tmp_path / "train" / "metrics.json").exists()
    assert metrics["final_loss"] is not None
    assert metrics["final_val_loss"] is not None
    assert metrics["best_val_loss"] is not None
    assert metrics["best_epoch"] is not None
    assert metrics["epochs_trained"] == 2
    assert metrics["stopped_early"] is False
    assert metrics["gradient_accumulation_steps"] == 2
    assert metrics["history"][0]["optimizer_steps"] == 1.0


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
