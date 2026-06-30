import json

import jams
import numpy as np

from tismir.evaluation import evaluate_pair, format_evaluation
from tismir.evaluation.segments import TrackEvaluation, evaluate_prediction_manifest, summarize_evaluations
from tismir.data.manifest import save_manifest
from tismir.data.schemas import Track


def test_evaluate_pair_identical_segmentations(tmp_path):
    ref = tmp_path / "ref.jams"
    pred = tmp_path / "pred.jams"
    _write_jams(ref, [(0.0, 1.0, "intro"), (1.0, 2.0, "verse")])
    _write_jams(pred, [(0.0, 1.0, "intro"), (1.0, 2.0, "verse")])

    scores = evaluate_pair(ref, pred)

    assert scores["F-measure@0.5"] == 1.0
    assert scores["F-measure@3.0"] == 1.0


def test_format_evaluation_summary():
    evaluation = summarize_evaluations(
        [
            TrackEvaluation(
                track_id="track",
                dataset="dataset",
                prediction_path="pred.jams",
                scores={
                    "Precision@0.5": 1.0,
                    "Recall@0.5": 1.0,
                    "F-measure@0.5": 1.0,
                    "Precision@3.0": 1.0,
                    "Recall@3.0": 1.0,
                    "F-measure@3.0": 1.0,
                    "Pairwise F-measure": 1.0,
                    "NCE F-measure": 1.0,
                },
            )
        ]
    )

    formatted = format_evaluation(evaluation)

    assert "Tracks: 1" in formatted
    assert "F-measure@0.5: 1.0000" in formatted


def test_evaluate_prediction_manifest_reports_frame_accuracy(tmp_path):
    ref = tmp_path / "ref.jams"
    _write_jams(ref, [(0.0, 1.5, "intro"), (1.5, 2.0, "verse")])
    track = Track(
        track_id="track",
        audio_path=tmp_path / "audio.wav",
        jams_path=ref,
        dataset="dataset",
    )
    manifest = tmp_path / "manifest.jsonl"
    save_manifest(manifest, [track])

    _write_embedding_fixture(
        root=tmp_path,
        labels=["intro", "verse"],
        beats=[0.0, 0.5, 1.0, 1.5],
        duration=2.0,
    )
    predictions_root = tmp_path / "predictions"
    pred_dir = predictions_root / "dataset"
    pred_dir.mkdir(parents=True)
    _write_jams(pred_dir / "track.jams", [(0.0, 1.5, "intro"), (1.5, 2.0, "verse")])
    with (pred_dir / "track.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "labels": ["intro", "verse"],
                "frame_label_indices": [0, 1, 1, 1],
            },
            handle,
        )

    evaluation = evaluate_prediction_manifest(
        reference_manifest=manifest,
        predictions_root=predictions_root,
        audio_embedding_root=tmp_path / "audio_embeddings",
        audio_encoder="placeholder",
        text_embedding_root=tmp_path / "text_embeddings",
        text_encoder="placeholder",
        candidate_label_strategy="track_labels",
    )

    assert evaluation["summary"]["Acc"]["mean"] == 0.5
    assert np.isclose(evaluation["summary"]["Balanced Acc"]["mean"], (1.0 / 3.0 + 1.0) / 2.0)


def _write_jams(path, segments):
    jam = jams.JAMS()
    jam.file_metadata.duration = segments[-1][1]
    annotation = jams.Annotation(namespace="segment_open")
    for start, end, label in segments:
        annotation.append(time=start, duration=end - start, value=label)
    jam.annotations.append(annotation)
    jam.save(str(path))


def _write_embedding_fixture(root, labels, beats, duration):
    audio_dir = root / "audio_embeddings" / "placeholder" / "dataset" / "track"
    audio_dir.mkdir(parents=True)
    np.save(audio_dir / "beat_sync.npy", np.zeros((len(beats), 4), dtype=np.float32))
    np.save(audio_dir / "beats.npy", np.asarray(beats, dtype=np.float32))
    with (audio_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump({"outputs": {"duration": duration}}, handle)

    text_dir = root / "text_embeddings" / "placeholder" / "dataset"
    text_dir.mkdir(parents=True)
    with (text_dir / "labels.json").open("w", encoding="utf-8") as handle:
        json.dump({"labels": labels}, handle)
    np.save(text_dir / "embeddings.npy", np.zeros((len(labels), 3), dtype=np.float32))
