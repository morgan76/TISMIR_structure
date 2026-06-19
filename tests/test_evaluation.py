import jams

from tismir.evaluation import evaluate_pair, format_evaluation
from tismir.evaluation.segments import TrackEvaluation, summarize_evaluations


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


def _write_jams(path, segments):
    jam = jams.JAMS()
    jam.file_metadata.duration = segments[-1][1]
    annotation = jams.Annotation(namespace="segment_open")
    for start, end, label in segments:
        annotation.append(time=start, duration=end - start, value=label)
    jam.annotations.append(annotation)
    jam.save(str(path))
