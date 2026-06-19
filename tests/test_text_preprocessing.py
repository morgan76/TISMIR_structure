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
    )

    assert len(results) == 1
    output_dir = Path(results[0].output_dir)
    assert np.load(output_dir / "embeddings.npy").shape == (2, 8)

    labels = json.loads((output_dir / "labels.json").read_text(encoding="utf-8"))
    assert labels["labels"] == ["intro", "verse A"]
    assert labels["prompts"] == ["music section: intro", "music section: verse A"]


def _write_jams(path: Path) -> None:
    jam = jams.JAMS()
    jam.file_metadata.duration = 3.0
    annotation = jams.Annotation(namespace="segment_open")
    annotation.append(time=0.0, duration=1.0, value="intro")
    annotation.append(time=1.0, duration=1.0, value="verse A")
    annotation.append(time=2.0, duration=1.0, value="verse A")
    jam.annotations.append(annotation)
    jam.save(str(path))
