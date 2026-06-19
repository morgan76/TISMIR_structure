from pathlib import Path

from tismir.data.manifest import load_manifest, save_manifest
from tismir.data.schemas import Track


def test_manifest_round_trip(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    tracks = [
        Track(
            track_id="track",
            audio_path=Path("audio.wav"),
            jams_path=Path("audio.jams"),
            dataset="dataset",
            split="train",
        )
    ]

    save_manifest(manifest_path, tracks)
    loaded = load_manifest(manifest_path)

    assert loaded[0].track_id == "track"
    assert loaded[0].audio_path == tmp_path / "audio.wav"
    assert loaded[0].jams_path == tmp_path / "audio.jams"
