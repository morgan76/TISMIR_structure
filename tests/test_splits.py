from tismir.data.schemas import Track
from tismir.data.splits import split_tracks


def test_split_tracks_sets_split_fields(tmp_path):
    tracks = [
        Track(
            track_id=str(index),
            audio_path=tmp_path / f"{index}.wav",
            jams_path=tmp_path / f"{index}.jams",
            dataset="dataset",
        )
        for index in range(10)
    ]

    splits = split_tracks(tracks, train_ratio=0.8, val_ratio=0.2, seed=0)

    assert len(splits["train"]) == 8
    assert len(splits["val"]) == 2
    assert {track.split for track in splits["train"]} == {"train"}
    assert {track.split for track in splits["val"]} == {"val"}
    assert {track.track_id for split in splits.values() for track in split} == {
        str(index) for index in range(10)
    }
