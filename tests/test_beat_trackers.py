from tismir.encoders.beats import beat_trackers


def test_builtin_beat_trackers_are_registered():
    names = beat_trackers.names()

    assert "uniform" in names
    assert "beat_this" in names
    assert "madmom" in names
