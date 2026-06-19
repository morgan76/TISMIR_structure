from tismir.encoders.audio import audio_encoders


def test_builtin_audio_encoders_are_registered():
    names = audio_encoders.names()

    assert "placeholder" in names
    assert "mert" in names
