from tismir.encoders.text import text_encoders


def test_builtin_text_encoders_are_registered():
    names = text_encoders.names()

    assert "placeholder" in names
    assert "sentence_transformers" in names
