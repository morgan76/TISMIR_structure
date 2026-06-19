from tismir.data.schemas import Section
from tismir.data.validation import format_summary, summarize_validation, TrackValidationResult


def test_summarize_validation_counts_valid_labels():
    results = [
        TrackValidationResult(
            track_id="1",
            dataset="test",
            audio_path="audio.wav",
            jams_path="audio.jams",
            valid=True,
            num_sections=2,
            duration=10.0,
            labels=("intro", "verse"),
        ),
        TrackValidationResult(
            track_id="2",
            dataset="test",
            audio_path="missing.wav",
            jams_path="missing.jams",
            valid=False,
            errors=("missing",),
        ),
    ]

    summary = summarize_validation(results)

    assert summary.num_tracks == 2
    assert summary.num_valid_tracks == 1
    assert summary.num_invalid_tracks == 1
    assert summary.label_counts == {"intro": 1, "verse": 1}
    assert "Tracks: 1/2 valid" in format_summary(summary)
