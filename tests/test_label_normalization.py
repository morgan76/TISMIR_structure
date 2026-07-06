import pytest

from tismir.preprocessing.label_normalization import normalize_label, normalize_labels

SALAMI = {"name": "salami", "normalize_whitespace": True}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # plain vocabulary is lowercased
        ("Verse", "verse"),
        ("Chorus", "chorus"),
        ("Silence", "silence"),
        ("silence", "silence"),
        # underscores and hyphens become spaces
        ("no_function", "no function"),
        ("Main_Theme", "main theme"),
        ("Secondary_Theme", "secondary theme"),
        ("Fade-out", "fade out"),
        ("Pre-Chorus", "pre chorus"),
        ("post-chorus", "post chorus"),
        ("count-in", "count in"),
        ("stage_sounds", "stage sounds"),
        ("variation_2", "variation 2"),
        # SALAMI-specific notation aliases
        ("&pause", "pause"),
        ("w/dialog", "with dialog"),
        # semantically distinct labels are NOT merged
        ("Head", "head"),
        ("Theme", "theme"),
        ("da_capo", "da capo"),
    ],
)
def test_salami_normalization(raw, expected):
    assert normalize_label(raw, SALAMI) == expected


def test_salami_keeps_theme_variants_distinct():
    assert normalize_labels(["Head", "Theme", "Main_Theme"], SALAMI) == [
        "head",
        "theme",
        "main theme",
    ]


def test_salami_respects_overrides():
    config = {"name": "salami", "overrides": {"nofunction": "unlabeled section"}}
    assert normalize_label("no_function", config) == "unlabeled section"
