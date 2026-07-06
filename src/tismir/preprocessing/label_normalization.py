from __future__ import annotations

import re
from typing import Any


HARMONIX_ALIASES = {
    "altchorus": "alternate chorus",
    "bigoutro": "big outro",
    "chorushalf": "chorus half",
    "chorusinst": "instrumental chorus",
    "drumroll": "drum roll",
    "fadein": "fade in",
    "fadeout": "fade out",
    "fastchorus": "fast chorus",
    "gtr": "guitar",
    "gtrbreak": "guitar break",
    "guitarsolo": "guitar solo",
    "inst": "instrumental",
    "instbridge": "instrumental bridge",
    "instchorus": "instrumental chorus",
    "instintro": "instrumental intro",
    "instrumentalverse": "instrumental verse",
    "introverse": "intro verse",
    "mainriff": "main riff",
    "oddriff": "odd riff",
    "postchorus": "post chorus",
    "postverse": "post verse",
    "ppostchorus": "post post chorus",
    "prechorus": "pre chorus",
    "preverse": "pre verse",
    "quietchorus": "quiet chorus",
    "rhythmlessintro": "rhythmless intro",
    "slowverse": "slow verse",
    "transtiion": "transition",
    "verseslow": "slow verse",
    "verseinst": "instrumental verse",
    "vocalintro": "vocal intro",
    "vocaloutro": "vocal outro",
}


# AIST RWC-MDB-P-2001 CHORUS labels are already readable words; the trailing
# capital letters ("verse A", "chorus B") mark musically distinct variants of a
# section type and are kept in the prompt text. Only `nothing` (non-section
# regions) is rephrased for the text encoder — the raw target label stays
# `nothing`, which the training pipeline already treats as silence-like.
RWC_ALIASES = {
    "nothing": "silence",
}


# Prompt-text aliases for SALAMI function labels that the generic rules do not
# render as natural text. Keys are canonical keys (see _canonical_key). This is
# deliberately minimal: it only rewrites notation, never merges semantically
# distinct labels (Head / Theme / Main_Theme stay distinct). Casing and
# underscore/hyphen variants ("silence"/"Silence", "Fade-out", "no_function",
# "Main_Theme") are handled by the generic lowercase/despace rules.
# See data/datasets/SALAMI_LABELS.md for the full mapping table.
SALAMI_ALIASES = {
    "&pause": "pause",  # SALAMI guide notation for a musical pause
    "w/dialog": "with dialog",
}


def normalize_label(label: str, config: dict[str, Any] | None = None) -> str:
    """Return text-facing label form while preserving the raw target label elsewhere."""

    config = {} if config is None else dict(config)
    name = str(config.get("name", "none")).lower()
    overrides = dict(config.get("overrides", {}))
    if name == "none":
        normalized = label
    elif name == "harmonix":
        normalized = _normalize_harmonix_label(label, overrides=overrides)
    elif name == "salami":
        normalized = _normalize_salami_label(label, overrides=overrides)
    elif name == "generic":
        normalized = _normalize_generic_label(label, overrides=overrides)
    elif name == "rwc":
        normalized = _normalize_rwc_label(label, overrides=overrides)
    else:
        raise ValueError(f"Unknown label normalization preset: {name}")

    if config.get("normalize_whitespace", True):
        normalized = " ".join(normalized.split())
    return normalized


def normalize_labels(labels: list[str], config: dict[str, Any] | None = None) -> list[str]:
    return [normalize_label(label, config=config) for label in labels]


def _normalize_rwc_label(label: str, overrides: dict[str, str]) -> str:
    key = " ".join(str(label).split()).lower()
    if key in overrides:
        return overrides[key]
    if key in RWC_ALIASES:
        return RWC_ALIASES[key]
    return str(label)


def _normalize_harmonix_label(label: str, overrides: dict[str, str]) -> str:
    key = _canonical_key(label)
    if key in overrides:
        return overrides[key]

    stem, suffix = _split_suffix(key)
    if stem in overrides:
        return _join_suffix(overrides[stem], suffix)
    if key in HARMONIX_ALIASES:
        return HARMONIX_ALIASES[key]
    if stem in HARMONIX_ALIASES:
        return _join_suffix(HARMONIX_ALIASES[stem], suffix)
    return _normalize_generic_label(label, overrides=overrides)


def _normalize_salami_label(label: str, overrides: dict[str, str]) -> str:
    key = _canonical_key(label)
    if key in overrides:
        return overrides[key]
    if key in SALAMI_ALIASES:
        return SALAMI_ALIASES[key]
    return _normalize_generic_label(label, overrides=overrides)


def _normalize_generic_label(label: str, overrides: dict[str, str]) -> str:
    key = _canonical_key(label)
    if key in overrides:
        return overrides[key]

    label = re.sub(r"[_\-]+", " ", label.strip().lower())
    label = re.sub(r"(?<=[a-z])(?=\d)", " ", label)
    label = re.sub(r"(?<=\d)(?=[a-z])", " ", label)
    return label


def _canonical_key(label: str) -> str:
    return re.sub(r"[\s_\-]+", "", label.strip().lower())


def _split_suffix(label: str) -> tuple[str, str]:
    match = re.fullmatch(r"([a-z]+)([0-9]+[a-z]?)", label)
    if match is None:
        return label, ""
    return match.group(1), match.group(2)


def _join_suffix(label: str, suffix: str) -> str:
    if not suffix:
        return label
    suffix = re.sub(r"(?<=\d)(?=[a-z])", " ", suffix)
    return f"{label} {suffix}"
