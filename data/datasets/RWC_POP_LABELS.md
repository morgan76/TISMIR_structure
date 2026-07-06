# RWC Pop (AIST CHORUS) label space

Source: `AIST.RWC-MDB-P-2001.CHORUS/RM-P###.CHORUS.TXT`, one file per track,
rows `start<TAB>end<TAB>"label"`, times in 10 ms units (÷100 → seconds).
Segments tile the annotated span contiguously per track.

## Vocabulary (counts over all 100 tracks)

| label | count | notes |
|---|---|---|
| chorus A | 340 | letter = musically distinct variant, **not** occurrence number |
| verse A | 237 | |
| verse B | 201 | |
| chorus B | 168 | |
| bridge A | 159 | |
| intro | 102 | |
| ending | 98 | RWC's word for outro; kept distinct, not merged |
| pre-chorus | 92 | |
| verse C | 86 | |
| bridge B | 84 | |
| post-chorus | 71 | |
| bridge C | 38 | |
| chorus C | 26 | |
| bridge D | 5 | |
| nothing | 3 | non-section region (silence-like) |
| chorus D | 1 | |

16 distinct labels; base types: intro, verse, chorus, bridge, pre-chorus,
post-chorus, ending, nothing.

## Adaptation decisions

- **Targets**: raw labels verbatim in JAMS (`segment_open`). The variant
  letters carry real information (different musical material) and RWC's
  annotation style differs from Harmonix's occurrence-free 8-class set —
  exactly the cross-dataset label-space variation the project studies.
- **Prompts** (`label_normalization: name: rwc`): pass-through except
  `nothing` → "silence" (better text-encoder semantics; the raw target stays
  `nothing`, which `tismir.data.annotations` already treats as silence-like
  for synthetic `__T_MIN`/`__T_MAX` boundaries).
- **Not done (open experiments)**: merging `ending`→`outro`; stripping
  variant letters (`chorus A/B` → `chorus`) — both are `overrides:`-level
  ablations, or can use `annotation_processing` policies at train time.
- Contrast with Harmonix corrected set: no occurrence enumeration is needed
  here; `enumerate_*` policies remain available for prompt ablations.

## Format note

Chorus rows may carry a 4th column such as `(-10)`: the AIST modulation
annotation (key shift in semitones relative to the first chorus). It is not
part of the label space and the converter ignores it.
