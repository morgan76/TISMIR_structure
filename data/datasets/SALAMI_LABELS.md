# SALAMI label space and adaptation decisions

Source: SALAMI Data Set v2.0 (public annotations, CC0), local copy at
`/scratch/ick/music_structure/salami/`. 1359 annotated tracks under
`annotations/<song_id>/`. Converter: `scripts/convert_salami_to_jams.py`
(JAMS output at `/scratch/ick/music_structure/salami/jams/<id>.jams`).

## Annotation layers (functions vs letter hierarchy)

Each track has one or two free-form annotation files (`textfile1.txt` =
annotator 1, `textfile2.txt` = annotator 2). The official SALAMI parser
(`get_sections_sept_2012.rb` + `funct_vocab_dictionary.txt`) splits each into
three time-aligned layers under `parsed/`:

- `textfile{1,2}_functions.txt` — **function labels** (Verse, Chorus, Solo, ...).
  This is what we convert to `segment_open` JAMS. The parser has already
  applied the vocab dictionary: typo fixes (`Chrous` -> `Chorus`), synonym
  folding (`First theme` -> `Main_Theme`, `closing` -> `Outro`), and mapping of
  segments that carry only letter labels to `no_function`.
- `textfile{1,2}_uppercase.txt` — large-scale letter sections (A, B, A', ...).
- `textfile{1,2}_lowercase.txt` — small-scale letter phrases (a, b, a', ...).

The uppercase/lowercase letter layers are a **finer-granularity /
open-vocabulary-free label set** for the same boundaries — a future
experimental knob for the label-space variable of this project. They are not
converted yet; the converter only reads the functions layer
(`annotation.sandbox.level = "functions"` records this).

Coverage: 1348/1359 tracks have annotator-1 functions; 895 have annotator-2
functions; 11 tracks have only annotator 2 (ids 70, 75, 90, 240, 339, 349,
376, 390, 423, 1015, 1258); 884 tracks have both.

## File format and terminators

`parsed/*_functions.txt` lines are `time<TAB>label` (seconds). Every surveyed
file contains exactly one `End` marker whose timestamp is the track end.
Conventions adopted (mirroring the HarmonixSet converter):

- **`End`** is a duration terminator: it sets `file_metadata.duration` and is
  never emitted as a segment.
- **`Silence` / `silence`** are kept as regular segments (leading/trailing
  silence is real signal; the pipeline can map synthetic boundaries to
  silence-like labels). `Silence` is annotator-written; lowercase `silence`
  originates from the small-scale layer or the parser (see artifacts below).

## Artifacts found (and how they are handled)

1. **`End0.0` / `Silence0.0` are NOT in the data.** They appear only when
   concatenating parsed files (e.g. `cat */parsed/*_functions.txt`): the files
   have **no trailing newline**, so the last label of one file glues onto the
   first timestamp `0.0` of the next file. Per-file parsing shows the real
   labels are plain `End` / `Silence`. No repair is needed; the converter
   parses per file and per line.
2. **Duplicate-timestamp rows (690 across both annotators).** Exact-tie rows
   share a timestamp with the previous row (there are zero non-monotonic
   rows). Three provenances:
   - Raw lines carrying two co-occurring functions, split by the parser into
     two rows at the same time (top pairs: `Instrumental, Solo` x237;
     `Main_Theme, Theme` x32; `Fade-out, Outro` x33; `Chorus, Outro` x12;
     `Head, Verse` x11; `Secondary_Theme, Theme` x10, ...).
   - A **synthetic lowercase `silence`** inserted by the official parser at
     time 0.0 whenever a song "didn't begin with silence" (see
     `get_sections_sept_2012.rb` lines 106-108); it is ordered after the real
     label (`Intro -> silence` x83, `Intro -> Silence` x47, ...).
   - A zero-length trailing `Silence` at the same timestamp as `End`
     (99 rows, incl. 33 annotator-1 files ending `... End / Silence`).

   **Policy: keep the FIRST label of each tie group, drop the rest.** This
   keeps the annotator's first-listed (primary) function for dual-function
   segments, drops the parser-synthetic `silence`, and drops the zero-length
   trailing `Silence`. All 690 dropped rows are counted and reported by the
   converter (`rows_skipped_duplicate_timestamp`). Consequence: 11 raw labels
   that only ever occur as tie-secondaries do not appear in the JAMS (`out`,
   `guitar`, `third`, `first`, `female`, `w/dialog`, `organ`, `strings`,
   `piece_1`, `piece_2`, `bagpipes`).
3. **Casing/underscore variants** (`silence` vs `Silence`, `outro` vs `Outro`,
   `Break` vs `break`, `Secondary_theme` vs `Secondary_Theme`): preserved
   verbatim in JAMS (they are the training targets); folded only in prompt
   text by the `salami` normalization preset.
4. **Letter labels leaked into the functions layer** (`d/b`, `b/c'`,
   `a'/a''`): kept verbatim (8 occurrences total).

## Converter summary

`scripts/convert_salami_to_jams.py` wrote **1359 JAMS** (one per track):

- Annotator 1 primary; annotator 2 primary for the 11 tracks missing
  annotator 1 (`tracks_fallback_annotator2: 11`); where both exist (884
  tracks) annotator 2 is a second `segment_open` annotation in the same JAMS
  (`annotation.sandbox.annotator` = 1 or 2).
- `file_metadata.duration` = max of the annotators' `End` times; title/artist
  from `metadata/metadata.csv`.
- Source collection stored in `jam.sandbox.collection` (and each annotation's
  sandbox) for later audio-availability filtering:

  | collection | tracks |
  |---|---|
  | Codaich | 778 |
  | IA (Internet Archive) | 446 |
  | Isophonics | 48 |
  | RWC | 87 |

- Skips: 0 tracks skipped; 690 duplicate-timestamp rows dropped (see above).

## Function-label vocabulary (as stored in JAMS) and prompt normalization

71 distinct verbatim labels (82 in the raw functions files incl. `End` and
tie-only secondaries). Prompt text comes from the `salami` preset in
`src/tismir/preprocessing/label_normalization.py`: lowercase, underscores and
hyphens to spaces, digits split off, plus two notation aliases
(`&pause` -> `pause`, `w/dialog` -> `with dialog`). **No semantic merging**:
`Head`, `Theme`, `Main_Theme` stay distinct; `no_function` becomes the literal
`no function` (defensible: it is the parser's explicit marker for segments the
annotator labelled only with letters, i.e. "no function given", not silence
and not an unlabeled gap). Casing folds pure case variants
(`Silence`/`silence`, `Break`/`break`, `Secondary_Theme`/`Secondary_theme`,
`Outro`/`outro`) in prompt space only.

| raw label (JAMS value) | annot. 1 | annot. 2 | total | prompt text (`salami` preset) |
|---|---|---|---|---|
| `Verse` | 2809 | 1605 | 4414 | verse |
| `Chorus` | 2521 | 1438 | 3959 | chorus |
| `Silence` | 2258 | 1274 | 3532 | silence |
| `no_function` | 1965 | 1365 | 3330 | no function |
| `Solo` | 1577 | 814 | 2391 | solo |
| `Intro` | 1110 | 771 | 1881 | intro |
| `Outro` | 648 | 547 | 1195 | outro |
| `Interlude` | 704 | 282 | 986 | interlude |
| `Theme` | 418 | 459 | 877 | theme |
| `Transition` | 405 | 461 | 866 | transition |
| `Bridge` | 415 | 327 | 742 | bridge |
| `Instrumental` | 452 | 266 | 718 | instrumental |
| `Head` | 349 | 320 | 669 | head |
| `Pre-Chorus` | 284 | 110 | 394 | pre chorus |
| `Main_Theme` | 173 | 180 | 353 | main theme |
| `Coda` | 220 | 41 | 261 | coda |
| `Fade-out` | 149 | 86 | 235 | fade out |
| `Secondary_Theme` | 81 | 40 | 121 | secondary theme |
| `silence` | 32 | 72 | 104 | silence |
| `Pre-Verse` | 54 | 37 | 91 | pre verse |
| `post-chorus` | 51 | 17 | 68 | post chorus |
| `break` | 24 | 10 | 34 | break |
| `applause` | 29 | 0 | 29 | applause |
| `Development` | 16 | 5 | 21 | development |
| `voice` | 5 | 10 | 15 | voice |
| `outro` | 7 | 4 | 11 | outro |
| `stage_sounds` | 9 | 0 | 9 | stage sounds |
| `spoken` | 6 | 1 | 7 | spoken |
| `groove` | 0 | 6 | 6 | groove |
| `variation` | 4 | 1 | 5 | variation |
| `Secondary_theme` | 0 | 5 | 5 | secondary theme |
| `male` | 0 | 5 | 5 | male |
| `Recap` | 4 | 0 | 4 | recap |
| `spoken_voice` | 4 | 0 | 4 | spoken voice |
| `tag` | 0 | 4 | 4 | tag |
| `ostinato` | 0 | 4 | 4 | ostinato |
| `dialog` | 0 | 4 | 4 | dialog |
| `d/b` | 0 | 3 | 3 | d/b |
| `crowd_sounds` | 2 | 0 | 2 | crowd sounds |
| `banjo` | 2 | 0 | 2 | banjo |
| `stage_speaking` | 2 | 0 | 2 | stage speaking |
| `variation_2` | 2 | 0 | 2 | variation 2 |
| `vocals` | 1 | 1 | 2 | vocals |
| `Exposition` | 1 | 1 | 2 | exposition |
| `call_and_response` | 2 | 0 | 2 | call and response |
| `response` | 0 | 2 | 2 | response |
| `post-verse` | 0 | 2 | 2 | post verse |
| `contrasting_middle` | 0 | 2 | 2 | contrasting middle |
| `b/c'` | 0 | 2 | 2 | b/c' |
| `post-cadential` | 0 | 2 | 2 | post cadential |
| `violin` | 0 | 2 | 2 | violin |
| `hammond` | 1 | 0 | 1 | hammond |
| `&pause` | 1 | 0 | 1 | pause |
| `count-in` | 1 | 0 | 1 | count in |
| `variation_1` | 1 | 0 | 1 | variation 1 |
| `steel` | 1 | 0 | 1 | steel |
| `vocalizations` | 1 | 0 | 1 | vocalizations |
| `muted` | 1 | 0 | 1 | muted |
| `backing` | 1 | 0 | 1 | backing |
| `harpsichord` | 1 | 0 | 1 | harpsichord |
| `pick-up` | 1 | 0 | 1 | pick up |
| `bass` | 1 | 0 | 1 | bass |
| `build` | 1 | 0 | 1 | build |
| `da_capo` | 1 | 0 | 1 | da capo |
| `gypsy` | 1 | 0 | 1 | gypsy |
| `Break` | 0 | 1 | 1 | break |
| `a'/a''` | 0 | 1 | 1 | a'/a'' |
| `piano` | 0 | 1 | 1 | piano |
| `ritornello` | 0 | 1 | 1 | ritornello |
| `Code` | 0 | 1 | 1 | code |
| `trumpet` | 0 | 1 | 1 | trumpet |

Notes on tail labels: instrument names (`banjo`, `hammond`, `violin`, ...) and
performance descriptors (`male`, `groove`, `tag`, ...) are annotator free-text
that survived the dictionary; they are rare (<= 6 occurrences each), preserved
verbatim, and only case/space-normalized for prompts. `Code` is a SALAMI
dictionary artifact (its own typo for `Coda`, see `codetta -> Code` in
`funct_vocab_dictionary.txt`); it is preserved verbatim per the
no-semantic-merge rule.

## Audio sourcing status (2026-07-06)

- **RWC (87)**: fully sourced. Zenodo record 18656623 provides all four RWC
  zips; SALAMI (disc, track) joined to RWCID via rwc-annotations metadata.csv
  (all 87 duration-validated < 3 s against the SALAMI annotation).
- **Internet Archive (290/476 fetched, 272 with annotations)**: stored 2011
  URLs are dead; `scripts/fetch_salami_internetarchive.py` recovers files via
  the archive.org metadata API (61% yield; coverage CSV written next to the
  audio). Remainder: items deleted or track files renamed beyond recognition.
- **Codaich (778) / Isophonics (48)**: no sourceable audio.
- Combined manifest: `scripts/build_salami_manifest.py` -> 359 usable tracks.
