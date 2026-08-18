# Master Experiment Table

Consolidated view of every experiment run so far, **normalized on Harmonix as the
common eval set**. "Harmonix val F@3.0" is the shared anchor column; each
experiment's native benchmark (SALAMI / RWC-Pop) is kept in a separate column so
nothing is lost. All Harmonix numbers are val-split, at the best-F@3.0 epoch.

Per-experiment detail lives in `beat_pooling_results.md`, `salami_transfer_results.md`,
`label_synonym_results.md`, and `model_variant_results.md`.

## Group 1 — Beat-pooling sweep (in-domain, Harmonix)

Fixed adapter model, varied pooling. This is the source of the shared anchor.

| Variant | Pooling | Harmonix val F@3.0 | F@0.5 | PairwiseF | NCE-F | Acc |
|---|---|---|---|---|---|---|
| attention (A2) | attention | **0.610** | 0.251 | **0.694** | **0.811** | **0.618** |
| multi_stat | multi_stat | 0.596 | 0.250 | 0.689 | 0.809 | 0.610 |
| max | max | 0.564 | 0.228 | 0.686 | 0.807 | 0.607 |
| mean (repo baseline) | mean | 0.557 | 0.239 | 0.683 | 0.807 | 0.613 |
| energy_weighted | energy_weighted | 0.549 | 0.230 | 0.675 | 0.803 | 0.609 |
| last | last | 0.491 | 0.209 | 0.659 | 0.790 | 0.561 |
| first | first | 0.504 | 0.223 | 0.648 | 0.786 | 0.516 |

## Group 2 — Cross-dataset transfer & augmentation

Same A2 checkpoints; Harmonix is in-domain, SALAMI is the zero-shot native target.

| Experiment | Variant | Harmonix val F@3.0 | Native: SALAMI F@3.0 | SALAMI PairwiseF | SALAMI NCE-F |
|---|---|---|---|---|---|
| Transfer (A2) | attention | 0.610 | 0.322 | 0.446 | 0.525 |
| Transfer | multi_stat | 0.596 | 0.284 | 0.445 | 0.511 |
| Synonym aug | A2 + core | 0.599 | 0.324 | 0.442 | 0.523 |
| Synonym aug | A2 + transfer | 0.598 | 0.326 | 0.435 | 0.524 |

Synonym augmentation: no meaningful SALAMI gain, small in-domain Harmonix cost — a
wash (see `label_synonym_results.md`).

## Group 3 — Model-architecture variants (Morgan)

Different architecture line, best decoder per model. **Not yet evaluated on
Harmonix**, so the common anchor is unavailable; native benchmark is RWC-Pop test.

| Variant | Best decoder | Harmonix val F@3.0 | Native: RWC-Pop F@3.0 | RWC-Pop F@0.5 | RWC-Pop NCE-F |
|---|---|---|---|---|---|
| projection_ce | argmax | n/a — not run | 0.469 | 0.094 | 0.610 |
| adapter_rope_ce | viterbi | n/a — not run | 0.442 | 0.116 | 0.637 |
| hybrid_ce | argmax | n/a — not run | 0.526 | 0.129 | 0.724 |
| hybrid_ce_boundary | boundary_viterbi | n/a — not run | 0.531 | 0.376 | 0.720 |
| hybrid_ce_link | viterbi | n/a — not run | 0.567 | 0.221 | 0.735 |
| hybrid_ce_link_boundary | boundary_viterbi | n/a — not run | **0.638** | **0.465** | **0.749** |

## Caveats on the "common eval set"

- **Group 3 has no Harmonix eval.** The architecture variants were run on RWC-Pop
  only; their anchor cells are `n/a — not run`, not fabricated. To truly unify all
  three groups on Harmonix, the five variants must be re-run/evaluated on the
  Harmonix val split.
- **Benchmarks differ across groups:** Harmonix val (Groups 1–2 anchor), SALAMI
  zero-shot (Group 2 native), RWC-Pop test (Group 3 native). F-numbers do not
  cross-compare directly across benchmarks — a 0.64 on RWC-Pop and a 0.61 on
  Harmonix are not the same difficulty.
- **Axes are complementary, not competing:** Groups 1–2 fix architecture and vary
  pooling; Group 3 fixes pooling and varies architecture + decoder.
- **Metric coverage varies:** Group 3 additionally reports Acc / Bal Acc; the
  pooling sweep reports Acc; the SALAMI eval reports the full mir_eval set.

## Next step to close the gap

Evaluate the Group 3 variants on Harmonix val (same split/decoder config as Group 1)
to fill the anchor column and make a single fully-normalized table possible.
