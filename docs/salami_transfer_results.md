# Zero-shot Cross-dataset Transfer: Harmonix → SALAMI

Harmonix-trained checkpoints evaluated on **all 1359 SALAMI tracks** with no SALAMI
training (zero-shot). The model is open-vocabulary, so transfer only requires
SALAMI audio embeddings (same MERT + pooling) and SALAMI candidate-label text
embeddings.

## Setup
- **Checkpoints:** best-F@3.0 checkpoints from the Harmonix pooling sweep.
- **SALAMI audio:** `data/embeddings/audio_pool_multistat` (multi_stat) and
  `data/embeddings/audio_dense` (attention/A2 dense path).
- **SALAMI text:** `text_salami_function_merge_bare` — **bare** prompts (matching
  the Harmonix training prompt style) over the `salami_function_merge` label set.
- **Namespace:** `segment_salami_function`; **candidate labels:** `dataset_labels`
  (open-set — model chooses among the full SALAMI function vocabulary, not per-track
  oracle labels).

## Results (SALAMI, all 1359 tracks)

| model | F@0.5 | F@3.0 | Pairwise F | NCE F |
|---|---|---|---|---|
| multi_stat | 0.134 | 0.284 | 0.445 | 0.511 |
| attention (A2) | **0.144** | **0.322** | **0.446** | **0.525** |

For reference, in-domain Harmonix val: multi_stat F@3.0 0.596, A2 F@3.0 0.610.

## Takeaways
- **The A2 attention pool's advantage transfers:** it beats multi_stat cross-dataset
  on every metric (F@3.0 0.322 vs 0.284), consistent with the in-domain ranking.
- **Large domain gap, as expected for zero-shot:** F@3.0 drops ~0.60 → ~0.30 moving
  Harmonix→SALAMI. Drivers: SALAMI is more diverse (classical/jazz/live, not just
  pop), a shifted/larger label vocabulary, and no SALAMI supervision.
- **Boundary transfer is the softer spot** (F@0.5 ~0.14) — fine-grained boundaries
  are dataset-specific; the coarser F@3.0 and the labelling metrics (Pairwise/NCE)
  hold up better.

## Caveats / next steps
- Open-set `dataset_labels` was used because the bare text vocab (40 labels) did not
  cover every reference label (e.g. `exposition`) under `track_labels` — a text
  annotation-selection vs inference discrepancy worth reconciling for a track-oracle
  number.
- This is **mode 1** (prompt-consistent zero-shot). **Mode 2** — a shared
  cross-dataset label ontology with enriched descriptions, regenerated for *both*
  train and eval, then retraining Harmonix — is the path to close the gap and is
  where label-set enrichment pays off.
