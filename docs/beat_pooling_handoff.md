# Beat-Pooling Experiments — Handoff

Resume doc for the `experiment/beat-pooling` branch. Full design rationale lives in the approved plan
at `~/.claude/plans/get-things-started-on-greedy-music.md`; this file is the concrete, current state
plus exactly what remains.

## TL;DR of the goal

Add alternative **beat-pooling methods** for MERT (today only `mean` is supported) and benchmark them
on **Harmonix** (HF `m-a-p/harmonixset_bigvgan`) and **SALAMI**. Two tracks:
- **A1** (parameter-free, cached in `beat_sync.npy`) — DONE at the core level.
- **A2** (learnable in-model attention pool over dense frames) — NOT STARTED (phase 2).

Chosen methods: `max`, `energy_weighted`, `multi_stat` (mean+max+std), `first`, `last` (plus existing `mean`).

---

## Environment

- **Interpreter/venv:** `/Users/chrisick/Repos/TISMIR_structure/.venv/bin/python` (Python 3.10.19, from
  pyenv `3.10.19`). Created this session; `.venv/` is gitignored. **No system/conda env had the project
  deps** — always use this venv.
- Installed: light deps (`numpy`, `PyYAML`, `pytest`, `jams` 0.3.5, editable `tismir`) ✅.
- **Heavy ML deps** installed ✅ via `pip install -e '.[hf-audio,beat,text]'`: `torch` 2.13.0 (MPS
  available), `transformers` 5.14.1, `librosa` 0.11, `soundfile`, `soxr`, `sentence-transformers` 5.6,
  `beat-this` 1.1.0. Verified importable.
- **madmom is NOT installed** and is not in `pyproject.toml`. The existing SALAMI/Harmonix preprocessing
  configs use the `madmom` beat tracker. Either install madmom per README (`README.md:137`, from GitHub)
  OR switch the new pooling configs to the `beat_this` tracker (installed and verified). Recommended:
  use `beat_this` to avoid the madmom build.

## Datasets on disk (all under `data/raw/`, gitignored)

- **SALAMI** — `data/raw/salami/` (unzipped from the 5 `data/drive-download-*.zip` parts):
  - `audio/<id>/audio.mp3` — 1447 track dirs
  - `jams/<id>.jams` — 1359 annotation files (some audio has no jams; skip those)
  - `ddmal_metadata.csv`
  - Layout is **mp3 + nested `audio/<id>/` + `jams/`**, NOT the `.wav`/`SALAMI_`/`references/` layout the
    current `scripts/create_salami_manifest.py` assumes. Script needs updating (see remaining work).
- **Harmonix** — audio `data/raw/harmonix/harmonixset_bigvgan/tracks/<data_id>.wav` (912 wavs), and raw
  download in `data/raw/harmonix_bigvgan/` (the 8 GB zip + `harmonixset.corrected.20250821.jsonl`).
  - JSONL record shape: `{"data_id": "0001_12step", "dataset_type": "harmonixset_8class",
    "msa_info": [[time, label], ...], "split": "train|val|test"}`.
  - `msa_info` = **onset time + label** pairs; interval `i` is `[t_i, t_{i+1})` with `label_i`; the final
    entry is `[t_last, "end"]` marking track duration (NOT a real section — use it as the closing boundary).
  - Splits: train 512 / val 200 / test 200. Labels (9): intro, verse, chorus, outro, silence, bridge,
    pre-chorus, inst, (+ `end` boundary marker). Clean labels — no compact-code normalization needed.
  - `data_id` matches the wav stem exactly.

## What is DONE (committed to working tree, not yet `git commit`-ed)

1. **Branch** `experiment/beat-pooling` created off `main`.
2. **Pooling family** — `src/tismir/preprocessing/beat_sync.py` rewritten:
   - New `pool_frames_to_intervals(embeddings, times, intervals, method, empty, *, temperature, stats)`
     dispatcher. Methods: `mean`, `max`, `energy_weighted` (softmax over per-frame L2 norm, `temperature`
     param), `multi_stat` (concat of `stats`, default `[mean,max,std]` → dim × n_stats), `first`, `last`.
   - Shared interval-membership (`[start,end)`) + empty-interval policy (`nearest`/`zeros`/`raise`), the
     empty branch respects the multi_stat output dim.
   - `mean_pool_to_intervals` kept as a thin wrapper (backward compat). Constants `POOLING_METHODS`,
     `MULTI_STAT_CHOICES` exported.
3. **Wiring** — `src/tismir/preprocessing/audio.py`:
   - Imports the dispatcher; replaced the `method != "mean"` hard-raise with validation against
     `POOLING_METHODS`; threads `empty`/`temperature`/`stats`; records them in `metadata.json`.
   - `scripts/preprocess_audio.py` already forwards `config["pooling"]` — unchanged.
4. **Harmonix download script** — `scripts/download_harmonix.py` (uses `huggingface_hub.snapshot_download`).
5. **Verified:** all 6 methods produce correct outputs (multi_stat → 6-dim on a 2-dim toy); existing
   `pytest tests/test_beat_sync.py tests/test_audio_preprocessing.py` → 4 passed.

**Key compatibility fact:** the model infers `audio_dim` from the saved array
(`training/data.py:251`, `models/factory.py:51`), so `multi_stat`'s larger dim needs **no model change**.

## Progress update (2026-07-24 session)

Decisions taken: **madmom** beat tracker (installed from source into `.venv`);
**Phase 1 + A2** both implemented; **full** MERT preprocessing kicked off.

Done this session:
- **A1 tests** — `tests/test_beat_sync.py` (per-method values, temperature, empty
  policies, multi_stat dims) + a multi_stat case in `tests/test_audio_preprocessing.py`.
- **A1 configs** — 6 per-method configs (+ `_salami` variants) at
  `configs/preprocessing/audio_mert_madmom_pool_{mean,max,energy,multistat,first,last}.yaml`,
  each with its own `output_root` (`data/embeddings/audio_pool_<method>`), plus
  `audio_mert_madmom_keep_dense.yaml` for A2.
- **Encode-once multipool** — `src/tismir/preprocessing/audio.py` refactored to
  encode a track once and write all pooling variants; `scripts/preprocess_audio_multipool.py`
  drives it (≈1 encode instead of 6).
- **B1** — `scripts/create_salami_manifest.py` gained `--layout nested-mp3`;
  `data/manifests/salami.local.jsonl` (1359 tracks) written + validated.
- **B2** — `scripts/convert_harmonix_jsonl_to_jams.py` → `segment_open` JAMS +
  `data/manifests/harmonix_bigvgan{,_train,_val,_test}.local.jsonl` (512/200/200);
  all 912 validated.
- **C** — harmonix text embeddings computed (`data/embeddings/text/...`); 6
  per-method self-attention train configs
  (`configs/train/harmonix_bigvgan_self_attention_pool_*.yaml`); README
  "Beat-Pooling Experiments" section.
- **A2** — `SegmentAttentionPool` (scatter-based segment softmax, O(F) memory) in
  `models/adapters.py`; dense-path loading + `frame_segment_ids` + collate in
  `training/data.py`; frame threading through `training/loop.py`; `beat_pool`
  wiring in `models/factory.py`; train config
  `configs/train/harmonix_bigvgan_self_attention_beat_pool_attention.yaml`. Tests
  in `tests/test_adapter_model.py` + `tests/test_training_data.py`.
- Full test suite: **130 passed**.

Environment gotcha fixed: this pyenv 3.10.19 was built without `_lzma`
(librosa→pooch import crashed). A scoped shim lives at
`.venv/lib/python3.10/site-packages/_lzma.py` (never actually used — audio loads
via soundfile). Rebuilding Python against liblzma would let you delete it, but a
forced `pyenv install` would destroy the sibling `py310` env, so the shim was the
safe choice.

Still open:
- **Full preprocessing run in progress** in the background (`logs/preprocess_harmonix.log`
  then `logs/preprocess_salami.log`, `--skip-existing`, resumable). Harmonix
  first (~912), then SALAMI (~1359).
- **A2 dense embeddings NOT generated** — needs a separate `keep_dense` run
  (`audio_mert_madmom_keep_dense.yaml`), which is large on disk (~55 MB/track).
- **SALAMI text + train configs** not created (only Harmonix). SALAMI text
  configs already exist under `configs/preprocessing/text_salami_*`.
- **Nothing committed** yet on `experiment/beat-pooling`.

## What REMAINS

### A1 finish (pooling — small, do first)
- **Tests:** extend `tests/test_beat_sync.py` with per-method value checks, empty-interval policy per
  method, and multi_stat dim; add a case in `tests/test_audio_preprocessing.py` proving a non-`mean`
  method runs and writes correct `metadata["pooling"]` (method/stats/temperature). Use the existing
  `placeholder` audio encoder + `uniform` beat tracker (no torch needed) as in the current test.
- **Preprocessing configs** (mirror `configs/preprocessing/audio_mert_madmom.yaml`), one per method, each
  with a distinct `output_root` so cached arrays don't collide:
  `audio_mert_madmom_pool_mean.yaml` → `data/embeddings/audio_pool_mean`, and `..._max`, `..._energy`
  (`pooling.temperature`), `..._multistat` (`pooling.stats: [mean, max, std]`), `..._first`, `..._last`.
  Make a SALAMI-specific variant set too (or override `--manifest`/`--output-root` on the CLI).

### B1 — SALAMI manifest
- Update `scripts/create_salami_manifest.py` (or add a `--layout nested-mp3` mode) for the actual layout:
  glob `data/raw/salami/jams/*.jams`; for each `<id>` include it only if
  `data/raw/salami/audio/<id>/audio.mp3` exists; emit `Track(track_id=<id>,
  audio_path=.../audio/<id>/audio.mp3, jams_path=.../jams/<id>.jams, dataset="salami")`.
  Output `data/manifests/salami.local.jsonl`. Then `scripts/validate_dataset.py --manifest ...`.
- SALAMI namespaces (`segment_salami_function` / `_lower`) + policies already exist; the SALAMI text +
  audio configs already exist (`configs/preprocessing/*salami*`).

### B2 — Harmonix converter + manifest
- **New** `scripts/convert_harmonix_jsonl_to_jams.py`: read
  `data/raw/harmonix_bigvgan/harmonixset.corrected.20250821.jsonl`; per record build `Section`s from
  consecutive `msa_info` onsets (`[t_i, t_{i+1})` label `label_i`; final `end` → duration; drop/omit the
  `end` label as a section). Write `data/raw/harmonix/jams/<data_id>.jams` in namespace **`segment_open`**
  using the `jams` library (build `jams.JAMS`, add a `segment_open` annotation with `(time, duration,
  value)` observations, set `file_metadata.duration`). Carry `split` (e.g. into a sidecar or the manifest).
- **Manifest with official splits:** build `data/manifests/harmonix_bigvgan.local.jsonl`
  (`audio_path=data/raw/harmonix/harmonixset_bigvgan/tracks/<data_id>.wav`,
  `jams_path=data/raw/harmonix/jams/<data_id>.wav→.jams`, `dataset="harmonix"`, `split` from JSONL) and
  emit `harmonix_bigvgan_{train,val,test}.local.jsonl` directly from the embedded split (no random
  `split_manifest.py`). Reuse `tismir.data.manifest.save_manifest` + `Track`.

### C — train configs, text embeddings, docs
- Text embeddings: `scripts/preprocess_text.py` with existing Harmonix/SALAMI text configs (pooling-independent).
- Train configs: clone a self-attention adapter config (e.g.
  `configs/train/harmonix_split_fact_self_attention_32d_4blocks_base_occurrences.yaml`), one per pooling
  method, differing only in `audio_embedding_root`. NOTE: `configs/train/` is gitignored
  (`.gitignore` `/configs/train`), so those files won't be tracked — keep an untracked set, or document.
- README: add a "Beat-Pooling Experiments" section + the Harmonix-bigvgan/SALAMI-zip setup steps.

### A2 — learnable in-model attention pool (phase 2, larger)
- Preprocess with `keep_dense: true` (new `configs/preprocessing/audio_mert_madmom_keep_dense.yaml`) →
  saves `dense.npy`/`dense_times.npy`.
- `src/tismir/training/data.py`: support `audio_embedding_key: "dense"` — load dense frames + times, reuse
  `build_beat_intervals` (already at `data.py:134`) to make a per-frame beat segment id, carry `frames` +
  `segment_ids` through `TrainingExample` + `collate_training_examples` (pad frames, mask, segment index).
  Guard behind config so the default `beat_sync` path is untouched.
- `src/tismir/models/adapters.py`: `SegmentAttentionPool` (frames `[B,F,mert_dim]` + segment ids →
  `[B,num_beats,mert_dim]` via segment-masked softmax attention with a learned query), inserted before
  `audio_projection` (`adapters.py:212`); select via `model.audio.beat_pool` in `models/factory.py`.
- Tests in `tests/test_adapter_model.py` + a dense-path data test.

## Verification commands

```bash
PY=.venv/bin/python
# unit
$PY -m pytest tests/test_beat_sync.py tests/test_audio_preprocessing.py -q
# preprocess smoke (needs heavy deps + a beat tracker installed): a few SALAMI tracks
$PY scripts/preprocess_audio.py --config configs/preprocessing/audio_mert_madmom_pool_multistat.yaml \
   --manifest data/manifests/salami.local.jsonl --limit 3
# confirm multi_stat beat_sync dim == mert_dim * n_stats, others == mert_dim
```

## Open decisions for the resumer
- **Beat tracker for preprocessing:** install `madmom` (matches existing configs) or switch configs to
  `beat_this` (already installed). madmom on py3.10 usually works; needs a from-source install.
- Whether to `git commit` the current working-tree changes (beat_sync.py, audio.py, download_harmonix.py,
  this doc) before proceeding — nothing has been committed yet on this branch.
