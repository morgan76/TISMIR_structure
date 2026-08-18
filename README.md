# TISMIR Structure

Text-conditioned audio-based music structure analysis.

This project explores music structure segmentation as open-vocabulary, label-set-conditioned prediction. Given an audio track and an arbitrary set of text labels, the model should produce a segmentation whose granularity follows the supplied label set.

## Design

The initial codebase is organized around a few reusable contracts:

- Annotations are represented as JAMS section annotations.
- Audio foundation model embeddings are precomputed and saved as NumPy arrays.
- Dense audio embeddings are mean-pooled to beat-synchronous embeddings.
- Audio encoders, text encoders, and beat trackers are selected by registry/config.
- Models consume arrays, labels, and beat-level targets rather than raw dataset-specific files.

## Layout

```text
configs/        Example YAML configs for preprocessing, models, and training
scripts/        Command-line entry points
src/tismir/     Python package
tests/          Lightweight unit tests
```

## Environment

Create the dedicated Conda environment from the repository root:

```bash
conda env create -f environment.yml
conda activate tismir-structure
```

If the environment already exists and `environment.yml` has changed, update it with:

```bash
conda env update -f environment.yml --prune
conda activate tismir-structure
```

The base environment installs the package in editable mode with the lightweight annotation dependencies needed for JAMS support.

Optional extras can be installed as needed:

```bash
# Development tools
python -m pip install -e ".[dev]"

# Text encoders such as sentence-transformers/E5
python -m pip install -e ".[text]"

# PyTorch model training utilities
python -m pip install -e ".[torch]"

# BeatThis beat tracking backend
python -m pip install -e ".[beat]"

# Hugging Face audio encoders such as MERT
python -m pip install -e ".[hf-audio]"
```

Heavy dependencies for audio foundation models, text encoders, and beat trackers should be added behind registry backends as those integrations land.

## Development

After activating the environment, run tests with:

```bash
pytest
```

Run a quick import check:

```bash
python -c "import tismir; print(tismir.__version__)"
```

## Dataset Manifests

Datasets are exposed to the rest of the codebase through JSONL manifests. Each row points to one audio file and one JAMS structure annotation:

```json
{"track_id": "1", "audio_path": "/path/to/audio/1.wav", "jams_path": "/path/to/references/1.jams", "dataset": "rwc_pop", "split": null, "metadata": {}}
```

For paired folders like `Documents/RWC-Pop/audio` and `Documents/RWC-Pop/references`, create a local manifest with:

```bash
python scripts/create_manifest.py \
  --audio-dir ~/Documents/RWC-Pop/audio \
  --jams-dir ~/Documents/RWC-Pop/references \
  --dataset rwc_pop \
  --output data/manifests/rwc_pop.local.jsonl \
  --absolute-paths
```

Validate and summarize a manifest with:

```bash
python scripts/validate_dataset.py \
  --manifest data/manifests/rwc_pop.local.jsonl \
  --summary-json outputs/rwc_pop_summary.json
```

The validator checks that audio/JAMS paths exist, loads `segment_open` annotations by default, reports label counts, and exits with a non-zero status if any track is invalid.

## Audio Preprocessing

Audio preprocessing consumes a manifest and writes dense plus beat-synchronous NumPy arrays:

```bash
python scripts/preprocess_audio.py \
  --config configs/preprocessing/audio.yaml \
  --manifest data/manifests/rwc_pop.local.jsonl
```

The initial `placeholder` audio encoder and `uniform` beat tracker are deterministic dependency-free backends for testing the pipeline. They are not intended as research features. Real audio foundation models and beat trackers will use the same registry/config interface.

Beat tracker configs are provided for the current backends:

```bash
# Dependency-free development backend
python scripts/preprocess_audio.py \
  --config configs/preprocessing/audio.yaml \
  --manifest data/manifests/rwc_pop.local.jsonl \
  --limit 2

# BeatThis backend; downloads the selected checkpoint on first use
python -m pip install -e ".[beat]"
python scripts/preprocess_audio.py \
  --config configs/preprocessing/audio_beat_this.yaml \
  --manifest data/manifests/rwc_pop.local.jsonl \
  --limit 2

# madmom backend
python -m pip install git+https://github.com/CPJKU/madmom.git
python scripts/preprocess_audio.py \
  --config configs/preprocessing/audio_madmom.yaml \
  --manifest data/manifests/rwc_pop.local.jsonl \
  --limit 2
```

The BeatThis Python API follows the package documentation: `beat_this.inference.File2Beats` returns beat and downbeat times for an audio file. The madmom backend uses its RNN beat/downbeat processors followed by DBN tracking.

On macOS, importing PyTorch/torchaudio together with other numerical packages can trigger a duplicate OpenMP runtime error. If that happens, run BeatThis commands with:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/preprocess_audio.py \
  --config configs/preprocessing/audio_beat_this.yaml \
  --manifest data/manifests/rwc_pop.local.jsonl \
  --limit 2
```

This workaround is useful for local experiments. If it becomes a persistent issue across machines, we should pin a cleaner torch/OpenMP dependency set.

Outputs are written to:

```text
data/embeddings/audio/{encoder}/{dataset}/{track_id}/
  dense.npy
  dense_times.npy
  beats.npy
  downbeats.npy
  beat_sync.npy
  metadata.json
```

## Audio Encoder Backends

The first real audio foundation model backend is MERT through Hugging Face:

```bash
python -m pip install -e ".[hf-audio,beat]"
KMP_DUPLICATE_LIB_OK=TRUE python scripts/preprocess_audio.py \
  --config configs/preprocessing/audio_mert_beat_this.yaml \
  --manifest data/manifests/rwc_pop.local.jsonl \
  --limit 1
```

The MERT config supports selecting a hidden layer:

```yaml
audio_encoder:
  name: mert
  checkpoint: m-a-p/MERT-v1-95M
  layer: -1      # or an integer layer, or "mean"
  device: cpu
```

This follows the same broad pattern as the reference embedding scripts in `ax-le/msa_deep_embeddings`: each model backend owns its sampling rate and model-specific processor, returns a time sequence of embeddings, and the shared preprocessing pipeline pools those embeddings over beat intervals.

For longer preprocessing runs, reuse completed outputs with:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/preprocess_audio.py \
  --config configs/preprocessing/audio_mert_beat_this.yaml \
  --manifest data/manifests/rwc_pop.local.jsonl \
  --skip-existing
```

## Text Preprocessing

Text preprocessing reads labels from JAMS annotations and saves label embeddings:

```bash
python -m pip install -e ".[text]"
python scripts/preprocess_text.py \
  --config configs/preprocessing/text.yaml \
  --manifest data/manifests/rwc_pop.local.jsonl
```

Outputs are written to:

```text
data/embeddings/text/{encoder}/{dataset}/
  labels.json
  embeddings.npy
  metadata.json
```

The default scope is dataset-level vocabulary. Track-level vocabularies are also supported:

```bash
python scripts/preprocess_text.py \
  --config configs/preprocessing/text.yaml \
  --manifest data/manifests/rwc_pop.local.jsonl \
  --scope track
```

Prompt modes are configurable. The default is `bare`, which embeds the label
text directly. `compact` adds a short task prefix, and `descriptive` adds a
music-structure description for ablations:

```yaml
prompt:
  mode: bare
  normalize_whitespace: true
```

Available built-in modes:

```yaml
bare: "{label}"
compact: "Music structure label: {label}"
descriptive: "Music structure label: {label}. Base type: {base_label}. Occurrence: {occurrence_description}. Meaning: {description}. Use this label for frames belonging to this section."
```

For custom ablations, `prompt.template` can still be set explicitly and will
override the selected mode template.

Annotation label post-processing is also configurable. This is useful when a
dataset contains adjacent sections with the same label, such as `verse -> verse`.
The raw JAMS files are not modified; the policy is applied while reading labels
and targets:

```yaml
annotation_processing:
  policy: keep
```

Supported policies:

```text
keep                           preserve annotations as-is
merge                          merge consecutive same-label sections
enumerate_all_occurrences       number every repeated label occurrence, e.g. verse 1, verse 2
enumerate_consecutive_repeats   if a label repeats consecutively, number all occurrences of that label
```

The same policy must be used for text preprocessing and training data loading
when the policy changes label names. For example:

```yaml
# configs/preprocessing/text_*.yaml
annotation_processing:
  policy: enumerate_consecutive_repeats

# configs/train/*.yaml
data:
  annotation_processing:
    policy: enumerate_consecutive_repeats
```

The command-line tools also expose overrides:

```bash
python scripts/preprocess_text.py ... --annotation-policy merge
python scripts/infer.py ... --annotation-policy merge
python scripts/diagnose.py ... --annotation-policy merge
python scripts/evaluate.py ... --reference-annotation-policy merge
python scripts/validate_dataset.py ... --annotation-policy merge
```

## Training Data Inspection

After audio and text preprocessing, inspect model-ready examples with:

```bash
python scripts/inspect_training_data.py \
  --config configs/train/baseline.yaml \
  --limit 3
```

Each example contains:

```text
audio:   [num_beats, audio_dim]
text:    [num_track_labels, text_dim] by default during training
targets: [num_beats]
mask:    [num_beats] after batching
```

Training configs use `candidate_label_strategy: track_labels`, so the text side
only sees labels present in the current track. Dataset-level text embeddings are
still precomputed once and then subset per track. Inference can still use a
larger label set, such as all dataset labels or a future user-provided label set.

Training uses full tracks as examples. With the default `batch_size: 1`, each
microbatch contains one complete beat-synchronous sequence, shaped
`[1, num_beats, audio_dim]`; no temporal cropping is applied. To simulate a
larger optimizer batch without padding multiple long tracks together, keep
`batch_size: 1` and set:

```yaml
optimization:
  gradient_accumulation_steps: 4
```

Targets are projected with a LinkSeg-style adjusted annotation timeline: section intervals are adjusted to the song duration with `mir_eval.util.adjust_intervals`, then beat-synchronous frames are assigned by timeline position. Synthetic boundary labels such as `__T_MIN` and `__T_MAX` map to a silence-like candidate label when available, otherwise they use the training `ignore_index`, defaulting to `-100`.

## Baseline Training

Train the initial projection baseline with:

```bash
python scripts/train.py --config configs/train/baseline.yaml
```

The baseline projects beat-synchronous audio embeddings and candidate text-label embeddings into a shared space, then optimizes frame-label cross entropy over the provided label set. Checkpoints and metrics are saved under the configured `output_dir`.

When a validation manifest is configured, training saves `best_checkpoint.pt`
using validation loss. You can also save `best_segmentation_checkpoint.pt`
using an in-memory segmentation metric on a validation subset:

```yaml
validation:
  manifest: data/manifests/harmonix_val.local.jsonl
  segmentation:
    enabled: true
    limit: 64
    monitor: F-measure@3.0
    smoothing_window: 9
    decoder: viterbi
    transition_penalty: 1.5
```

Progress bars within each epoch can be enabled with:

```yaml
optimization:
  progress: true
```

A compact PyTorch model summary is printed before training starts by default.
It can be customized or disabled with:

```yaml
optimization:
  model_summary:
    enabled: true
    depth: 2
    max_lines: 80
```

Frame-label CE can be regularized with a pairwise probability loss. This loss
computes the full valid frame-pair matrix, uses the model-implied same-label
probability `p_i^T p_j`, and applies BCE against whether the reference labels
match:

```yaml
loss:
  pairwise_probability:
    weight: 0.5
    balance: true
```

The next model family adds transformer adapters on top of the precomputed embeddings. Audio frames receive sinusoidal beat-position encodings before audio self-attention, text label tokens are refined with text self-attention, and an optional cross-attention block lets audio frames attend to the current candidate label set before frame-label scoring:

```bash
python scripts/train.py --config configs/train/rwc_pop_mert_adapter.yaml
python scripts/train.py --config configs/train/rwc_pop_mert_cross_attention.yaml
```

Audio positional encoding is configurable. Existing boolean configs remain supported, but new configs should prefer the explicit form:

```yaml
audio:
  positional_encoding:
    type: sinusoidal  # none | sinusoidal | rope
```

RoPE variants are provided for the RWC-Pop mini experiment:

```bash
python scripts/train.py --config configs/train/rwc_pop_mert_adapter_rope.yaml
python scripts/train.py --config configs/train/rwc_pop_mert_cross_attention_rope.yaml
```

## Baseline Inference

Run baseline inference from a checkpoint and precomputed embeddings:

```bash
python scripts/infer.py \
  --checkpoint outputs/train/baseline/checkpoint.pt \
  --manifest data/manifests/rwc_pop.local.jsonl \
  --audio-encoder placeholder \
  --text-encoder sentence_transformers \
  --output-dir outputs/infer/baseline \
  --smoothing-window 5 \
  --min-segment-duration 3.0
```

Predictions are saved as JAMS plus JSON summaries under the output directory.
Inference defaults to `--candidate-label-strategy track_labels`, matching the
training condition where the model only sees the current track's label set.
Use `--candidate-label-strategy dataset_labels` to score against all
precomputed labels for the dataset instead.

For structured decoding, use the Viterbi decoder with a constant transition
penalty between labels. This usually reduces short framewise label flicker:

```bash
python scripts/infer.py \
  ... \
  --smoothing-window 9 \
  --decoder viterbi \
  --transition-penalty 1.5
```

Diagnostics can be generated after validation to inspect token relations:

```bash
python scripts/diagnose.py \
  --checkpoint outputs/train/.../best_segmentation_checkpoint.pt \
  --manifest data/manifests/harmonix_val.local.jsonl \
  --audio-encoder mert \
  --text-encoder sentence_transformers \
  --output-dir outputs/diagnostics/run_name \
  --candidate-label-strategy track_labels \
  --annotation-policy enumerate_base_occurrences
```

Per-track diagnostics include audio/text similarity heatmaps, the predicted
`P P^T` same-label probability matrix against the reference same-label matrix,
and an audio/text token t-SNE projection.

## Evaluation

Evaluate prediction JAMS files against a reference manifest:

```bash
python scripts/evaluate.py \
  --manifest data/manifests/rwc_pop.local.jsonl \
  --predictions-root outputs/infer/baseline \
  --output-json outputs/eval/baseline.json
```

The evaluator follows the same MIR-style setup as LinkSeg: reference and predicted intervals are adjusted to the song duration, then `mir_eval.segment.evaluate(..., trim=True)` is used to report boundary, pairwise, and NCE metrics.

## RWC-Pop Mini Baseline

For a first end-to-end local experiment with MERT audio embeddings, BeatThis beats, and sentence-transformer text embeddings, create a small ignored manifest subset:

```bash
sed -n '1,10p' data/manifests/rwc_pop.local.jsonl > data/manifests/rwc_pop_10.local.jsonl
```

Then run:

```bash
python scripts/preprocess_text.py \
  --config configs/preprocessing/text.yaml \
  --manifest data/manifests/rwc_pop.local.jsonl

KMP_DUPLICATE_LIB_OK=TRUE python scripts/preprocess_audio.py \
  --config configs/preprocessing/audio_mert_beat_this.yaml \
  --manifest data/manifests/rwc_pop_10.local.jsonl

python scripts/inspect_training_data.py \
  --config configs/train/rwc_pop_mert_baseline.yaml \
  --limit 3

python scripts/train.py \
  --config configs/train/rwc_pop_mert_baseline.yaml

python scripts/train.py \
  --config configs/train/rwc_pop_mert_adapter.yaml

python scripts/train.py \
  --config configs/train/rwc_pop_mert_cross_attention.yaml

python scripts/infer.py \
  --checkpoint outputs/train/rwc_pop_mert_baseline/checkpoint.pt \
  --manifest data/manifests/rwc_pop_10.local.jsonl \
  --audio-encoder mert \
  --text-encoder sentence_transformers \
  --output-dir outputs/infer/rwc_pop_mert_baseline \
  --device cpu \
  --smoothing-window 5 \
  --min-segment-duration 3.0

python scripts/evaluate.py \
  --manifest data/manifests/rwc_pop_10.local.jsonl \
  --predictions-root outputs/infer/rwc_pop_mert_baseline \
  --output-json outputs/eval/rwc_pop_mert_baseline.json
```

## RWC-Pop Mini Split

Create train/validation manifests from the 10-track local subset:

```bash
python scripts/split_manifest.py \
  --manifest data/manifests/rwc_pop_10.local.jsonl \
  --output-dir data/manifests \
  --name rwc_pop_10 \
  --train-ratio 0.8 \
  --val-ratio 0.2 \
  --seed 0
```

Split-aware training configs monitor validation loss and save both the final checkpoint and `best_checkpoint.pt`:

```bash
python scripts/train.py --config configs/train/rwc_pop_10_split_baseline.yaml
python scripts/train.py --config configs/train/rwc_pop_10_split_adapter.yaml
python scripts/train.py --config configs/train/rwc_pop_10_split_cross_attention.yaml
python scripts/train.py --config configs/train/rwc_pop_10_split_adapter_rope.yaml
python scripts/train.py --config configs/train/rwc_pop_10_split_cross_attention_rope.yaml
```

Evaluate the validation-best checkpoint on held-out tracks:

```bash
python scripts/infer.py \
  --checkpoint outputs/train/rwc_pop_10_split_adapter_rope/best_checkpoint.pt \
  --manifest data/manifests/rwc_pop_10_val.local.jsonl \
  --audio-encoder mert \
  --text-encoder sentence_transformers \
  --output-dir outputs/infer/rwc_pop_10_split_adapter_rope_best \
  --device cpu \
  --smoothing-window 5 \
  --min-segment-duration 3.0

python scripts/evaluate.py \
  --manifest data/manifests/rwc_pop_10_val.local.jsonl \
  --predictions-root outputs/infer/rwc_pop_10_split_adapter_rope_best \
  --output-json outputs/eval/rwc_pop_10_split_adapter_rope_best.json
```

## RWC-Pop Full Split

After preprocessing all 100 RWC-Pop tracks, create the full local split:

```bash
python scripts/split_manifest.py \
  --manifest data/manifests/rwc_pop.local.jsonl \
  --output-dir data/manifests \
  --name rwc_pop \
  --train-ratio 0.8 \
  --val-ratio 0.2 \
  --seed 0
```

Initial full-split configs compare the projection baseline against the RoPE adapter:

```bash
python scripts/train.py --config configs/train/rwc_pop_split_baseline.yaml
python scripts/train.py --config configs/train/rwc_pop_split_adapter_rope.yaml
```

Evaluate validation-best checkpoints on the held-out split:

```bash
python scripts/infer.py \
  --checkpoint outputs/train/rwc_pop_split_adapter_rope/best_checkpoint.pt \
  --manifest data/manifests/rwc_pop_val.local.jsonl \
  --audio-encoder mert \
  --text-encoder sentence_transformers \
  --output-dir outputs/infer/rwc_pop_split_adapter_rope_best \
  --device cpu \
  --smoothing-window 5 \
  --min-segment-duration 3.0

python scripts/evaluate.py \
  --manifest data/manifests/rwc_pop_val.local.jsonl \
  --predictions-root outputs/infer/rwc_pop_split_adapter_rope_best \
  --output-json outputs/eval/rwc_pop_split_adapter_rope_best.json
```

## Harmonix

Create and validate a local Harmonix manifest:

```bash
python scripts/create_manifest.py \
  --audio-dir ~/Documents/Harmonix/audio \
  --jams-dir ~/Documents/Harmonix/references \
  --dataset harmonix \
  --output data/manifests/harmonix.local.jsonl \
  --absolute-paths

python scripts/validate_dataset.py \
  --manifest data/manifests/harmonix.local.jsonl \
  --summary-json outputs/harmonix_summary.json
```

Precompute dataset-level text embeddings. Harmonix uses compact label codes, so
this config keeps raw labels for targets while normalizing prompt text
(`fadeout` -> `fade out`, `verseinst` -> `instrumental verse`):

```bash
python scripts/preprocess_text.py \
  --config configs/preprocessing/text_harmonix.yaml \
  --manifest data/manifests/harmonix.local.jsonl
```

The default Harmonix text config uses `prompt.mode: bare`. For prompt ablations,
use `configs/preprocessing/text_harmonix_compact.yaml` or
`configs/preprocessing/text_harmonix_descriptive.yaml`.

Create a full local split:

```bash
python scripts/split_manifest.py \
  --manifest data/manifests/harmonix.local.jsonl \
  --output-dir data/manifests \
  --name harmonix \
  --train-ratio 0.8 \
  --val-ratio 0.2 \
  --seed 0
```

Full MERT+BeatThis preprocessing is resumable but long for Harmonix:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/preprocess_audio.py \
  --config configs/preprocessing/audio_mert_beat_this.yaml \
  --manifest data/manifests/harmonix.local.jsonl \
  --skip-existing
```

Once audio embeddings are available, train the initial full-split baselines:

```bash
python scripts/train.py --config configs/train/harmonix_split_baseline.yaml
python scripts/train.py --config configs/train/harmonix_split_adapter_rope.yaml
```

## Beat-Pooling Experiments

MERT frames are pooled to one vector per beat before training. Beyond the
default `mean`, the pooling family in `src/tismir/preprocessing/beat_sync.py`
supports `max`, `energy_weighted` (softmax over per-frame L2 norm, with a
`temperature`), `multi_stat` (concatenation of `mean`/`max`/`std`, so the audio
dim widens to `mert_dim * len(stats)`), `first`, and `last`. The model infers
its audio input dim from the saved array, so the wider `multi_stat` output needs
no model change. The pooling method, empty-interval policy, temperature, and
stats are recorded in each track's `metadata.json`.

### Datasets

This experiment set targets the HarmonixSet (bigvgan render) and SALAMI. Both
ship in non-standard local layouts, so there are dedicated converters.

HarmonixSet — audio at `data/raw/harmonix/harmonixset_bigvgan/tracks/<id>.wav`,
structure in the corrected JSONL. Convert `msa_info` onset/label pairs into
`segment_open` JAMS and emit manifests carrying the official train/val/test
split:

```bash
python scripts/convert_harmonix_jsonl_to_jams.py
# -> data/raw/harmonix/jams/<id>.jams
# -> data/manifests/harmonix_bigvgan.local.jsonl (+ _{train,val,test})
python scripts/validate_dataset.py \
  --manifest data/manifests/harmonix_bigvgan.local.jsonl \
  --namespace segment_open
```

SALAMI (nested-mp3 dump) — `audio/<id>/audio.mp3` + `jams/<id>.jams`:

```bash
python scripts/create_salami_manifest.py --layout nested-mp3
# -> data/manifests/salami.local.jsonl (1359 tracks)
python scripts/validate_dataset.py \
  --manifest data/manifests/salami.local.jsonl \
  --namespace segment_salami_function
```

### Preprocessing all pooling methods in one pass

The dense MERT frames are identical across pooling methods, so encoding once and
pooling every method is far cheaper than one full pass per method.
`scripts/preprocess_audio_multipool.py` does exactly that, taking the per-method
configs (each supplies its own `output_root` and `pooling` block). Because the
dataset name is a path component, one `output_root` per method serves both
datasets:

```bash
# HarmonixSet: encode once, write all 6 pooling variants (resumable)
KMP_DUPLICATE_LIB_OK=TRUE python scripts/preprocess_audio_multipool.py \
  --manifest data/manifests/harmonix_bigvgan.local.jsonl --skip-existing

# SALAMI: same variants, same output roots (dataset subdir keeps them separate)
KMP_DUPLICATE_LIB_OK=TRUE python scripts/preprocess_audio_multipool.py \
  --manifest data/manifests/salami.local.jsonl --skip-existing
```

Outputs land at
`data/embeddings/audio_pool_<method>/mert/<dataset>/<id>/beat_sync.npy`. To run a
single method the ordinary way, use its config with `scripts/preprocess_audio.py`
(e.g. `configs/preprocessing/audio_mert_madmom_pool_multistat.yaml`).

> The pooling configs use the `madmom` beat tracker
> (`python -m pip install git+https://github.com/CPJKU/madmom.git`). Switch the
> `beat_tracker` block to `beat_this` if you prefer to avoid the from-source
> build.

### Text embeddings and training

Text embeddings are pooling-independent (one set per dataset):

```bash
python scripts/preprocess_text.py \
  --config configs/preprocessing/text_harmonix.yaml \
  --manifest data/manifests/harmonix_bigvgan.local.jsonl
```

One self-attention adapter train config per pooling method differs only in
`audio_embedding_root` (`configs/train/harmonix_bigvgan_self_attention_pool_*.yaml`).
Note `configs/train/` is gitignored, so these live untracked:

```bash
python scripts/train.py \
  --config configs/train/harmonix_bigvgan_self_attention_pool_multistat.yaml
```

### Learnable in-model beat pooling (dense path)

Instead of a fixed parameter-free pool, a `SegmentAttentionPool` can learn to
attend over the dense frames within each beat. Preprocess with dense frames kept
(`configs/preprocessing/audio_mert_madmom_keep_dense.yaml` saves `dense.npy` /
`dense_times.npy` alongside a mean `beat_sync.npy`):

```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/preprocess_audio.py \
  --config configs/preprocessing/audio_mert_madmom_keep_dense.yaml \
  --manifest data/manifests/harmonix_bigvgan.local.jsonl --skip-existing
```

Then train with `data.audio_embedding_key: dense` and `model.audio.beat_pool`
enabled (`configs/train/harmonix_bigvgan_self_attention_beat_pool_attention.yaml`).
The data loader carries the dense frames and each frame's beat segment id; the
model pools frames to one vector per beat with a learned query before the audio
projection. The default beat-sync path is untouched when `beat_pool` is absent.

