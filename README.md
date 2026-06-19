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

Prompt templates are configurable:

```yaml
prompt:
  template: "{label}"
  normalize_whitespace: true
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
text:    [num_labels, text_dim]
targets: [num_beats]
mask:    [num_beats] after batching
```

Targets are projected with a LinkSeg-style adjusted annotation timeline: section intervals are adjusted to the song duration with `mir_eval.util.adjust_intervals`, then beat-synchronous frames are assigned by timeline position. Synthetic boundary labels such as `__T_MIN` and `__T_MAX` map to a silence-like candidate label when available, otherwise they use the training `ignore_index`, defaulting to `-100`.

## Baseline Training

Train the initial projection baseline with:

```bash
python scripts/train.py --config configs/train/baseline.yaml
```

The baseline projects beat-synchronous audio embeddings and candidate text-label embeddings into a shared space, then optimizes frame-label cross entropy over the provided label set. Checkpoints and metrics are saved under the configured `output_dir`.

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
