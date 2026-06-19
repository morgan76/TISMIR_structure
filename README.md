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
