#!/bin/bash
# Create the GPU venv for cluster jobs. Run ONCE from a login node (downloads ~3 GB).
#
# The repo's ~/TISMIR_structure/.venv is CPU-only torch (2.12.1+cpu) and is used
# for local encoder verification. Cluster GPUs are B300 (Blackwell, sm_103), which
# need a cu128+ torch build — the default PyPI wheel is too old.
#
#   bash scripts/slurm/setup_gpu_venv.sh
set -euo pipefail

VENV="${TISMIR_GPU_VENV:-/scratch/ick/music_structure/venv-gpu}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

uv venv --clear --python 3.12 "$VENV"
# cu128 torch first, from the CUDA index (sm_103 support).
uv pip install --python "$VENV/bin/python" \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch torchaudio
# Project + extras (deps resolve from PyPI; torch req already satisfied).
uv pip install --python "$VENV/bin/python" \
  -e "$REPO_ROOT[annotations,progress,hf-audio,text,beat,dev]" \
  "numba>=0.60" "llvmlite>=0.43"
# NOTES:
# - [diagnostics] omitted — umap-learn's pynndescent pin resolves to a
#   py<3.10 llvmlite on Python 3.12. Install matplotlib/scikit-learn ad hoc.
# - numba/llvmlite pinned forward: without them uv backtracks librosa's numba
#   dep to a py<3.10 sdist and the build fails.

"$VENV/bin/python" - <<'EOF'
import torch
print("torch", torch.__version__, "| cuda build:", torch.version.cuda)
EOF
# Shared data layout used by the sbatch scripts (incl. the SLURM log dir,
# which must exist before the first job is submitted).
mkdir -p /scratch/ick/music_structure/tismir_data/{manifests,embeddings/audio,embeddings/text,outputs,logs}

echo "GPU venv ready at $VENV"
echo "NOTE: sbatch scripts set PYTHONPATH to the code they run, so the editable"
echo "install above is only a dependency anchor."
