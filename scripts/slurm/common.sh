# Shared environment for TISMIR SLURM jobs. Source from sbatch scripts.
# Override TISMIR_ROOT to run code from a different worktree/branch.
export TISMIR_ROOT="${TISMIR_ROOT:-/scratch/ick/wt-slurm}"
export TISMIR_DATA="${TISMIR_DATA:-/scratch/ick/music_structure/tismir_data}"
export TISMIR_GPU_VENV="${TISMIR_GPU_VENV:-/scratch/ick/music_structure/venv-gpu}"

# Model/checkpoint caches (already populated — see /scratch/ick/music_structure/models).
_MODELS=/scratch/ick/music_structure/models
export HF_HOME="$_MODELS/hf"
export TORCH_HOME="$_MODELS/torch"
export XDG_CACHE_HOME="$_MODELS/cache"

source "$TISMIR_GPU_VENV/bin/activate"
export PYTHONPATH="$TISMIR_ROOT/src"
cd "$TISMIR_ROOT"

echo "Job ${SLURM_JOB_ID:-<none>} on $(hostname) | code: $TISMIR_ROOT | data: $TISMIR_DATA"
