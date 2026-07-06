#!/bin/bash
#SBATCH --job-name=tismir_pre_audio
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/ick/music_structure/tismir_data/logs/%x-%j.out
#
# Audio preprocessing: MERT dense embeddings + BeatThis beats -> beat-sync arrays.
# Resumable: --skip-existing skips tracks that already have beat_sync.npy, so the
# same job can be resubmitted if it times out.
#
#   sbatch scripts/slurm/preprocess_audio.sbatch <manifest.jsonl> [audio_config.yaml]
set -euo pipefail
source "${TISMIR_ROOT:-/scratch/ick/wt/slurm}/scripts/slurm/common.sh"

MANIFEST="${1:?usage: preprocess_audio.sbatch <manifest.jsonl> [config.yaml]}"
CONFIG="${2:-configs/preprocessing/audio_mert_beat_this_cuda.yaml}"

# sm103_shim disables the cuDNN SDPA backend (broken on B300); MERT/BeatThis
# attention then uses the mem-efficient/math backends.
srun python scripts/slurm/sm103_shim.py scripts/preprocess_audio.py \
  --config "$CONFIG" \
  --manifest "$MANIFEST" \
  --output-root "$TISMIR_DATA/embeddings/audio" \
  --device cuda \
  --skip-existing
