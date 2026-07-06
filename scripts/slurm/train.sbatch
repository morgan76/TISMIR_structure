#!/bin/bash
#SBATCH --job-name=tismir_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/ick/music_structure/tismir_data/logs/%x-%j.out
#
# Train a baseline / adapter model from precomputed embeddings.
# Use a cluster config (configs/train/cluster/*.yaml) — those point manifests,
# embedding roots and output_dir at $TISMIR_DATA instead of repo-relative paths.
#
#   sbatch scripts/slurm/train.sbatch configs/train/cluster/harmonix_split_adapter_rope.yaml
set -euo pipefail
source "${TISMIR_ROOT:-/scratch/ick/wt-slurm}/scripts/slurm/common.sh"

CONFIG="${1:?usage: train.sbatch <train_config.yaml>}"

# Shim: the temporal_text_adapter uses nn.MultiheadAttention -> SDPA -> would
# hit the broken cuDNN backend on sm_103.
srun python scripts/slurm/sm103_shim.py scripts/train.py --config "$CONFIG"
