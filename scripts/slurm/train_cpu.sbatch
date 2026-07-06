#!/bin/bash
#SBATCH --job-name=tismir_train_cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ick/music_structure/tismir_data/logs/%x-%j.out
#
# CPU variant of train.sbatch. The baseline/adapter models are tiny and train
# from precomputed embeddings, so CPU is viable and keeps GPU slots free for
# encoder preprocessing. Config `device: auto` resolves to cpu here.
#
#   sbatch scripts/slurm/train_cpu.sbatch <train_config.yaml>
set -euo pipefail
source /scratch/ick/wt/slurm/scripts/slurm/common.sh

CONFIG="${1:?usage: train_cpu.sbatch <train_config.yaml>}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python scripts/train.py --config "$CONFIG"
