#!/bin/bash
#SBATCH --job-name=tismir_pre_text
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ick/music_structure/tismir_data/logs/%x-%j.out
#
# Text preprocessing (dataset-level label vocabulary -> E5 embeddings). CPU is
# plenty: one embedding pass over a few dozen labels per dataset.
#
#   sbatch scripts/slurm/preprocess_text.sbatch <manifest.jsonl> [text_config.yaml]
set -euo pipefail
source "${TISMIR_ROOT:-/scratch/ick/wt-slurm}/scripts/slurm/common.sh"

MANIFEST="${1:?usage: preprocess_text.sbatch <manifest.jsonl> [config.yaml]}"
CONFIG="${2:-configs/preprocessing/text.yaml}"

srun python scripts/preprocess_text.py \
  --config "$CONFIG" \
  --manifest "$MANIFEST" \
  --output-root "$TISMIR_DATA/embeddings/text"
