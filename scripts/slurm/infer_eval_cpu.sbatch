#!/bin/bash
#SBATCH --job-name=tismir_eval_cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/ick/music_structure/tismir_data/logs/%x-%j.out
#
# CPU variant of infer_eval.sbatch: inference is a tiny-model forward over
# precomputed embeddings; evaluation is pure mir_eval. No GPU needed.
#
#   sbatch scripts/slurm/infer_eval_cpu.sbatch <checkpoint.pt> <manifest.jsonl> <run_name>
set -euo pipefail
source /scratch/ick/wt/slurm/scripts/slurm/common.sh

CHECKPOINT="${1:?usage: infer_eval_cpu.sbatch <checkpoint.pt> <manifest.jsonl> <run_name>}"
MANIFEST="${2:?missing reference manifest}"
RUN_NAME="${3:?missing run name}"

AUDIO_ENCODER="${AUDIO_ENCODER:-mert}"
TEXT_ENCODER="${TEXT_ENCODER:-sentence_transformers}"
SMOOTHING="${SMOOTHING:-9}"
DECODER="${DECODER:-viterbi}"
TRANSITION_PENALTY="${TRANSITION_PENALTY:-1.5}"
MIN_SEG="${MIN_SEG:-3.0}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

INFER_DIR="$TISMIR_DATA/outputs/infer/$RUN_NAME"
EVAL_JSON="$TISMIR_DATA/outputs/eval/$RUN_NAME.json"
mkdir -p "$(dirname "$EVAL_JSON")"

srun python scripts/infer.py \
  --checkpoint "$CHECKPOINT" \
  --manifest "$MANIFEST" \
  --audio-embedding-root "$TISMIR_DATA/embeddings/audio" \
  --audio-encoder "$AUDIO_ENCODER" \
  --text-embedding-root "$TISMIR_DATA/embeddings/text" \
  --text-encoder "$TEXT_ENCODER" \
  --output-dir "$INFER_DIR" \
  --device cpu \
  --candidate-label-strategy track_labels \
  --smoothing-window "$SMOOTHING" \
  --decoder "$DECODER" \
  --transition-penalty "$TRANSITION_PENALTY" \
  --min-segment-duration "$MIN_SEG"

srun python scripts/evaluate.py \
  --manifest "$MANIFEST" \
  --predictions-root "$INFER_DIR" \
  --output-json "$EVAL_JSON"

echo "eval written to $EVAL_JSON"
