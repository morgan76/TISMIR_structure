#!/bin/bash
#SBATCH --job-name=tismir_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ick/music_structure/tismir_data/logs/%x-%j.out
#
# Inference (checkpoint + precomputed embeddings -> predicted JAMS) followed by
# mir_eval segmentation metrics against the reference manifest.
#
#   sbatch scripts/slurm/infer_eval.sbatch <checkpoint.pt> <val_manifest.jsonl> <run_name>
#
# Optional env overrides: AUDIO_ENCODER (mert), TEXT_ENCODER (sentence_transformers),
# SMOOTHING (9), DECODER (viterbi), TRANSITION_PENALTY (1.5), MIN_SEG (3.0).
set -euo pipefail
# SLURM tooling always lives on the infra/slurm worktree; TISMIR_ROOT
# (set in common.sh) selects which branch worktree the *pipeline code* runs from.
source /scratch/ick/wt/slurm/scripts/slurm/common.sh

CHECKPOINT="${1:?usage: infer_eval.sbatch <checkpoint.pt> <manifest.jsonl> <run_name>}"
MANIFEST="${2:?missing reference manifest}"
RUN_NAME="${3:?missing run name}"

AUDIO_ENCODER="${AUDIO_ENCODER:-mert}"
TEXT_ENCODER="${TEXT_ENCODER:-sentence_transformers}"
SMOOTHING="${SMOOTHING:-9}"
DECODER="${DECODER:-viterbi}"
TRANSITION_PENALTY="${TRANSITION_PENALTY:-1.5}"
MIN_SEG="${MIN_SEG:-3.0}"

INFER_DIR="$TISMIR_DATA/outputs/infer/$RUN_NAME"
EVAL_JSON="$TISMIR_DATA/outputs/eval/$RUN_NAME.json"
mkdir -p "$(dirname "$EVAL_JSON")"

srun python /scratch/ick/wt/slurm/scripts/slurm/sm103_shim.py scripts/infer.py \
  --checkpoint "$CHECKPOINT" \
  --manifest "$MANIFEST" \
  --audio-embedding-root "$TISMIR_DATA/embeddings/audio" \
  --audio-encoder "$AUDIO_ENCODER" \
  --text-embedding-root "$TISMIR_DATA/embeddings/text" \
  --text-encoder "$TEXT_ENCODER" \
  --output-dir "$INFER_DIR" \
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
