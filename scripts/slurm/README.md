# SLURM jobs for TISMIR structure experiments

Cluster: B300 (Blackwell sm_103) GPU nodes, partitions `gpu` (7d) / `gpu_hp`
(unlimited) / `debug` (30 min) / `cpu` (default — has a `nogpu` QOS, so GPU jobs
**must** pass `-p gpu`). See `~/slurm_howto/slurm_cheatsheet.md`.

## Layout

| Path | Purpose |
|------|---------|
| `/scratch/ick/music_structure/venv-gpu` | cu128 torch venv (created by `setup_gpu_venv.sh`) |
| `/scratch/ick/music_structure/models` | HF / torch / checkpoint caches (`HF_HOME` etc.) |
| `/scratch/ick/music_structure/tismir_data` | manifests, embeddings, outputs, SLURM logs |
| `$TISMIR_ROOT` (default `/scratch/ick/wt-slurm`) | code checkout jobs run from (`PYTHONPATH=$TISMIR_ROOT/src`) |

`common.sh` wires all of this; every sbatch script sources it. Override
`TISMIR_ROOT` at submit time to run code from another worktree:
`TISMIR_ROOT=/scratch/ick/wt-dataset-rwc_pop sbatch --export=ALL ...`

## sm_103 gotcha

The cuDNN SDPA backend is broken on these GPUs (`cuDNN Frontend error: No valid
execution plans built`). All GPU python entry points run through
`sm103_shim.py`, which calls `torch.backends.cuda.enable_cudnn_sdp(False)` and
then executes the target script. Do not bypass it for MERT, BeatThis, or
adapter training.

## Order of operations

```bash
# one-time
bash scripts/slurm/setup_gpu_venv.sh

# per dataset (manifest must exist under tismir_data/manifests)
sbatch scripts/slurm/preprocess_text.sbatch  /scratch/ick/music_structure/tismir_data/manifests/harmonix.jsonl configs/preprocessing/text_harmonix.yaml
sbatch scripts/slurm/preprocess_audio.sbatch /scratch/ick/music_structure/tismir_data/manifests/harmonix.jsonl

# per experiment
sbatch scripts/slurm/train.sbatch configs/train/cluster/harmonix_split_adapter_rope.yaml
sbatch scripts/slurm/infer_eval.sbatch \
  /scratch/ick/music_structure/tismir_data/outputs/train/harmonix_split_adapter_rope/best_checkpoint.pt \
  /scratch/ick/music_structure/tismir_data/manifests/harmonix_val.jsonl \
  harmonix_split_adapter_rope_best
```

`preprocess_audio.sbatch` passes `--skip-existing`, so resubmitting after a
timeout resumes where it stopped.
