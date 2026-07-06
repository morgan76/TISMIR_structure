"""Run a script with the cuDNN SDPA backend disabled (B300 / sm_103 workaround).

On this cluster's Blackwell GPUs, any scaled_dot_product_attention call that
dispatches to cuDNN raises `cuDNN Frontend error: ... No valid execution plans
built`. This affects HF transformers (MERT), BeatThis, and the project's
nn.MultiheadAttention adapters. The mem-efficient/math SDPA backends work fine.
(Do NOT disable cuDNN entirely — that segfaults.)

Usage: python scripts/slurm/sm103_shim.py <script.py> [args...]
"""
from __future__ import annotations

import runpy
import sys

import torch

if torch.cuda.is_available():
    torch.backends.cuda.enable_cudnn_sdp(False)

if len(sys.argv) < 2:
    raise SystemExit("usage: sm103_shim.py <script.py> [args...]")

target = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(target, run_name="__main__")
