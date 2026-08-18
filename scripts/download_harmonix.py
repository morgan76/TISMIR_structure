#!/usr/bin/env python3
"""Download the Harmonix BigVGAN dataset from the Hugging Face Hub.

Fetches the audio archive (``harmonixset_bigvgan.zip``) and the corrected
annotation file (``harmonixset.corrected.20250821.jsonl``) from
``m-a-p/harmonixset_bigvgan`` into a local directory. The archive is large
(~8.5 GB); run this detached and unzip afterwards.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "m-a-p/harmonixset_bigvgan"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Harmonix BigVGAN dataset from HF.")
    parser.add_argument("--output-dir", default="data/raw/harmonix_bigvgan", type=Path)
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=None,
        help="Optional glob(s) restricting which files to download (repeatable).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    local_dir = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(args.output_dir),
        allow_patterns=args.allow_pattern,
    )
    print(f"Downloaded {REPO_ID} to {local_dir}")


if __name__ == "__main__":
    main()
