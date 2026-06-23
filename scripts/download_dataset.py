#!/usr/bin/env python3
"""Download music structure analysis datasets described by ``configs/dataset/*.yaml``.

Each dataset is declared by a small YAML file listing the artifacts to fetch
(git repositories and/or zip archives). This script reads those declarations and
materializes the datasets under a destination directory.

Examples
--------
List the datasets that have a config available on this branch::

    python scripts/download_dataset.py --list

Download one dataset into a chosen location::

    python scripts/download_dataset.py --dataset rwc_pop --dest /scratch/ick/music_structure

Download every available dataset but skip large audio archives::

    python scripts/download_dataset.py --dataset all --skip-audio
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs" / "dataset"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download music structure datasets declared in configs/dataset/.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        metavar="NAME",
        help="Dataset name(s) to download, or 'all' for every available config.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path(os.environ.get("TISMIR_DATA_ROOT", REPO_ROOT / "data" / "raw")),
        help="Destination directory for downloaded datasets "
        "(default: $TISMIR_DATA_ROOT or <repo>/data/raw).",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help="Directory containing dataset YAML configs.",
    )
    parser.add_argument(
        "--skip-audio",
        action="store_true",
        help="Skip steps flagged 'audio: true' (e.g. multi-GB audio archives).",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep downloaded .zip archives after extraction.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List datasets with a config in --config-dir and exit.",
    )
    args = parser.parse_args()

    configs = load_configs(args.config_dir)

    if args.list:
        list_datasets(configs)
        return
    if not args.dataset:
        list_datasets(configs)
        parser.error("nothing to do: pass --dataset NAME [NAME ...] or --dataset all")

    selected = resolve_selection(args.dataset, configs)
    args.dest.mkdir(parents=True, exist_ok=True)
    for name in selected:
        download_dataset(name, configs[name], args.dest, args.skip_audio, args.keep_archives)
    print(f"\nDone. Datasets under: {args.dest}")


def load_configs(config_dir: Path) -> dict[str, dict[str, Any]]:
    if not config_dir.is_dir():
        raise SystemExit(f"Config directory not found: {config_dir}")
    configs: dict[str, dict[str, Any]] = {}
    for path in sorted(config_dir.glob("*.yaml")):
        with path.open() as handle:
            config = yaml.safe_load(handle) or {}
        name = config.get("name", path.stem)
        configs[name] = config
    return configs


def resolve_selection(requested: list[str], configs: dict[str, dict[str, Any]]) -> list[str]:
    if len(requested) == 1 and requested[0] == "all":
        return list(configs)
    unknown = [name for name in requested if name not in configs]
    if unknown:
        available = ", ".join(configs) or "(none)"
        raise SystemExit(f"Unknown dataset(s): {', '.join(unknown)}. Available: {available}")
    return requested


def list_datasets(configs: dict[str, dict[str, Any]]) -> None:
    if not configs:
        print("No dataset configs found.")
        return
    print("Available datasets:")
    for name, config in configs.items():
        audio = "audio+annotations" if config.get("audio_included") else "annotations only"
        display = config.get("display_name", name)
        print(f"  {name:12s} {display}  [{audio}]")


def download_dataset(
    name: str,
    config: dict[str, Any],
    dest: Path,
    skip_audio: bool,
    keep_archives: bool,
) -> None:
    print(f"\n=== {config.get('display_name', name)} ({name}) ===")
    if not config.get("audio_included", False) and config.get("audio_note"):
        print(f"note: {config['audio_note'].strip()}")
    for step in config.get("steps", []):
        if step.get("audio") and skip_audio:
            print(f"[skip-audio] skipping {step.get('url')}")
            continue
        kind = step.get("kind")
        if kind == "git":
            fetch_git(step, dest)
        elif kind == "zip":
            fetch_zip(step, dest, keep_archives)
        elif kind == "file":
            fetch_file(step, dest)
        else:
            raise SystemExit(f"Unknown step kind '{kind}' in dataset '{name}'")


def fetch_git(step: dict[str, Any], dest: Path) -> None:
    url = step["url"]
    target = dest / step["dest"]
    if (target / ".git").is_dir():
        print(f"[git] {target.name} exists; pulling")
        subprocess.run(["git", "-C", str(target), "pull", "--ff-only"], check=False)
        return
    print(f"[git] clone {url} -> {target}")
    cmd = ["git", "clone", "--depth", "1", url, str(target)]
    subprocess.run(cmd, check=True)


def fetch_zip(step: dict[str, Any], dest: Path, keep_archives: bool) -> None:
    url = step["url"]
    target = dest / step["dest"]
    target.mkdir(parents=True, exist_ok=True)
    marker = step.get("extracted_marker")
    if marker and (target / marker).exists():
        print(f"[zip] {target / marker} already present; skipping")
        return
    archive = target / Path(url.split("?")[0]).name
    download(url, archive)
    print(f"[zip] extracting {archive.name}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(target)
    if not keep_archives:
        archive.unlink()


def fetch_file(step: dict[str, Any], dest: Path) -> None:
    target = dest / step["dest"]
    target.parent.mkdir(parents=True, exist_ok=True)
    download(step["url"], target)


def download(url: str, target: Path) -> None:
    if target.exists():
        print(f"[get] {target.name} already present; skipping")
        return
    print(f"[get] {url}")
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url) as response:  # noqa: S310 (trusted dataset hosts)
        total = int(response.headers.get("Content-Length", 0))
        read = 0
        with tmp.open("wb") as handle:
            while chunk := response.read(1 << 20):
                handle.write(chunk)
                read += len(chunk)
                if total:
                    pct = 100 * read / total
                    print(f"\r      {read >> 20} / {total >> 20} MiB ({pct:4.1f}%)", end="")
        if total:
            print()
    shutil.move(str(tmp), str(target))


if __name__ == "__main__":
    main()
