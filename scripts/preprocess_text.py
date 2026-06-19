#!/usr/bin/env python3
from __future__ import annotations

import argparse

from tismir.data.manifest import load_manifest
from tismir.io import load_yaml
from tismir.preprocessing.text import preprocess_dataset_text, result_to_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute text label embeddings.")
    parser.add_argument("--config", default="configs/preprocessing/text.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--namespace", default="segment_open")
    parser.add_argument("--scope", choices=["dataset", "track"], default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    output_root = args.output_root or config["output_root"]
    text_config = dict(config.get("text_encoder", {}))
    text_name = text_config.pop("name")
    tracks = load_manifest(args.manifest)

    results = preprocess_dataset_text(
        tracks=tracks,
        output_root=output_root,
        text_encoder_name=text_name,
        text_encoder_params=text_config,
        prompt=config.get("prompt", {}),
        namespace=args.namespace,
        scope=args.scope or config.get("scope", "dataset"),
    )
    for result in results:
        print(
            f"{result.dataset}: labels={result.num_labels}, "
            f"embeddings={result.embedding_shape}, output={result.output_dir}"
        )


if __name__ == "__main__":
    main()
