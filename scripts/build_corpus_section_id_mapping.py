#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tismir.data.annotations import (
    ANNOTATION_PROCESSING_POLICIES,
    build_corpus_section_id_mapping,
)
from tismir.data.jams import load_processed_structure_sections, unique_labels
from tismir.data.manifest import load_manifest


SOURCE_POLICIES = sorted(
    ANNOTATION_PROCESSING_POLICIES
    - {"section_ids_ordered", "section_ids_shuffled", "section_ids_corpus"}
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a fixed corpus-level mapping from real section labels to anonymous section IDs."
    )
    parser.add_argument("--manifest", action="append", required=True, help="Manifest path; can be repeated.")
    parser.add_argument("--output", required=True, help="Output JSON mapping path.")
    parser.add_argument("--namespace", default="segment_open")
    parser.add_argument("--prefix", default="section")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--preserve-label",
        action="append",
        default=None,
        help="Label to keep unchanged instead of anonymizing; can be repeated.",
    )
    parser.add_argument(
        "--source-annotation-policy",
        choices=SOURCE_POLICIES,
        default=None,
        help="Optional source label processing to apply before building the mapping.",
    )
    parser.add_argument(
        "--annotation-selection",
        choices=["first", "richest_function"],
        default=None,
        help="Optional annotation-selection strategy for multi-annotator JAMS files.",
    )
    args = parser.parse_args()

    source_processing = None
    if args.source_annotation_policy is not None:
        source_processing = {"policy": args.source_annotation_policy}
        if args.annotation_selection is not None:
            source_processing["annotation_selection"] = args.annotation_selection
    elif args.annotation_selection is not None:
        source_processing = {"policy": "keep", "annotation_selection": args.annotation_selection}

    preserve_labels = args.preserve_label or ["silence"]
    labels: list[str] = []
    for manifest_path in args.manifest:
        for track in load_manifest(manifest_path):
            sections = load_processed_structure_sections(
                track.jams_path,
                namespace=args.namespace,
                annotation_processing=source_processing,
            )
            labels.extend(unique_labels(sections))

    labels = list(dict.fromkeys(labels))
    mapping = build_corpus_section_id_mapping(
        labels,
        prefix=args.prefix,
        seed=args.seed,
        preserve_labels=preserve_labels,
    )
    payload = {
        "mapping": mapping,
        "labels": labels,
        "prefix": args.prefix,
        "seed": args.seed,
        "preserve_labels": preserve_labels,
        "namespace": args.namespace,
        "source_annotation_processing": source_processing,
        "manifests": args.manifest,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"saved {len(mapping)} mapped labels to {output}")


if __name__ == "__main__":
    main()
