#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_rwc_pop_text_encoder_ablation import SETUPS as BARE_SETUPS
from scripts.run_rwc_pop_text_encoder_ablation import main as _bare_main


SETUPS = {
    "e5_base_v2": {
        "text_embedding_root": "data/embeddings/text_rwc_pop_base_compact_definition",
        "text_encoder": "sentence_transformers",
        "annotation_processing": {"policy": "base_labels"},
    },
    "mpnet_base_v2": {
        "text_embedding_root": "data/embeddings/text_rwc_pop_base_compact_definition_mpnet",
        "text_encoder": "sentence_transformers",
        "annotation_processing": {"policy": "base_labels"},
    },
    "clap_music_speech": {
        "text_embedding_root": "data/embeddings/text_rwc_pop_base_compact_definition_clap_music_speech",
        "text_encoder": "clap",
        "annotation_processing": {"policy": "base_labels"},
    },
    "muq_mulan_large": {
        "text_embedding_root": "data/embeddings/text_rwc_pop_base_compact_definition_muq_mulan",
        "text_encoder": "muq_mulan",
        "annotation_processing": {"policy": "base_labels"},
    },
}


def main() -> None:
    BARE_SETUPS.clear()
    BARE_SETUPS.update(SETUPS)
    _bare_main()


if __name__ == "__main__":
    main()
