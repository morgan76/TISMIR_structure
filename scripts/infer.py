#!/usr/bin/env python3
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run label-set-conditioned structure inference.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.parse_args()

    raise NotImplementedError("Inference scaffold is ready; model loading and decoding will be added next.")


if __name__ == "__main__":
    main()
