#!/usr/bin/env python3
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predicted music structure segmentations.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--references", required=True)
    parser.parse_args()

    raise NotImplementedError("Evaluation scaffold is ready; metrics will be added with datasets.")


if __name__ == "__main__":
    main()
