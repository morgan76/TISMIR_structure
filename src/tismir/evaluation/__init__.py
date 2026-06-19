"""Evaluation utilities."""

from tismir.evaluation.segments import (
    evaluate_pair,
    evaluate_prediction_manifest,
    format_evaluation,
    save_evaluation,
)

__all__ = [
    "evaluate_pair",
    "evaluate_prediction_manifest",
    "format_evaluation",
    "save_evaluation",
]
