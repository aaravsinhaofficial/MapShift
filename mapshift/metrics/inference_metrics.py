"""Inference metrics used by MapShift."""

from __future__ import annotations


def accuracy(correct: int, total: int) -> float:
    """Compute simple accuracy."""

    if total <= 0:
        return 0.0
    return correct / total


def auroc_placeholder(scores: list[float], labels: list[int]) -> float:
    """Placeholder AUROC helper until a full implementation is added."""

    if not scores or len(scores) != len(labels):
        return 0.0
    return 0.5
