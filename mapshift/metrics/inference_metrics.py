"""Inference metrics used by MapShift."""

from __future__ import annotations

from typing import Iterable, Sequence


def accuracy_from_flags(values: Iterable[bool]) -> float:
    """Compute simple accuracy from boolean correctness flags."""

    flags = list(values)
    if not flags:
        return 0.0
    return sum(1 for value in flags if value) / len(flags)


def accuracy(correct: int, total: int) -> float:
    """Compute simple accuracy from counts."""

    if total <= 0:
        return 0.0
    return correct / total


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Return AUROC for binary labels using average ranks."""

    if not scores or len(scores) != len(labels):
        return 0.0
    positives = sum(1 for label in labels if label == 1)
    negatives = sum(1 for label in labels if label == 0)
    if positives == 0 or negatives == 0:
        return 0.5

    indexed = sorted(enumerate(zip(scores, labels)), key=lambda item: item[1][0])
    ranks = [0.0 for _ in scores]
    index = 0
    while index < len(indexed):
        next_index = index + 1
        while next_index < len(indexed) and indexed[next_index][1][0] == indexed[index][1][0]:
            next_index += 1
        average_rank = (index + 1 + next_index) / 2.0
        for offset in range(index, next_index):
            original_index = indexed[offset][0]
            ranks[original_index] = average_rank
        index = next_index

    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    return (positive_rank_sum - (positives * (positives + 1) / 2.0)) / (positives * negatives)


def change_detection_auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Return AUROC for change-detection inference tasks."""

    return auroc(scores, labels)


def masked_state_inference_accuracy(values: Iterable[bool]) -> float:
    """Return accuracy on masked-state inference tasks."""

    return accuracy_from_flags(values)
