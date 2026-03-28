"""Ranking utilities for protocol comparison."""

from __future__ import annotations

from math import sqrt
from typing import Mapping, Sequence


def rank_by_metric(metric_values: Mapping[str, float], reverse: bool = True) -> list[str]:
    """Return item names sorted by metric value with deterministic tie breaks."""

    return [
        name
        for name, _value in sorted(
            metric_values.items(),
            key=lambda item: ((-item[1]) if reverse else item[1], item[0]),
        )
    ]


def kendall_tau(order_a: Sequence[str], order_b: Sequence[str]) -> float:
    """Return Kendall tau for two complete orders over the same items."""

    if len(order_a) != len(order_b) or set(order_a) != set(order_b):
        raise ValueError("kendall_tau requires the same items in both orderings")
    if len(order_a) < 2:
        return 1.0
    positions_a = {name: index for index, name in enumerate(order_a)}
    positions_b = {name: index for index, name in enumerate(order_b)}
    concordant = 0
    discordant = 0
    items = list(order_a)
    for left_index, left in enumerate(items):
        for right in items[left_index + 1 :]:
            delta_a = positions_a[left] - positions_a[right]
            delta_b = positions_b[left] - positions_b[right]
            if delta_a * delta_b > 0:
                concordant += 1
            elif delta_a * delta_b < 0:
                discordant += 1
    denominator = concordant + discordant
    if denominator == 0:
        return 1.0
    return (concordant - discordant) / denominator


def rank_positions(order: Sequence[str]) -> dict[str, int]:
    """Return one-based positions for an ordering."""

    return {name: index + 1 for index, name in enumerate(order)}


def ranking_spread(metric_values: Mapping[str, float]) -> float:
    """Return max-min spread for ranked metrics."""

    if not metric_values:
        return 0.0
    values = list(metric_values.values())
    return max(values) - min(values)


def ranking_stddev(metric_values: Mapping[str, float]) -> float:
    """Return standard deviation across metric values."""

    values = list(metric_values.values())
    if len(values) < 2:
        return 0.0
    mean_value = sum(values) / len(values)
    return sqrt(sum((value - mean_value) ** 2 for value in values) / len(values))


def rank_reversals(order_a: Sequence[str], order_b: Sequence[str]) -> list[dict[str, str | int]]:
    """Return pairwise reversals between two orderings."""

    if len(order_a) != len(order_b) or set(order_a) != set(order_b):
        raise ValueError("rank_reversals requires the same items in both orderings")
    positions_a = {name: index for index, name in enumerate(order_a)}
    positions_b = {name: index for index, name in enumerate(order_b)}
    reversals: list[dict[str, str | int]] = []
    items = list(order_a)
    for left_index, left in enumerate(items):
        for right in items[left_index + 1 :]:
            if (positions_a[left] - positions_a[right]) * (positions_b[left] - positions_b[right]) < 0:
                reversals.append(
                    {
                        "left": left,
                        "right": right,
                        "order_a_left_rank": positions_a[left] + 1,
                        "order_a_right_rank": positions_a[right] + 1,
                        "order_b_left_rank": positions_b[left] + 1,
                        "order_b_right_rank": positions_b[right] + 1,
                    }
                )
    return reversals
