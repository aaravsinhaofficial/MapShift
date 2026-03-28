"""Ranking utilities for protocol comparison."""

from __future__ import annotations

from typing import Dict


def rank_by_metric(metric_values: Dict[str, float], reverse: bool = True) -> list[str]:
    """Return item names sorted by metric value."""

    return [name for name, _value in sorted(metric_values.items(), key=lambda item: item[1], reverse=reverse)]


def kendall_tau_placeholder(order_a: list[str], order_b: list[str]) -> float:
    """Return 1.0 for exact agreement, 0.0 otherwise, until a full implementation lands."""

    return 1.0 if order_a == order_b else 0.0
