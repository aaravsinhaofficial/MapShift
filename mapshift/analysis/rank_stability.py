"""Rank-stability analysis scaffold."""

from __future__ import annotations

from typing import Dict

from mapshift.metrics.ranking import rank_by_metric


def rank_summary(metric_values: Dict[str, float]) -> list[str]:
    """Return a ranked summary for the supplied metric values."""

    return rank_by_metric(metric_values)
