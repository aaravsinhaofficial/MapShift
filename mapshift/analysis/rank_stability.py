"""Rank-stability analysis helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence

from mapshift.metrics.ranking import rank_by_metric, rank_positions, ranking_spread
from mapshift.metrics.statistics import mean_or_zero


@dataclass(frozen=True)
class RankStabilitySummary:
    """Bootstrap-style rank stability summary for one protocol/family slice."""

    protocol_name: str
    family: str
    order: tuple[str, ...]
    spread: float
    mean_rank: dict[str, float]
    top1_frequency: dict[str, float]
    position_distributions: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def rank_summary(metric_values: dict[str, float]) -> list[str]:
    """Return a ranked summary for the supplied metric values."""

    return rank_by_metric(metric_values)


def summarize_rank_samples(
    *,
    protocol_name: str,
    family: str,
    point_estimates: dict[str, float],
    sampled_orders: Sequence[Sequence[str]],
) -> RankStabilitySummary:
    """Summarize rank distributions from sampled orders."""

    order = tuple(rank_by_metric(point_estimates))
    methods = sorted(point_estimates)
    position_counts: dict[str, dict[int, int]] = {method: {} for method in methods}
    for sampled_order in sampled_orders:
        positions = rank_positions(sampled_order)
        for method, position in positions.items():
            position_counts[method][position] = position_counts[method].get(position, 0) + 1

    order_count = max(1, len(sampled_orders))
    top1_frequency = {
        method: position_counts[method].get(1, 0) / order_count
        for method in methods
    }
    mean_rank = {
        method: mean_or_zero(
            [
                position
                for position, count in position_counts[method].items()
                for _ in range(count)
            ]
        )
        for method in methods
    }
    distributions = {
        method: {
            str(position): count / order_count
            for position, count in sorted(position_counts[method].items())
        }
        for method in methods
    }
    return RankStabilitySummary(
        protocol_name=protocol_name,
        family=family,
        order=order,
        spread=ranking_spread(point_estimates),
        mean_rank=mean_rank,
        top1_frequency=top1_frequency,
        position_distributions=distributions,
    )
