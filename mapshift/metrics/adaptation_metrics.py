"""Adaptation metrics used by MapShift."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass(frozen=True)
class AdaptationCurveSummary:
    """Compact summary of a recovery curve."""

    start: float
    end: float
    improvement: float
    area_under_curve: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def adaptation_sample_efficiency(success_curve: Sequence[float]) -> float:
    """A simple area-under-curve proxy for adaptation efficiency."""

    if not success_curve:
        return 0.0
    return sum(float(value) for value in success_curve) / len(success_curve)


def summarize_adaptation_curve(success_curve: Sequence[float]) -> AdaptationCurveSummary:
    """Return a deterministic summary of one adaptation curve."""

    if not success_curve:
        return AdaptationCurveSummary(start=0.0, end=0.0, improvement=0.0, area_under_curve=0.0)
    start = float(success_curve[0])
    end = float(success_curve[-1])
    area = adaptation_sample_efficiency(success_curve)
    return AdaptationCurveSummary(
        start=start,
        end=end,
        improvement=end - start,
        area_under_curve=area,
    )


def recovery_vs_budget(curves: Sequence[Sequence[float]]) -> list[dict[str, float]]:
    """Aggregate recovery performance as a function of adaptation step index."""

    if not curves:
        return []
    max_length = max(len(curve) for curve in curves)
    rows: list[dict[str, float]] = []
    for step_index in range(max_length):
        values = [float(curve[step_index]) for curve in curves if step_index < len(curve)]
        mean_value = sum(values) / len(values) if values else 0.0
        rows.append({"budget_index": float(step_index), "mean_performance": mean_value})
    return rows
