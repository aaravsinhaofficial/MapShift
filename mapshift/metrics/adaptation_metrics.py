"""Adaptation metrics used by MapShift."""

from __future__ import annotations


def adaptation_sample_efficiency(success_curve: list[float]) -> float:
    """A simple area-under-curve proxy for adaptation efficiency."""

    if not success_curve:
        return 0.0
    return sum(success_curve) / len(success_curve)
