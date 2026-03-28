"""Planning metrics used by MapShift."""

from __future__ import annotations


def success_rate(successes: list[bool]) -> float:
    """Compute the fraction of successful planning episodes."""

    if not successes:
        return 0.0
    return sum(1 for value in successes if value) / len(successes)


def normalized_path_efficiency(optimal_length: float, observed_length: float) -> float:
    """Return path efficiency clipped to [0, 1]."""

    if optimal_length <= 0.0 or observed_length <= 0.0:
        return 0.0
    return max(0.0, min(1.0, optimal_length / observed_length))
