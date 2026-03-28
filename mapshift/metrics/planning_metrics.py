"""Planning metrics used by MapShift."""

from __future__ import annotations

from typing import Iterable, Sequence


def success_rate(successes: Iterable[bool]) -> float:
    """Compute the fraction of successful planning episodes."""

    values = list(successes)
    if not values:
        return 0.0
    return sum(1 for value in values if value) / len(values)


def normalized_path_efficiency(optimal_length: float, observed_length: float) -> float:
    """Return path efficiency clipped to [0, 1]."""

    if optimal_length <= 0.0 or observed_length <= 0.0:
        return 0.0
    return max(0.0, min(1.0, optimal_length / observed_length))


def mean_normalized_path_efficiency(optimal_lengths: Sequence[float | None], observed_lengths: Sequence[float | None]) -> float:
    """Return mean normalized path efficiency across paired path lengths."""

    efficiencies = [
        normalized_path_efficiency(float(optimal), float(observed))
        for optimal, observed in zip(optimal_lengths, observed_lengths)
        if optimal is not None and observed is not None and optimal > 0.0 and observed > 0.0
    ]
    if not efficiencies:
        return 0.0
    return sum(efficiencies) / len(efficiencies)


def oracle_gap(observed_length: float | None, optimal_length: float | None) -> float | None:
    """Return the non-negative excess cost over oracle."""

    if observed_length is None or optimal_length is None:
        return None
    return max(0.0, observed_length - optimal_length)


def counterfactual_planning_accuracy(successes: Iterable[bool]) -> float:
    """Return success rate for counterfactual planning tasks."""

    return success_rate(successes)


def long_horizon_rollout_consistency(path_efficiencies: Sequence[float], successes: Sequence[bool]) -> float:
    """Return a simple long-horizon consistency score in [0, 1]."""

    if not path_efficiencies or not successes:
        return 0.0
    paired = list(zip(path_efficiencies, successes))
    if not paired:
        return 0.0
    return sum(max(0.0, min(1.0, efficiency)) if success else 0.0 for efficiency, success in paired) / len(paired)
