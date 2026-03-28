"""Statistical helpers for MapShift analyses."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from statistics import mean, median
from typing import Iterable, Sequence


@dataclass(frozen=True)
class NumericSummary:
    """Compact summary for a numeric distribution."""

    count: int
    min: float
    max: float
    mean: float
    median: float
    sum: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def summarize_numeric(values: Sequence[float | int]) -> NumericSummary:
    """Return a deterministic numeric summary."""

    if not values:
        return NumericSummary(count=0, min=0.0, max=0.0, mean=0.0, median=0.0, sum=0.0)

    numeric_values = [float(value) for value in values]
    return NumericSummary(
        count=len(numeric_values),
        min=min(numeric_values),
        max=max(numeric_values),
        mean=mean(numeric_values),
        median=median(numeric_values),
        sum=sum(numeric_values),
    )


def proportion_true(values: Iterable[bool]) -> float:
    """Return the fraction of truthy values in an iterable."""

    values_list = list(values)
    if not values_list:
        return 0.0
    return sum(1 for value in values_list if value) / len(values_list)


def mean_or_zero(values: Sequence[float | int]) -> float:
    """Return the mean of values or zero if empty."""

    if not values:
        return 0.0
    return mean(float(value) for value in values)


def histogram_counts(values: Sequence[float | int], bin_edges: Sequence[float | int]) -> dict[str, int]:
    """Return histogram counts for deterministic bin edges."""

    counts: dict[str, int] = {}
    if len(bin_edges) < 2:
        return counts

    numeric_values = [float(value) for value in values]
    edges = [float(edge) for edge in bin_edges]

    for left, right in zip(edges, edges[1:]):
        label = f"[{left:g}, {right:g})"
        counts[label] = 0

    for value in numeric_values:
        for index, (left, right) in enumerate(zip(edges, edges[1:])):
            is_last = index == len(edges) - 2
            if (left <= value < right) or (is_last and math.isclose(value, right)):
                label = f"[{left:g}, {right:g})"
                counts[label] += 1
                break

    return counts


def bootstrap_mean_interval(values: list[float], resamples: int = 1000, confidence_level: float = 0.95, seed: int = 0) -> tuple[float, float, float]:
    """Return mean and a simple percentile bootstrap interval."""

    if not values:
        return 0.0, 0.0, 0.0

    rng = random.Random(seed)
    samples = []
    for _ in range(resamples):
        draw = [values[rng.randrange(len(values))] for _ in range(len(values))]
        samples.append(mean(draw))
    samples.sort()

    lower_index = int(((1.0 - confidence_level) / 2.0) * (len(samples) - 1))
    upper_index = int(((1.0 + confidence_level) / 2.0) * (len(samples) - 1))
    return mean(values), samples[lower_index], samples[upper_index]
