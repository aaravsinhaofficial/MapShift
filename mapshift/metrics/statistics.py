"""Statistical helpers for MapShift analyses."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from statistics import mean, median
from typing import Callable, Iterable, Sequence, TypeVar

T = TypeVar("T")


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


@dataclass(frozen=True)
class BootstrapInterval:
    """Percentile bootstrap summary for one scalar statistic."""

    point_estimate: float
    lower: float
    upper: float
    resamples: int
    confidence_level: float
    sample_count: int

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


def stddev_or_zero(values: Sequence[float | int]) -> float:
    """Return population standard deviation or zero if empty."""

    if len(values) < 2:
        return 0.0
    numeric_values = [float(value) for value in values]
    average = sum(numeric_values) / len(numeric_values)
    return math.sqrt(sum((value - average) ** 2 for value in numeric_values) / len(numeric_values))


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


def percentile_interval(samples: Sequence[float], confidence_level: float = 0.95) -> tuple[float, float]:
    """Return percentile interval bounds from sorted or unsorted samples."""

    if not samples:
        return 0.0, 0.0
    ordered = sorted(float(sample) for sample in samples)
    lower_index = int(((1.0 - confidence_level) / 2.0) * (len(ordered) - 1))
    upper_index = int(((1.0 + confidence_level) / 2.0) * (len(ordered) - 1))
    return ordered[lower_index], ordered[upper_index]


def bootstrap_statistic(
    values: Sequence[T],
    statistic: Callable[[Sequence[T]], float],
    *,
    resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> BootstrapInterval:
    """Return a percentile bootstrap summary for an arbitrary statistic."""

    if not values:
        return BootstrapInterval(
            point_estimate=0.0,
            lower=0.0,
            upper=0.0,
            resamples=resamples,
            confidence_level=confidence_level,
            sample_count=0,
        )

    rng = random.Random(seed)
    samples = []
    for _ in range(resamples):
        draw = [values[rng.randrange(len(values))] for _ in range(len(values))]
        samples.append(float(statistic(draw)))
    lower, upper = percentile_interval(samples, confidence_level=confidence_level)
    return BootstrapInterval(
        point_estimate=float(statistic(values)),
        lower=lower,
        upper=upper,
        resamples=resamples,
        confidence_level=confidence_level,
        sample_count=len(values),
    )


def bootstrap_by_units(
    values: Sequence[T],
    unit_ids: Sequence[str],
    statistic: Callable[[Sequence[T]], float],
    *,
    resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> BootstrapInterval:
    """Bootstrap a statistic by resampling grouped units with replacement."""

    if not values or not unit_ids or len(values) != len(unit_ids):
        return BootstrapInterval(
            point_estimate=0.0,
            lower=0.0,
            upper=0.0,
            resamples=resamples,
            confidence_level=confidence_level,
            sample_count=0,
        )

    grouped: dict[str, list[T]] = {}
    for value, unit_id in zip(values, unit_ids):
        grouped.setdefault(str(unit_id), []).append(value)
    unique_units = sorted(grouped)
    point_estimate = float(statistic(values))
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(resamples):
        draw: list[T] = []
        for _unit_index in range(len(unique_units)):
            sampled_unit = unique_units[rng.randrange(len(unique_units))]
            draw.extend(grouped[sampled_unit])
        samples.append(float(statistic(draw)))
    lower, upper = percentile_interval(samples, confidence_level=confidence_level)
    return BootstrapInterval(
        point_estimate=point_estimate,
        lower=lower,
        upper=upper,
        resamples=resamples,
        confidence_level=confidence_level,
        sample_count=len(unique_units),
    )


def bootstrap_mean_interval(values: list[float], resamples: int = 1000, confidence_level: float = 0.95, seed: int = 0) -> tuple[float, float, float]:
    """Return mean and a simple percentile bootstrap interval."""

    summary = bootstrap_statistic(values, mean_or_zero, resamples=resamples, confidence_level=confidence_level, seed=seed)
    return summary.point_estimate, summary.lower, summary.upper
