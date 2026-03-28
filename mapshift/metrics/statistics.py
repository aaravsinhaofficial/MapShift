"""Statistical helpers for MapShift analyses."""

from __future__ import annotations

import random
from statistics import mean


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
