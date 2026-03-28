"""Severity-response and coverage utilities for MapShift."""

from __future__ import annotations

from itertools import product
from typing import Any, Sequence

from mapshift.core.schemas import FamilyInterventionConfig


def monotone_degradation(values: Sequence[float]) -> bool:
    """Check whether a sequence is monotonically non-increasing."""

    return all(left >= right for left, right in zip(values, values[1:]))


def severity_deltas_from_noop(family_config: FamilyInterventionConfig) -> list[float]:
    """Return absolute severity deltas from the severity-0 value."""

    base = family_config.severity_levels["0"].value
    return [abs(family_config.severity_levels[str(level)].value - base) for level in range(4)]


def severity_is_monotone(family_config: FamilyInterventionConfig) -> bool:
    """Return whether severity magnitude is monotone away from no-op."""

    deltas = severity_deltas_from_noop(family_config)
    return all(left <= right for left, right in zip(deltas, deltas[1:]))


def summarize_family_severity_counts(records: Sequence[dict[str, Any]], key: str = "count") -> dict[str, dict[str, int]]:
    """Aggregate records into a family x severity table."""

    table: dict[str, dict[str, int]] = {}
    for record in records:
        family = str(record["family"])
        severity = str(record["severity"])
        value = int(record.get(key, 1))
        table.setdefault(family, {})
        table[family][severity] = table[family].get(severity, 0) + value
    return {family: dict(sorted(severity_counts.items(), key=lambda item: int(item[0]))) for family, severity_counts in sorted(table.items())}


def find_undercovered_cells(
    records: Sequence[dict[str, Any]],
    dimensions: Sequence[str],
    min_coverage: int = 1,
    expected_values: dict[str, Sequence[str | int]] | None = None,
) -> list[dict[str, Any]]:
    """Return records whose count falls below the minimum required coverage."""

    counts: dict[tuple[str, ...], int] = {}
    for record in records:
        key = tuple(str(record[dimension]) for dimension in dimensions)
        counts[key] = counts.get(key, 0) + int(record.get("count", 1))

    undercovered = []
    keys = set(counts)
    if expected_values is not None:
        missing_dimensions = [dimension for dimension in dimensions if dimension not in expected_values]
        if missing_dimensions:
            raise ValueError(f"Missing expected coverage values for dimensions: {missing_dimensions}")
        expected_keys = set(
            tuple(str(value) for value in combination)
            for combination in product(*(expected_values[dimension] for dimension in dimensions))
        )
        keys |= expected_keys

    for key in sorted(keys):
        count = counts.get(key, 0)
        if count < min_coverage:
            undercovered.append({dimension: value for dimension, value in zip(dimensions, key)} | {"count": count})
    return undercovered
