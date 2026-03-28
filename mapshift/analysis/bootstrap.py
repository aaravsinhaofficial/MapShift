"""Bootstrap analysis helpers for grouped MapShift outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

from mapshift.metrics.statistics import BootstrapInterval, bootstrap_by_units


@dataclass(frozen=True)
class BootstrappedMetricRow:
    """Bootstrap summary for one grouped metric."""

    group: dict[str, Any]
    metric_name: str
    summary: BootstrapInterval

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": dict(self.group),
            "metric_name": self.metric_name,
            "summary": self.summary.to_dict(),
        }


def bootstrap_grouped_metric(
    records: Sequence[Any],
    *,
    group_fields: Sequence[str],
    unit_field: str,
    metric_name: str,
    statistic: Callable[[Sequence[Any]], float],
    resamples: int,
    confidence_level: float,
    seed: int = 0,
) -> list[BootstrappedMetricRow]:
    """Bootstrap one grouped metric by resampling unique units within each group."""

    grouped: dict[tuple[str, ...], list[Any]] = {}
    for record in records:
        key = tuple(str(getattr(record, field)) for field in group_fields)
        grouped.setdefault(key, []).append(record)

    rows: list[BootstrappedMetricRow] = []
    for group_index, (key, group_records) in enumerate(sorted(grouped.items())):
        unit_ids = [str(getattr(record, unit_field)) for record in group_records]
        summary = bootstrap_by_units(
            group_records,
            unit_ids,
            statistic,
            resamples=resamples,
            confidence_level=confidence_level,
            seed=seed + group_index,
        )
        rows.append(
            BootstrappedMetricRow(
                group={field: value for field, value in zip(group_fields, key)},
                metric_name=metric_name,
                summary=summary,
            )
        )
    return rows


__all__ = ["BootstrappedMetricRow", "bootstrap_grouped_metric"]
