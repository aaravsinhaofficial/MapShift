"""Mechanism-diagnostic analyses for stale reuse versus belief update."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from mapshift.metrics.statistics import percentile_interval
from mapshift.runners.evaluate import EvaluationRecord, _long_horizon_threshold, _metric_row


DEFAULT_BELIEF_UPDATE_BASELINE = "classical_belief_update_planner"
DEFAULT_STALE_MAP_BASELINE = "stale_map_planner"
DEFAULT_CEP_PROTOCOL = "cep"
DEFAULT_SAME_ENV_PROTOCOL = "same_environment"
DEFAULT_FAMILIES = ("topology", "semantic")


@dataclass(frozen=True)
class MechanismDiagnosticConfig:
    """Configuration for held-out stale-map versus belief-update analyses."""

    belief_update_baseline: str = DEFAULT_BELIEF_UPDATE_BASELINE
    stale_map_baseline: str = DEFAULT_STALE_MAP_BASELINE
    cep_protocol: str = DEFAULT_CEP_PROTOCOL
    same_environment_protocol: str = DEFAULT_SAME_ENV_PROTOCOL
    split_name: str = "test"
    families: tuple[str, ...] = DEFAULT_FAMILIES
    exclude_identity: bool = True
    substantial_delta_threshold: float = 0.05
    bootstrap_resamples: int = 1000
    confidence_level: float = 0.95
    bootstrap_unit_field: str = "base_environment_id"
    bootstrap_seed: int = 17


def analyze_mechanism_diagnostic_bundle(
    bundle_payload: Mapping[str, Any],
    *,
    config: MechanismDiagnosticConfig | None = None,
) -> dict[str, Any]:
    """Build P2/P3 analyses from a MapShift study bundle or protocol report payload."""

    active_config = config or MechanismDiagnosticConfig()
    records = records_from_payload(bundle_payload)
    selected_records = _filter_records(
        records,
        split_name=active_config.split_name,
        families=active_config.families,
        exclude_identity=active_config.exclude_identity,
    )
    heldout_rows = _heldout_consistency_rows(selected_records, active_config)
    heldout_summary = _summarize_heldout_rows(heldout_rows, active_config)
    bootstrap_rows = _paired_bootstrap_rows(selected_records, active_config)
    return {
        "analysis_name": "mechanism_diagnostic_p2_p3",
        "config": {
            "belief_update_baseline": active_config.belief_update_baseline,
            "stale_map_baseline": active_config.stale_map_baseline,
            "cep_protocol": active_config.cep_protocol,
            "same_environment_protocol": active_config.same_environment_protocol,
            "split_name": active_config.split_name,
            "families": list(active_config.families),
            "exclude_identity": active_config.exclude_identity,
            "substantial_delta_threshold": active_config.substantial_delta_threshold,
            "bootstrap_resamples": active_config.bootstrap_resamples,
            "confidence_level": active_config.confidence_level,
            "bootstrap_unit_field": active_config.bootstrap_unit_field,
            "bootstrap_seed": active_config.bootstrap_seed,
        },
        "record_count": len(selected_records),
        "heldout_motif_consistency": {
            "rows": heldout_rows,
            "summary_by_family": heldout_summary,
        },
        "paired_delta_bootstrap": {
            "rows": bootstrap_rows,
        },
    }


def records_from_payload(payload: Mapping[str, Any]) -> list[EvaluationRecord]:
    """Extract evaluation records from a study bundle, protocol report, or calibration report."""

    raw_reports = payload.get("raw_reports")
    if isinstance(raw_reports, Mapping):
        protocol_payload = raw_reports.get("protocol_comparison_report")
        if isinstance(protocol_payload, Mapping) and protocol_payload.get("protocol_reports"):
            return records_from_payload(protocol_payload)
        cep_payload = raw_reports.get("cep_report")
        if isinstance(cep_payload, Mapping):
            return records_from_payload(cep_payload)

    protocol_reports = payload.get("protocol_reports")
    if isinstance(protocol_reports, Mapping):
        records: list[EvaluationRecord] = []
        for protocol_name in sorted(protocol_reports):
            report = protocol_reports[protocol_name]
            if isinstance(report, Mapping):
                records.extend(_record_from_dict(item) for item in report.get("records", []))
        return records

    return [_record_from_dict(item) for item in payload.get("records", [])]


def render_heldout_consistency_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render held-out motif consistency rows as a compact Markdown table."""

    headers = [
        "Motif",
        "Family",
        "BU-STM CEP",
        "BU-STM Same-env",
        "STM CEP-Same",
        "BU CEP-Same",
        "Reversal/reduction",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["motif_tag"]),
                    str(row["family"]),
                    _fmt(row["belief_update_minus_stale_cep"]),
                    _fmt(row["belief_update_minus_stale_same_environment"]),
                    _fmt(row["stale_cep_minus_same_environment"]),
                    _fmt(row["belief_update_cep_minus_same_environment"]),
                    "yes" if row["reversal_or_substantial_reduction"] else "no",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def render_heldout_summary_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render held-out motif summary rows as Markdown."""

    headers = [
        "Family",
        "Motifs",
        "Mean BU-STM CEP",
        "Mean BU-STM Same-env",
        "Mean protocol delta",
        "BU wins CEP",
        "Reversal/reduction",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["family"]),
                    str(row["motif_count"]),
                    _fmt(row["mean_belief_update_minus_stale_cep"]),
                    _fmt(row["mean_belief_update_minus_stale_same_environment"]),
                    _fmt(row["mean_protocol_reversal_delta"]),
                    f"{row['belief_update_beats_stale_under_cep_count']}/{row['motif_count']}",
                    f"{row['reversal_or_substantial_reduction_count']}/{row['motif_count']}",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def render_paired_bootstrap_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render paired bootstrap rows as Markdown."""

    headers = ["Family", "Contrast", "Point", "95% CI", "Units", "Records"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["family"]),
                    str(row["contrast"]),
                    _fmt(row["point_estimate"]),
                    f"[{_fmt(row['lower'])}, {_fmt(row['upper'])}]",
                    str(row["unit_count"]),
                    str(row["record_count"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _record_from_dict(payload: Mapping[str, Any]) -> EvaluationRecord:
    record = dict(payload)
    record["adaptation_curve"] = tuple(record.get("adaptation_curve", ()))
    record.setdefault("metadata", {})
    return EvaluationRecord(**record)


def _filter_records(
    records: Sequence[EvaluationRecord],
    *,
    split_name: str,
    families: Sequence[str],
    exclude_identity: bool,
) -> list[EvaluationRecord]:
    family_set = set(families)
    return [
        record
        for record in records
        if (split_name == "all" or record.split_name == split_name)
        and record.family in family_set
        and (not exclude_identity or int(record.severity) > 0)
    ]


def _heldout_consistency_rows(
    records: Sequence[EvaluationRecord],
    config: MechanismDiagnosticConfig,
) -> list[dict[str, Any]]:
    motifs = sorted({record.motif_tag for record in records})
    rows: list[dict[str, Any]] = []
    for motif_tag in motifs:
        for family in config.families:
            motif_family_records = [record for record in records if record.motif_tag == motif_tag and record.family == family]
            if not motif_family_records:
                continue
            bu_cep = _score(motif_family_records, config.belief_update_baseline, config.cep_protocol, family)
            stale_cep = _score(motif_family_records, config.stale_map_baseline, config.cep_protocol, family)
            bu_same = _score(motif_family_records, config.belief_update_baseline, config.same_environment_protocol, family)
            stale_same = _score(motif_family_records, config.stale_map_baseline, config.same_environment_protocol, family)

            delta_cep = bu_cep - stale_cep
            delta_same = bu_same - stale_same
            protocol_delta = delta_cep - delta_same
            stale_advantage_same = stale_same - bu_same
            stale_advantage_cep = stale_cep - bu_cep
            rows.append(
                {
                    "split_name": config.split_name,
                    "motif_tag": motif_tag,
                    "family": family,
                    "belief_update_cep": bu_cep,
                    "stale_map_cep": stale_cep,
                    "belief_update_same_environment": bu_same,
                    "stale_map_same_environment": stale_same,
                    "belief_update_minus_stale_cep": delta_cep,
                    "belief_update_minus_stale_same_environment": delta_same,
                    "stale_cep_minus_same_environment": stale_cep - stale_same,
                    "belief_update_cep_minus_same_environment": bu_cep - bu_same,
                    "protocol_reversal_delta": protocol_delta,
                    "belief_update_beats_stale_under_cep": delta_cep > 0.0,
                    "same_environment_ranks_stale_at_or_above_belief_update": stale_advantage_same >= -config.substantial_delta_threshold,
                    "reversal_or_substantial_reduction": stale_advantage_cep
                    < stale_advantage_same - config.substantial_delta_threshold,
                    "episode_count": len(motif_family_records),
                }
            )
    return rows


def _summarize_heldout_rows(
    rows: Sequence[Mapping[str, Any]],
    config: MechanismDiagnosticConfig,
) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for family in config.families:
        family_rows = [row for row in rows if row["family"] == family]
        if not family_rows:
            continue
        summary_rows.append(
            {
                "split_name": config.split_name,
                "family": family,
                "motif_count": len({str(row["motif_tag"]) for row in family_rows}),
                "mean_belief_update_minus_stale_cep": _mean(row["belief_update_minus_stale_cep"] for row in family_rows),
                "mean_belief_update_minus_stale_same_environment": _mean(
                    row["belief_update_minus_stale_same_environment"] for row in family_rows
                ),
                "mean_protocol_reversal_delta": _mean(row["protocol_reversal_delta"] for row in family_rows),
                "belief_update_beats_stale_under_cep_count": sum(
                    1 for row in family_rows if row["belief_update_beats_stale_under_cep"]
                ),
                "same_environment_ranks_stale_at_or_above_belief_update_count": sum(
                    1 for row in family_rows if row["same_environment_ranks_stale_at_or_above_belief_update"]
                ),
                "reversal_or_substantial_reduction_count": sum(
                    1 for row in family_rows if row["reversal_or_substantial_reduction"]
                ),
            }
        )
    return summary_rows


def _paired_bootstrap_rows(
    records: Sequence[EvaluationRecord],
    config: MechanismDiagnosticConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    contrast_specs = [
        ("belief_update_minus_stale_cep", ((config.belief_update_baseline, config.cep_protocol, 1.0), (config.stale_map_baseline, config.cep_protocol, -1.0))),
        (
            "belief_update_minus_stale_same_environment",
            (
                (config.belief_update_baseline, config.same_environment_protocol, 1.0),
                (config.stale_map_baseline, config.same_environment_protocol, -1.0),
            ),
        ),
        (
            "protocol_reversal_delta",
            (
                (config.belief_update_baseline, config.cep_protocol, 1.0),
                (config.stale_map_baseline, config.cep_protocol, -1.0),
                (config.belief_update_baseline, config.same_environment_protocol, -1.0),
                (config.stale_map_baseline, config.same_environment_protocol, 1.0),
            ),
        ),
    ]
    for family_index, family in enumerate(config.families):
        family_records = [record for record in records if record.family == family]
        for contrast_index, (contrast_name, terms) in enumerate(contrast_specs):
            complete_records = _complete_unit_records(
                family_records,
                unit_field=config.bootstrap_unit_field,
                required_terms=terms,
            )
            summary = _bootstrap_contrast(
                complete_records,
                family=family,
                terms=terms,
                unit_field=config.bootstrap_unit_field,
                resamples=config.bootstrap_resamples,
                confidence_level=config.confidence_level,
                seed=config.bootstrap_seed + 100 * family_index + contrast_index,
            )
            rows.append(
                {
                    "split_name": config.split_name,
                    "family": family,
                    "contrast": contrast_name,
                    "formula": _contrast_formula(terms),
                    "point_estimate": summary["point_estimate"],
                    "lower": summary["lower"],
                    "upper": summary["upper"],
                    "confidence_level": config.confidence_level,
                    "resamples": config.bootstrap_resamples,
                    "bootstrap_unit_field": config.bootstrap_unit_field,
                    "unit_count": summary["unit_count"],
                    "record_count": len(complete_records),
                    "complete_unit_count": summary["unit_count"],
                }
            )
    return rows


def _complete_unit_records(
    records: Sequence[EvaluationRecord],
    *,
    unit_field: str,
    required_terms: Sequence[tuple[str, str, float]],
) -> list[EvaluationRecord]:
    required_cells = {(baseline, protocol) for baseline, protocol, _weight in required_terms}
    grouped: dict[str, list[EvaluationRecord]] = {}
    for record in records:
        grouped.setdefault(str(getattr(record, unit_field)), []).append(record)
    complete_units = {
        unit_id
        for unit_id, unit_records in grouped.items()
        if required_cells.issubset({(record.baseline_name, record.protocol_name) for record in unit_records})
    }
    return [record for record in records if str(getattr(record, unit_field)) in complete_units]


def _bootstrap_contrast(
    records: Sequence[EvaluationRecord],
    *,
    family: str,
    terms: Sequence[tuple[str, str, float]],
    unit_field: str,
    resamples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, float | int]:
    if not records:
        return {"point_estimate": 0.0, "lower": 0.0, "upper": 0.0, "unit_count": 0}
    grouped: dict[str, list[EvaluationRecord]] = {}
    for record in records:
        grouped.setdefault(str(getattr(record, unit_field)), []).append(record)
    unit_ids = sorted(grouped)
    point = _contrast(records, family=family, terms=terms)
    rng = random.Random(seed)
    samples: list[float] = []
    for _sample_index in range(resamples):
        draw: list[EvaluationRecord] = []
        for _unit_index in range(len(unit_ids)):
            draw.extend(grouped[unit_ids[rng.randrange(len(unit_ids))]])
        samples.append(_contrast(draw, family=family, terms=terms))
    lower, upper = percentile_interval(samples, confidence_level=confidence_level)
    return {
        "point_estimate": point,
        "lower": lower,
        "upper": upper,
        "unit_count": len(unit_ids),
    }


def _contrast(
    records: Sequence[EvaluationRecord],
    *,
    family: str,
    terms: Sequence[tuple[str, str, float]],
) -> float:
    return sum(weight * _score(records, baseline, protocol, family) for baseline, protocol, weight in terms)


def _score(records: Sequence[EvaluationRecord], baseline_name: str, protocol_name: str, family: str) -> float:
    selected = [
        record
        for record in records
        if record.baseline_name == baseline_name
        and record.protocol_name == protocol_name
        and record.family == family
    ]
    if not selected:
        return 0.0
    return float(_metric_row(selected, long_horizon_threshold=_long_horizon_threshold(selected))["family_primary_score"])


def _contrast_formula(terms: Sequence[tuple[str, str, float]]) -> str:
    pieces = []
    for baseline, protocol, weight in terms:
        sign = "+" if weight > 0 else "-"
        pieces.append(f"{sign} {baseline}@{protocol}")
    return " ".join(pieces).lstrip("+ ")


def _mean(values: Iterable[Any]) -> float:
    numeric = [float(value) for value in values]
    return sum(numeric) / len(numeric) if numeric else 0.0


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


__all__ = [
    "MechanismDiagnosticConfig",
    "analyze_mechanism_diagnostic_bundle",
    "records_from_payload",
    "render_heldout_consistency_markdown",
    "render_heldout_summary_markdown",
    "render_paired_bootstrap_markdown",
]
