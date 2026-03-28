"""Protocol comparison helpers for MapShift."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from mapshift.core.schemas import ReleaseBundle
from mapshift.metrics.ranking import kendall_tau, rank_reversals
from mapshift.runners.evaluate import CalibrationReport, EvaluationProtocol, run_calibration_suite


@dataclass(frozen=True)
class ProtocolComparisonReport:
    """Machine-readable comparison artifact across evaluation protocols."""

    release_name: str
    benchmark_version: str
    protocol_reports: dict[str, dict[str, Any]]
    pairwise_comparisons: dict[str, Any]
    pooled_vs_familywise: dict[str, Any]
    report_artifacts: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_PROTOCOLS = {
    "cep": EvaluationProtocol(name="cep", environment_mode="post_intervention", exploration_mode="reward_free", horizon_multiplier=1.0),
    "same_environment": EvaluationProtocol(name="same_environment", environment_mode="same_environment", exploration_mode="reward_free", horizon_multiplier=1.0),
    "no_exploration": EvaluationProtocol(name="no_exploration", environment_mode="post_intervention", exploration_mode="none", horizon_multiplier=1.0),
    "short_horizon": EvaluationProtocol(name="short_horizon", environment_mode="post_intervention", exploration_mode="reward_free", horizon_multiplier=0.5),
    "long_horizon": EvaluationProtocol(name="long_horizon", environment_mode="post_intervention", exploration_mode="reward_free", horizon_multiplier=2.0),
}


def run_protocol_comparison_suite(
    release_bundle: ReleaseBundle,
    baseline_run_configs: Sequence[str | Path],
    *,
    sample_count_per_motif: int = 1,
    task_samples_per_class: int = 1,
    severity_levels: Sequence[int] = (0, 1, 2, 3),
    motif_tags: Sequence[str] | None = None,
    family_names: Sequence[str] | None = None,
    protocol_names: Sequence[str] | None = None,
) -> ProtocolComparisonReport:
    """Run the configured protocol comparisons for the current release bundle."""

    protocol_reports: dict[str, CalibrationReport] = {}
    required_protocols = tuple(protocol_names or ("cep", "same_environment", "no_exploration", "short_horizon", "long_horizon"))
    for protocol_name in sorted(required_protocols):
        protocol_reports[protocol_name] = run_calibration_suite(
            release_bundle=release_bundle,
            baseline_run_configs=baseline_run_configs,
            sample_count_per_motif=sample_count_per_motif,
            task_samples_per_class=task_samples_per_class,
            severity_levels=severity_levels,
            protocol=DEFAULT_PROTOCOLS[protocol_name],
            motif_tags=motif_tags,
            family_names=family_names,
        )

    pairwise = {}
    if "same_environment" in protocol_reports and "cep" in protocol_reports:
        pairwise["same_environment_vs_cep"] = _compare_two_reports(protocol_reports["same_environment"], protocol_reports["cep"])
    if "no_exploration" in protocol_reports and "cep" in protocol_reports:
        pairwise["no_exploration_vs_reward_free_exploration"] = _compare_two_reports(protocol_reports["no_exploration"], protocol_reports["cep"])
    if "short_horizon" in protocol_reports and "long_horizon" in protocol_reports:
        pairwise["short_horizon_vs_long_horizon"] = _compare_two_reports(protocol_reports["short_horizon"], protocol_reports["long_horizon"])
    pooled_vs_familywise = _compare_pooled_vs_familywise(protocol_reports["cep"]) if "cep" in protocol_reports else {}
    protocol_payloads = {name: report.to_dict() for name, report in protocol_reports.items()}
    return ProtocolComparisonReport(
        release_name=release_bundle.root.release_name,
        benchmark_version=release_bundle.root.benchmark_version,
        protocol_reports=protocol_payloads,
        pairwise_comparisons=pairwise,
        pooled_vs_familywise=pooled_vs_familywise,
        report_artifacts={
            "tables": {
                "protocol_sensitivity_and_rank_correlation": pairwise,
                "pooled_vs_familywise_reporting": pooled_vs_familywise,
            },
            "figures": {
                "protocol_ranking_comparison": pairwise,
            },
        },
    )


def compare_protocol_outputs(outputs: dict[str, CalibrationReport]) -> dict[str, Any]:
    """Return pairwise comparisons for externally prepared protocol reports."""

    comparisons = {}
    if "same_environment" in outputs and "cep" in outputs:
        comparisons["same_environment_vs_cep"] = _compare_two_reports(outputs["same_environment"], outputs["cep"])
    if "no_exploration" in outputs and "cep" in outputs:
        comparisons["no_exploration_vs_reward_free_exploration"] = _compare_two_reports(outputs["no_exploration"], outputs["cep"])
    if "short_horizon" in outputs and "long_horizon" in outputs:
        comparisons["short_horizon_vs_long_horizon"] = _compare_two_reports(outputs["short_horizon"], outputs["long_horizon"])
    if "cep" in outputs:
        comparisons["pooled_vs_familywise_reporting"] = _compare_pooled_vs_familywise(outputs["cep"])
    return comparisons


def _compare_two_reports(left: CalibrationReport, right: CalibrationReport) -> dict[str, Any]:
    left_order = list(left.ranking_summary["pooled_supplementary_orders"][0]["order"])
    right_order = list(right.ranking_summary["pooled_supplementary_orders"][0]["order"])
    left_best = left_order[0] if left_order else ""
    right_best = right_order[0] if right_order else ""
    left_family_orders = {
        str(row["family"]): list(row["order"])
        for row in left.ranking_summary["family_orders"]
        if row["protocol_name"] == left.protocol_name
    }
    right_family_orders = {
        str(row["family"]): list(row["order"])
        for row in right.ranking_summary["family_orders"]
        if row["protocol_name"] == right.protocol_name
    }
    family_rank_changes = []
    for family in sorted(set(left_family_orders) & set(right_family_orders)):
        if left_family_orders[family] != right_family_orders[family]:
            family_rank_changes.append(
                {
                    "family": family,
                    "left_order": left_family_orders[family],
                    "right_order": right_family_orders[family],
                    "kendall_tau": kendall_tau(left_family_orders[family], right_family_orders[family]),
                    "rank_reversals": rank_reversals(left_family_orders[family], right_family_orders[family]),
                }
            )
    return {
        "left_protocol": left.protocol_name,
        "right_protocol": right.protocol_name,
        "left_order": left_order,
        "right_order": right_order,
        "kendall_tau": kendall_tau(left_order, right_order) if left_order and right_order else 0.0,
        "best_method_changes": left_best != right_best,
        "left_best_method": left_best,
        "right_best_method": right_best,
        "rank_reversals": rank_reversals(left_order, right_order) if left_order and right_order else [],
        "family_rank_changes": family_rank_changes,
    }


def _compare_pooled_vs_familywise(report: CalibrationReport) -> dict[str, Any]:
    pooled_order = list(report.ranking_summary["pooled_supplementary_orders"][0]["order"])
    familywise_order = list(report.ranking_summary["average_family_rank_orders"][0]["order"])
    family_orders = report.ranking_summary["family_orders"]
    family_best_methods = {
        str(row["family"]): list(row["order"])[0]
        for row in family_orders
        if row["protocol_name"] == report.protocol_name and row["order"]
    }
    pooled_best = pooled_order[0] if pooled_order else ""
    disagreements = [
        {"family": family, "pooled_best_method": pooled_best, "family_best_method": family_best}
        for family, family_best in sorted(family_best_methods.items())
        if family_best != pooled_best
    ]
    return {
        "protocol_name": report.protocol_name,
        "pooled_order": pooled_order,
        "familywise_average_rank_order": familywise_order,
        "kendall_tau": kendall_tau(pooled_order, familywise_order) if pooled_order and familywise_order else 0.0,
        "family_best_methods": family_best_methods,
        "best_method_changes": bool(disagreements),
        "disagreements": disagreements,
    }
