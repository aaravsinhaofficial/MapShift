"""Construct-validity and benchmark-health reporting for MapShift."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from mapshift.analysis.failure_taxonomy import summarize_rejection_log
from mapshift.analysis.severity import find_undercovered_cells, summarize_family_severity_counts
from mapshift.core.schemas import ReleaseBundle
from mapshift.envs.map2d.generator import Map2DGenerator
from mapshift.envs.map2d.validation import analyze_map2d_environment, summarize_environment_diagnostics, validate_map2d_instance
from mapshift.interventions import build_intervention
from mapshift.interventions.validators import (
    intervention_magnitude,
    validate_intervention_manifest_roundtrip,
    validate_intervention_pair,
)
from mapshift.metrics.statistics import NumericSummary, histogram_counts, mean_or_zero, proportion_true, summarize_numeric
from mapshift.splits.builders import build_canonical_release_split_bundle
from mapshift.tasks.samplers import TaskSampler, TaskSamplingRejected


def _count_by_dimensions(records: Iterable[dict[str, Any]], dimensions: tuple[str, ...]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, ...], int] = {}
    for record in records:
        key = tuple(str(record[dimension]) for dimension in dimensions)
        counts[key] = counts.get(key, 0) + int(record.get("count", 1))
    return [{dimension: value for dimension, value in zip(dimensions, key)} | {"count": count} for key, count in sorted(counts.items())]


def _orthogonal_heuristic_success(environment: Any, start: str, goal: str) -> bool:
    start_cell = environment.resolve_location(start)
    goal_cell = environment.resolve_location(goal)
    row, col = start_cell
    goal_row, goal_col = goal_cell

    def attempt(horizontal_first: bool) -> bool:
        current_row, current_col = row, col
        if horizontal_first:
            step = 1 if goal_col >= current_col else -1
            for new_col in range(current_col, goal_col + step, step):
                if not environment.is_free((current_row, new_col)):
                    return False
            step = 1 if goal_row >= current_row else -1
            for new_row in range(current_row, goal_row + step, step):
                if not environment.is_free((new_row, goal_col)):
                    return False
        else:
            step = 1 if goal_row >= current_row else -1
            for new_row in range(current_row, goal_row + step, step):
                if not environment.is_free((new_row, current_col)):
                    return False
            step = 1 if goal_col >= current_col else -1
            for new_col in range(current_col, goal_col + step, step):
                if not environment.is_free((goal_row, new_col)):
                    return False
        return True

    return attempt(horizontal_first=True) or attempt(horizontal_first=False)


@dataclass(frozen=True)
class BenchmarkHealthReport:
    benchmark_name: str
    benchmark_version: str
    release_name: str
    sample_count_per_motif: int
    task_samples_per_class: int
    environment_health: dict[str, Any]
    split_health: dict[str, Any]
    intervention_coverage: dict[str, Any]
    task_coverage: dict[str, Any]
    task_difficulty: dict[str, Any]
    validator_summary: dict[str, Any]
    rejection_statistics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_construct_validity(results: Any) -> dict[str, Any]:
    """Return a basic construct-validity summary for arbitrary result lists."""

    return {"status": "implemented", "result_count": len(results) if hasattr(results, "__len__") else None}


def generate_mapshift_2d_benchmark_health_report(
    release_bundle: ReleaseBundle,
    sample_count_per_motif: int = 1,
    task_samples_per_class: int = 1,
    min_cell_coverage: int = 1,
    motif_tags: Iterable[str] | None = None,
    family_names: Iterable[str] | None = None,
) -> BenchmarkHealthReport:
    """Generate a deterministic benchmark-health report for MapShift-2D."""

    generator = Map2DGenerator(release_bundle.env2d)
    sampler = TaskSampler(release_bundle.tasks)

    environment_diagnostics = []
    environment_validation_issues: list[dict[str, Any]] = []
    intervention_records: list[dict[str, Any]] = []
    intervention_validation_records: list[dict[str, Any]] = []
    severity_magnitude_records: list[dict[str, Any]] = []
    task_records: list[dict[str, Any]] = []

    motifs = list(motif_tags or release_bundle.env2d.motif_families)
    selected_families = tuple(family_names or release_bundle.interventions.canonical_family_order)
    for motif_index, motif in enumerate(motifs):
        for sample_index in range(sample_count_per_motif):
            seed = (motif_index + 1) * 100 + sample_index
            base_result = generator.generate(seed=seed, motif_tag=motif)
            base_environment = generator.replay_from_manifest(base_result.manifest)
            issues = validate_map2d_instance(base_environment)
            if issues:
                environment_validation_issues.append({"environment_id": base_environment.environment_id, "issues": issues})
            environment_diagnostics.append(analyze_map2d_environment(base_environment))

            for family in selected_families:
                intervention = build_intervention(family, release_bundle.interventions.families[family])
                for severity in range(4):
                    result = intervention.apply(base_environment, severity=severity, seed=seed + severity)
                    transformed_environment = result.environment.__class__.from_manifest_metadata(result.manifest.metadata)
                    intervention_records.append(
                        {
                            "family": family,
                            "severity": severity,
                            "motif": motif,
                            "split": base_environment.split_name,
                            "count": 1,
                        }
                    )

                    validation = validate_intervention_pair(
                        base_environment=base_environment,
                        transformed_environment=transformed_environment,
                        family=family,
                        severity=severity,
                        family_config=release_bundle.interventions.families[family],
                    )
                    roundtrip_issues = validate_intervention_manifest_roundtrip(result.manifest.metadata)
                    intervention_validation_records.append(
                        validation.to_dict()
                        | {
                            "motif": motif,
                            "split": base_environment.split_name,
                            "roundtrip_issues": roundtrip_issues,
                        }
                    )
                    severity_magnitude_records.append(
                        {
                            "environment_id": base_environment.environment_id,
                            "family": family,
                            "severity": severity,
                            "motif": motif,
                            "split": base_environment.split_name,
                            "magnitude": intervention_magnitude(base_environment, transformed_environment, family),
                        }
                    )

                    for task_class, class_config in release_bundle.tasks.classes.items():
                        if not class_config.enabled:
                            continue
                        for task_index in range(task_samples_per_class):
                            task_seed = seed * 1000 + severity * 100 + task_index * 10 + len(task_class)
                            try:
                                sampled = sampler.sample(
                                    base_environment=base_environment,
                                    intervened_environment=transformed_environment,
                                    family=family,
                                    seed=task_seed,
                                    task_class=task_class,
                                )
                            except TaskSamplingRejected:
                                continue

                            task_record = _task_record(
                                base_environment=base_environment,
                                intervened_environment=transformed_environment,
                                family=family,
                                severity=severity,
                                motif=motif,
                                split_name=base_environment.split_name,
                                task_class=task_class,
                                task=sampled.task,
                            )
                            task_records.append(task_record)

    environment_health = summarize_environment_diagnostics(environment_diagnostics)
    split_bundle = build_canonical_release_split_bundle(
        release_bundle=release_bundle,
        sample_count_per_motif=sample_count_per_motif,
        task_samples_per_class=task_samples_per_class,
    )
    split_health = _summarize_split_health(environment_diagnostics)
    split_health["canonical_split_coverage"] = split_bundle.coverage_summary
    split_health["canonical_split_validation_issues"] = list(split_bundle.validation_issues)
    split_health["canonical_split_leakage_report"] = split_bundle.leakage_report.to_dict()
    intervention_coverage = _summarize_intervention_coverage(
        intervention_records=intervention_records,
        release_bundle=release_bundle,
        min_cell_coverage=min_cell_coverage,
        motifs=tuple(motifs),
        families=selected_families,
    )
    task_coverage = _summarize_task_coverage(
        task_records=task_records,
        release_bundle=release_bundle,
        min_cell_coverage=min_cell_coverage,
        motifs=tuple(motifs),
        families=selected_families,
    )
    task_difficulty = _summarize_task_difficulty(task_records)
    validator_summary = _summarize_validator_outputs(
        environment_validation_issues=environment_validation_issues,
        intervention_validation_records=intervention_validation_records,
        severity_magnitude_records=severity_magnitude_records,
    )
    rejection_statistics = _summarize_rejections(sampler.rejection_log)

    return BenchmarkHealthReport(
        benchmark_name=release_bundle.root.benchmark_name,
        benchmark_version=release_bundle.root.benchmark_version,
        release_name=release_bundle.root.release_name,
        sample_count_per_motif=sample_count_per_motif,
        task_samples_per_class=task_samples_per_class,
        environment_health=environment_health,
        split_health=split_health,
        intervention_coverage=intervention_coverage,
        task_coverage=task_coverage,
        task_difficulty=task_difficulty,
        validator_summary=validator_summary,
        rejection_statistics=rejection_statistics,
    )


def _task_record(
    base_environment: Any,
    intervened_environment: Any,
    family: str,
    severity: int,
    motif: str,
    split_name: str,
    task_class: str,
    task: Any,
) -> dict[str, Any]:
    start_node = getattr(task, "start_node_id", intervened_environment.start_node_id)
    goal_node = getattr(task, "goal_node_id", intervened_environment.goal_node_id)
    base_distance = base_environment.shortest_path_length(base_environment.start_node_id, base_environment.goal_node_id)
    intervened_distance = intervened_environment.shortest_path_length(start_node, goal_node) if goal_node is not None else None
    oracle_solved = intervened_distance is not None
    heuristic_solved = goal_node is not None and _orthogonal_heuristic_success(intervened_environment, start_node, goal_node)
    reroute_delta = None
    if base_distance is not None and intervened_distance is not None:
        reroute_delta = intervened_distance - base_distance

    return {
        "family": family,
        "severity": severity,
        "motif": motif,
        "split": split_name,
        "task_class": task_class,
        "task_type": task.task_type,
        "base_distance": base_distance,
        "intervened_distance": intervened_distance,
        "reroute_delta": reroute_delta,
        "oracle_solved": oracle_solved,
        "heuristic_solved": heuristic_solved,
        "adaptation_budget_steps": getattr(task, "adaptation_budget_steps", None),
        "count": 1,
    }


def _summarize_split_health(environment_diagnostics: list[Any]) -> dict[str, Any]:
    grouped: dict[str, list[Any]] = {}
    for diagnostic in environment_diagnostics:
        grouped.setdefault(diagnostic.split_name, []).append(diagnostic)

    summaries = {}
    for split_name, items in sorted(grouped.items()):
        summaries[split_name] = {
            "environment_count": len(items),
            "path_length_summary": summarize_numeric([item.start_goal_distance for item in items]).to_dict(),
            "free_space_ratio_summary": summarize_numeric([item.free_space_ratio for item in items]).to_dict(),
            "connected_component_summary": summarize_numeric([item.connected_component_count for item in items]).to_dict(),
        }
    return summaries


def _summarize_intervention_coverage(
    intervention_records: list[dict[str, Any]],
    release_bundle: ReleaseBundle,
    min_cell_coverage: int,
    motifs: tuple[str, ...] | None = None,
    families: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    expected_values = {
        "family": families or release_bundle.interventions.canonical_family_order,
        "severity": tuple(range(4)),
        "motif": motifs or release_bundle.env2d.motif_families,
        "split": ("train", "val", "test"),
    }
    family_severity = summarize_family_severity_counts(intervention_records)
    by_family_severity_motif = _count_by_dimensions(intervention_records, ("family", "severity", "motif"))
    by_family_severity_split = _count_by_dimensions(intervention_records, ("family", "severity", "split"))
    undercovered = find_undercovered_cells(
        by_family_severity_motif,
        ("family", "severity", "motif"),
        min_cell_coverage,
        expected_values=expected_values,
    )
    return {
        "family_severity_table": family_severity,
        "family_severity_motif_table": by_family_severity_motif,
        "family_severity_split_table": by_family_severity_split,
        "undercovered_cells": undercovered,
    }


def _summarize_task_coverage(
    task_records: list[dict[str, Any]],
    release_bundle: ReleaseBundle,
    min_cell_coverage: int,
    motifs: tuple[str, ...] | None = None,
    families: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    expected_values = {
        "family": families or release_bundle.interventions.canonical_family_order,
        "severity": tuple(range(4)),
        "motif": motifs or release_bundle.env2d.motif_families,
        "task_class": tuple(class_name for class_name, config in release_bundle.tasks.classes.items() if config.enabled),
        "split": ("train", "val", "test"),
    }
    family_severity = summarize_family_severity_counts(task_records)
    by_family_severity_task = _count_by_dimensions(task_records, ("family", "severity", "task_class"))
    by_family_severity_split_task = _count_by_dimensions(task_records, ("family", "severity", "split", "task_class"))
    by_family_severity_motif_task = _count_by_dimensions(task_records, ("family", "severity", "motif", "task_class"))
    undercovered = find_undercovered_cells(
        by_family_severity_motif_task,
        ("family", "severity", "motif", "task_class"),
        min_cell_coverage,
        expected_values=expected_values,
    )
    return {
        "family_severity_table": family_severity,
        "family_severity_task_table": by_family_severity_task,
        "family_severity_split_task_table": by_family_severity_split_task,
        "family_severity_motif_task_table": by_family_severity_motif_task,
        "undercovered_cells": undercovered,
    }


def _group_task_records(records: list[dict[str, Any]], dimensions: tuple[str, ...]) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for record in records:
        key = tuple(str(record[dimension]) for dimension in dimensions)
        grouped.setdefault(key, []).append(record)
    return grouped


def _summarize_task_difficulty(records: list[dict[str, Any]]) -> dict[str, Any]:
    per_family_severity = {}
    for key, items in sorted(_group_task_records(records, ("family", "severity")).items()):
        family, severity = key
        distances = [record["intervened_distance"] for record in items if record["intervened_distance"] is not None]
        reroute_deltas = [record["reroute_delta"] for record in items if record["reroute_delta"] is not None]
        adaptation_budgets = [record["adaptation_budget_steps"] for record in items if record["adaptation_budget_steps"] is not None]
        per_family_severity[f"{family}:{severity}"] = {
            "task_count": len(items),
            "distance_summary": summarize_numeric(distances).to_dict(),
            "distance_histogram": histogram_counts(distances, (0, 5, 10, 20, 40, 80, 160)),
            "reroute_delta_summary": summarize_numeric(reroute_deltas).to_dict(),
            "reroute_delta_histogram": histogram_counts(reroute_deltas, (-40, -20, -5, 0, 5, 20, 40, 80)),
            "oracle_solvability": proportion_true(record["oracle_solved"] for record in items),
            "heuristic_success_estimate": proportion_true(record["heuristic_solved"] for record in items),
            "adaptation_budget_summary": summarize_numeric(adaptation_budgets).to_dict(),
        }

    return {
        "task_count": len(records),
        "overall_distance_summary": summarize_numeric([record["intervened_distance"] for record in records if record["intervened_distance"] is not None]).to_dict(),
        "overall_distance_histogram": histogram_counts(
            [record["intervened_distance"] for record in records if record["intervened_distance"] is not None],
            (0, 5, 10, 20, 40, 80, 160),
        ),
        "overall_reroute_delta_summary": summarize_numeric([record["reroute_delta"] for record in records if record["reroute_delta"] is not None]).to_dict(),
        "overall_reroute_delta_histogram": histogram_counts(
            [record["reroute_delta"] for record in records if record["reroute_delta"] is not None],
            (-40, -20, -5, 0, 5, 20, 40, 80),
        ),
        "oracle_solvability": proportion_true(record["oracle_solved"] for record in records),
        "heuristic_success_estimate": proportion_true(record["heuristic_solved"] for record in records),
        "average_difficulty_shift": mean_or_zero(
            [
                (record["intervened_distance"] - record["base_distance"])
                for record in records
                if record["intervened_distance"] is not None and record["base_distance"] is not None
            ]
        ),
        "per_family_severity": per_family_severity,
    }


def _summarize_validator_outputs(
    environment_validation_issues: list[dict[str, Any]],
    intervention_validation_records: list[dict[str, Any]],
    severity_magnitude_records: list[dict[str, Any]],
) -> dict[str, Any]:
    failed_interventions = [record for record in intervention_validation_records if record["issues"] or record["roundtrip_issues"]]
    issues_by_family: dict[str, int] = {}
    for record in failed_interventions:
        issues_by_family[record["family"]] = issues_by_family.get(record["family"], 0) + len(record["issues"]) + len(record["roundtrip_issues"])
    severity_monotonicity_failures = []
    for key, items in sorted(_group_task_records(severity_magnitude_records, ("environment_id", "family")).items()):
        ordered = sorted(items, key=lambda item: int(item["severity"]))
        magnitudes = [float(item["magnitude"]) for item in ordered]
        if any(left > right for left, right in zip(magnitudes, magnitudes[1:])):
            environment_id, family = key
            severity_monotonicity_failures.append(
                {
                    "environment_id": environment_id,
                    "family": family,
                    "magnitudes": magnitudes,
                }
            )
    return {
        "environment_validation_failures": environment_validation_issues,
        "intervention_validation_failures": failed_interventions,
        "failed_intervention_count": len(failed_interventions),
        "issues_by_family": dict(sorted(issues_by_family.items())),
        "severity_monotonicity_failures": severity_monotonicity_failures,
    }


def _summarize_rejections(rejection_log: list[Any]) -> dict[str, Any]:
    raw_entries = [
        {
            "family": rejection.family,
            "task_class": rejection.task_class,
            "task_type": rejection.task_type,
            "reason": rejection.reason,
            "details": dict(rejection.details),
        }
        for rejection in rejection_log
    ]
    summarized = summarize_rejection_log(raw_entries)
    return {
        "total_rejections": len(rejection_log),
        "rejections_by_reason": {reason: summary.count for reason, summary in summarized.items()},
        "detailed_rejections": {reason: asdict(summary) for reason, summary in summarized.items()},
    }
