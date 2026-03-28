"""Study orchestration and paper-facing result bundles for MapShift-2D."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from mapshift.analysis.construct_validity import generate_mapshift_2d_benchmark_health_report
from mapshift.analysis.figures import (
    familywise_degradation_curves_figure_data,
    mapshift_changes_conclusion_figure_data,
    protocol_ranking_comparison_figure_data,
)
from mapshift.core.manifests import StudyManifest
from mapshift.core.schemas import ReleaseBundle, load_release_bundle
from mapshift.metrics.statistics import correlation_matrix, mean_or_zero
from mapshift.runners.compare_protocols import run_protocol_comparison_suite
from mapshift.runners.evaluate import default_post_intervention_protocol, run_calibration_suite


@dataclass(frozen=True)
class MapShift2DStudyConfig:
    """Canonical config for the first full 2D MapShift study."""

    schema_version: str
    study_name: str
    benchmark_config: str
    baseline_run_configs: tuple[str, ...]
    sample_count_per_motif: int
    task_samples_per_class: int
    severity_levels: tuple[int, ...]
    protocol_names: tuple[str, ...]
    min_cell_coverage: int = 1
    motif_tags: tuple[str, ...] = ()
    family_names: tuple[str, ...] = ()
    output_subdir: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MapShift2DStudyBundle:
    """Machine-readable result bundle for the first full 2D study."""

    study_name: str
    release_name: str
    benchmark_version: str
    study_config: dict[str, Any]
    raw_reports: dict[str, Any]
    construct_validity: dict[str, Any]
    discriminative_power: dict[str, Any]
    protocol_sensitivity: dict[str, Any]
    benchmark_health: dict[str, Any]
    proposition_support: dict[str, Any]
    weakness_summary: dict[str, Any]
    report_artifacts: dict[str, Any]
    study_manifest: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_mapshift_2d_study_config(path: str | Path) -> MapShift2DStudyConfig:
    """Load and minimally validate a 2D study config."""

    config_path = Path(path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Study config must be an object: {config_path}")
    for field_name in ("schema_version", "study_name", "benchmark_config"):
        value = payload.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing or invalid {field_name} in study config: {config_path}")
    baseline_run_configs = payload.get("baseline_run_configs")
    if not isinstance(baseline_run_configs, list) or not baseline_run_configs:
        raise ValueError(f"baseline_run_configs must be a non-empty list in study config: {config_path}")

    study_root = config_path.parent
    benchmark_config = str((study_root / str(payload["benchmark_config"])).resolve())
    resolved_baseline_configs = tuple(str((study_root / str(item)).resolve()) for item in baseline_run_configs)
    output_subdir = str(payload.get("output_subdir", ""))
    return MapShift2DStudyConfig(
        schema_version=str(payload["schema_version"]),
        study_name=str(payload["study_name"]),
        benchmark_config=benchmark_config,
        baseline_run_configs=resolved_baseline_configs,
        sample_count_per_motif=int(payload.get("sample_count_per_motif", 1)),
        task_samples_per_class=int(payload.get("task_samples_per_class", 1)),
        severity_levels=tuple(int(level) for level in payload.get("severity_levels", [0, 1, 2, 3])),
        protocol_names=tuple(str(name) for name in payload.get("protocol_names", ("cep", "same_environment", "no_exploration", "short_horizon", "long_horizon"))),
        min_cell_coverage=int(payload.get("min_cell_coverage", 1)),
        motif_tags=tuple(str(tag) for tag in payload.get("motif_tags", [])),
        family_names=tuple(str(name) for name in payload.get("family_names", [])),
        output_subdir=output_subdir,
    )


def run_mapshift_2d_study(
    study_config: MapShift2DStudyConfig,
    *,
    release_bundle: ReleaseBundle | None = None,
) -> MapShift2DStudyBundle:
    """Run the first full 2D MapShift study using the frozen study config."""

    bundle = release_bundle or load_release_bundle(study_config.benchmark_config)
    cep_report = run_calibration_suite(
        release_bundle=bundle,
        baseline_run_configs=study_config.baseline_run_configs,
        sample_count_per_motif=study_config.sample_count_per_motif,
        task_samples_per_class=study_config.task_samples_per_class,
        severity_levels=study_config.severity_levels,
        protocol=default_post_intervention_protocol(),
        motif_tags=study_config.motif_tags or None,
        family_names=study_config.family_names or None,
    )
    protocol_report = run_protocol_comparison_suite(
        release_bundle=bundle,
        baseline_run_configs=study_config.baseline_run_configs,
        sample_count_per_motif=study_config.sample_count_per_motif,
        task_samples_per_class=study_config.task_samples_per_class,
        severity_levels=study_config.severity_levels,
        motif_tags=study_config.motif_tags or None,
        family_names=study_config.family_names or None,
        protocol_names=study_config.protocol_names,
    )
    benchmark_health = generate_mapshift_2d_benchmark_health_report(
        release_bundle=bundle,
        sample_count_per_motif=study_config.sample_count_per_motif,
        task_samples_per_class=study_config.task_samples_per_class,
        min_cell_coverage=study_config.min_cell_coverage,
        motif_tags=study_config.motif_tags or None,
        family_names=study_config.family_names or None,
    )
    return build_mapshift_2d_study_bundle(
        release_bundle=bundle,
        study_config=study_config,
        cep_report_payload=cep_report.to_dict(),
        protocol_report_payload=protocol_report.to_dict(),
        benchmark_health_payload=benchmark_health.to_dict(),
    )


def build_mapshift_2d_study_bundle(
    *,
    release_bundle: ReleaseBundle,
    study_config: MapShift2DStudyConfig,
    cep_report_payload: dict[str, Any],
    protocol_report_payload: dict[str, Any],
    benchmark_health_payload: dict[str, Any],
    artifact_paths: dict[str, str] | None = None,
) -> MapShift2DStudyBundle:
    """Build the paper-facing study bundle from raw report payloads."""

    construct_validity = _build_construct_validity_outputs(cep_report_payload, benchmark_health_payload)
    discriminative_power = _build_discriminative_power_outputs(cep_report_payload)
    protocol_sensitivity = _build_protocol_sensitivity_outputs(protocol_report_payload)
    proposition_support = _build_proposition_support(
        construct_validity=construct_validity,
        protocol_sensitivity=protocol_sensitivity,
        benchmark_health=benchmark_health_payload,
    )
    weakness_summary = _build_weakness_summary(
        proposition_support=proposition_support,
        benchmark_health=benchmark_health_payload,
        protocol_sensitivity=protocol_sensitivity,
    )
    report_artifacts = _build_report_artifacts(
        cep_report_payload=cep_report_payload,
        protocol_report_payload=protocol_report_payload,
        benchmark_health_payload=benchmark_health_payload,
        construct_validity=construct_validity,
        discriminative_power=discriminative_power,
        protocol_sensitivity=protocol_sensitivity,
        proposition_support=proposition_support,
    )
    study_manifest = _build_study_manifest(
        release_bundle=release_bundle,
        study_config=study_config,
        cep_report_payload=cep_report_payload,
        protocol_report_payload=protocol_report_payload,
        artifact_paths=artifact_paths or {},
    )
    return MapShift2DStudyBundle(
        study_name=study_config.study_name,
        release_name=release_bundle.root.release_name,
        benchmark_version=release_bundle.root.benchmark_version,
        study_config=study_config.to_dict(),
        raw_reports={
            "cep_report": cep_report_payload,
            "protocol_comparison_report": protocol_report_payload,
            "benchmark_health_report": benchmark_health_payload,
        },
        construct_validity=construct_validity,
        discriminative_power=discriminative_power,
        protocol_sensitivity=protocol_sensitivity,
        benchmark_health=benchmark_health_payload,
        proposition_support=proposition_support,
        weakness_summary=weakness_summary,
        report_artifacts=report_artifacts,
        study_manifest=study_manifest.to_dict(),
    )


def write_mapshift_2d_study_bundle(bundle: MapShift2DStudyBundle, output_dir: str | Path) -> dict[str, str]:
    """Write a study bundle and its component artifacts to disk."""

    root = Path(output_dir).resolve()
    raw_dir = root / "raw"
    tables_dir = root / "tables"
    figures_dir = root / "figures"
    manifests_dir = root / "manifests"
    for directory in (root, raw_dir, tables_dir, figures_dir, manifests_dir):
        directory.mkdir(parents=True, exist_ok=True)

    paths = {
        "study_bundle": str(root / "study_bundle.json"),
        "raw_cep_report": str(raw_dir / "cep_report.json"),
        "raw_protocol_comparison_report": str(raw_dir / "protocol_comparison_report.json"),
        "raw_benchmark_health_report": str(raw_dir / "benchmark_health_report.json"),
        "study_manifest": str(manifests_dir / "study_manifest.json"),
        "familywise_main_results": str(tables_dir / "familywise_main_results.json"),
        "severity_response": str(tables_dir / "severity_response.json"),
        "protocol_sensitivity_and_rank_correlation": str(tables_dir / "protocol_sensitivity_and_rank_correlation.json"),
        "benchmark_health_summary": str(tables_dir / "benchmark_health_summary.json"),
        "construct_validity_summary": str(tables_dir / "construct_validity_summary.json"),
        "discriminative_power_summary": str(tables_dir / "discriminative_power_summary.json"),
        "familywise_degradation_curves": str(figures_dir / "familywise_degradation_curves.json"),
        "protocol_ranking_comparison": str(figures_dir / "protocol_ranking_comparison.json"),
        "mapshift_changes_conclusion": str(figures_dir / "mapshift_changes_conclusion.json"),
    }

    updated_bundle = bundle.to_dict()
    updated_bundle["study_manifest"]["artifact_paths"] = dict(sorted(paths.items()))
    Path(paths["study_bundle"]).write_text(json.dumps(updated_bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(paths["raw_cep_report"]).write_text(json.dumps(bundle.raw_reports["cep_report"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(paths["raw_protocol_comparison_report"]).write_text(json.dumps(bundle.raw_reports["protocol_comparison_report"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(paths["raw_benchmark_health_report"]).write_text(json.dumps(bundle.raw_reports["benchmark_health_report"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(paths["study_manifest"]).write_text(json.dumps(updated_bundle["study_manifest"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(paths["familywise_main_results"]).write_text(json.dumps(bundle.report_artifacts["tables"]["familywise_main_results"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(paths["severity_response"]).write_text(json.dumps(bundle.report_artifacts["tables"]["severity_response"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(paths["protocol_sensitivity_and_rank_correlation"]).write_text(json.dumps(bundle.report_artifacts["tables"]["protocol_sensitivity_and_rank_correlation"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(paths["benchmark_health_summary"]).write_text(json.dumps(bundle.report_artifacts["tables"]["benchmark_health_summary"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(paths["construct_validity_summary"]).write_text(json.dumps(bundle.report_artifacts["tables"]["construct_validity_summary"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(paths["discriminative_power_summary"]).write_text(json.dumps(bundle.report_artifacts["tables"]["discriminative_power_summary"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(paths["familywise_degradation_curves"]).write_text(json.dumps(bundle.report_artifacts["figures"]["familywise_degradation_curves"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(paths["protocol_ranking_comparison"]).write_text(json.dumps(bundle.report_artifacts["figures"]["protocol_ranking_comparison"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(paths["mapshift_changes_conclusion"]).write_text(json.dumps(bundle.report_artifacts["figures"]["mapshift_changes_conclusion"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return paths


def _build_construct_validity_outputs(cep_report_payload: dict[str, Any], benchmark_health_payload: dict[str, Any]) -> dict[str, Any]:
    severity_rows = cep_report_payload["familywise_summary"]["by_severity"]
    family_rows = cep_report_payload["familywise_summary"]["rows"]
    family_signatures = _family_signature_vectors(severity_rows)
    return {
        "familywise_degradation_profiles": severity_rows,
        "severity_response_monotonicity": cep_report_payload["severity_summary"]["rows"],
        "cross_family_metric_correlations": {
            "family_primary_score": correlation_matrix(family_signatures),
        },
        "failure_profile_separability": _family_signature_separability(family_signatures),
        "coverage_guardrails": {
            "undercovered_task_cells": benchmark_health_payload["task_coverage"]["undercovered_cells"],
            "validator_failures": benchmark_health_payload["validator_summary"]["failed_intervention_count"],
        },
        "familywise_main_results": family_rows,
    }


def _family_signature_vectors(severity_rows: Sequence[dict[str, Any]]) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for row in sorted(
        severity_rows,
        key=lambda item: (
            str(item["family"]),
            str(item["baseline_name"]),
            int(item["severity"]),
        ),
    ):
        grouped.setdefault(str(row["family"]), []).append(float(row["family_primary_score"]))
    return grouped


def _family_signature_separability(family_signatures: dict[str, Sequence[float]]) -> dict[str, Any]:
    pairwise = []
    names = sorted(family_signatures)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            left_values = [float(value) for value in family_signatures[left]]
            right_values = [float(value) for value in family_signatures[right]]
            distance = sum(abs(left_value - right_value) for left_value, right_value in zip(left_values, right_values)) / max(1, len(left_values))
            pairwise.append({"left_family": left, "right_family": right, "mean_absolute_distance": distance})
    return {"pairwise_signature_distance": pairwise}


def _build_discriminative_power_outputs(cep_report_payload: dict[str, Any]) -> dict[str, Any]:
    family_rows = cep_report_payload["familywise_summary"]["rows"]
    ranking_summary = cep_report_payload["ranking_summary"]
    oracle_rows = [row for row in family_rows if row["baseline_name"] == "oracle_post_intervention_planner"]
    heuristic_rows = [row for row in family_rows if row["baseline_name"] == "weak_heuristic_baseline"]
    recurrent_rows = [row for row in family_rows if row["baseline_name"] == "monolithic_recurrent_world_model"]
    family_gaps = []
    for family in sorted({str(row["family"]) for row in family_rows}):
        oracle = next((row for row in oracle_rows if row["family"] == family), None)
        heuristic = next((row for row in heuristic_rows if row["family"] == family), None)
        recurrent = next((row for row in recurrent_rows if row["family"] == family), None)
        family_gaps.append(
            {
                "family": family,
                "oracle_minus_heuristic": _safe_gap(oracle, heuristic),
                "oracle_minus_recurrent": _safe_gap(oracle, recurrent),
                "recurrent_minus_heuristic": _safe_gap(recurrent, heuristic),
            }
        )
    return {
        "familywise_rankings": ranking_summary["family_orders"],
        "familywise_gaps": family_gaps,
        "ranking_spread": ranking_summary["family_spreads"],
        "rank_stability": ranking_summary["rank_stability"],
        "saturation_checks": cep_report_payload["saturation_summary"],
    }


def _safe_gap(left_row: dict[str, Any] | None, right_row: dict[str, Any] | None) -> float | None:
    if left_row is None or right_row is None:
        return None
    return float(left_row["family_primary_score"]) - float(right_row["family_primary_score"])


def _build_protocol_sensitivity_outputs(protocol_report_payload: dict[str, Any]) -> dict[str, Any]:
    pairwise = dict(protocol_report_payload["pairwise_comparisons"])
    pooled_vs_familywise = dict(protocol_report_payload["pooled_vs_familywise"])
    stable_conclusions = []
    fragile_conclusions = []
    for name, comparison in sorted(pairwise.items()):
        if float(comparison.get("kendall_tau", 0.0)) >= 0.95 and not comparison.get("rank_reversals") and not comparison.get("family_rank_changes"):
            stable_conclusions.append(name)
        else:
            fragile_conclusions.append(name)
    if pooled_vs_familywise:
        if pooled_vs_familywise.get("best_method_changes"):
            fragile_conclusions.append("pooled_vs_familywise_reporting")
        else:
            stable_conclusions.append("pooled_vs_familywise_reporting")
    return {
        "pairwise_comparisons": pairwise,
        "pooled_vs_familywise_reporting": pooled_vs_familywise,
        "stable_conclusions": stable_conclusions,
        "fragile_conclusions": fragile_conclusions,
    }


def _build_proposition_support(
    *,
    construct_validity: dict[str, Any],
    protocol_sensitivity: dict[str, Any],
    benchmark_health: dict[str, Any],
) -> dict[str, Any]:
    p1 = _p1_support(protocol_sensitivity)
    p2 = _p2_support(construct_validity)
    p3 = _p3_support(protocol_sensitivity)
    return {
        "P1": p1,
        "P2": p2,
        "P3": p3,
        "benchmark_health_context": {
            "undercovered_task_cells": len(benchmark_health["task_coverage"]["undercovered_cells"]),
            "validator_failures": benchmark_health["validator_summary"]["failed_intervention_count"],
        },
    }


def _p1_support(protocol_sensitivity: dict[str, Any]) -> dict[str, Any]:
    same_vs_cep = protocol_sensitivity["pairwise_comparisons"].get("same_environment_vs_cep", {})
    family_rank_changes = same_vs_cep.get("family_rank_changes", [])
    rank_reversals = same_vs_cep.get("rank_reversals", [])
    supported = bool(family_rank_changes or rank_reversals or same_vs_cep.get("best_method_changes"))
    return {
        "status": "supported" if supported else "mixed",
        "evidence": {
            "kendall_tau": same_vs_cep.get("kendall_tau", 0.0),
            "rank_reversal_count": len(rank_reversals),
            "family_rank_change_count": len(family_rank_changes),
        },
        "claim": "standard same-environment evaluation can differ materially from post-intervention CEP evaluation",
    }


def _p2_support(construct_validity: dict[str, Any]) -> dict[str, Any]:
    monotonicity_rows = construct_validity["severity_response_monotonicity"]
    monotone_count = sum(1 for row in monotonicity_rows if row["monotone_degradation"])
    correlation_rows = construct_validity["cross_family_metric_correlations"]["family_primary_score"]
    off_diagonal = [
        value
        for left_name, row in correlation_rows.items()
        for right_name, value in row.items()
        if left_name != right_name
    ]
    max_correlation = max(off_diagonal) if off_diagonal else 0.0
    supported = monotone_count > 0 and max_correlation < 0.98
    return {
        "status": "supported" if supported else "mixed",
        "evidence": {
            "monotone_profile_count": monotone_count,
            "max_cross_family_correlation": max_correlation,
            "pairwise_signature_distance": construct_validity["failure_profile_separability"]["pairwise_signature_distance"],
        },
        "claim": "intervention families induce non-redundant stress patterns",
    }


def _p3_support(protocol_sensitivity: dict[str, Any]) -> dict[str, Any]:
    comparisons = protocol_sensitivity["pairwise_comparisons"]
    evidence = {
        name: {
            "kendall_tau": comparison.get("kendall_tau", 0.0),
            "rank_reversal_count": len(comparison.get("rank_reversals", [])),
            "family_rank_change_count": len(comparison.get("family_rank_changes", [])),
            "best_method_changes": bool(comparison.get("best_method_changes")),
        }
        for name, comparison in comparisons.items()
    }
    pooled = protocol_sensitivity.get("pooled_vs_familywise_reporting", {})
    if pooled:
        evidence["pooled_vs_familywise_reporting"] = {
            "kendall_tau": pooled.get("kendall_tau", 0.0),
            "best_method_changes": bool(pooled.get("best_method_changes")),
            "disagreement_count": len(pooled.get("disagreements", [])),
        }
    supported = any(
        item.get("best_method_changes")
        or item.get("rank_reversal_count", 0) > 0
        or item.get("family_rank_change_count", 0) > 0
        or item.get("disagreement_count", 0) > 0
        for item in evidence.values()
    )
    return {
        "status": "supported" if supported else "mixed",
        "evidence": evidence,
        "claim": "evaluation protocol choices can change rankings or conclusions",
    }


def _build_weakness_summary(
    *,
    proposition_support: dict[str, Any],
    benchmark_health: dict[str, Any],
    protocol_sensitivity: dict[str, Any],
) -> dict[str, Any]:
    weaknesses = []
    if benchmark_health["task_coverage"]["undercovered_cells"]:
        weaknesses.append(
            {
                "category": "undercovered_task_cells",
                "count": len(benchmark_health["task_coverage"]["undercovered_cells"]),
                "details": benchmark_health["task_coverage"]["undercovered_cells"],
            }
        )
    if benchmark_health["rejection_statistics"]["rejections_by_reason"]:
        weaknesses.append(
            {
                "category": "task_filter_rejections",
                "details": benchmark_health["rejection_statistics"]["rejections_by_reason"],
            }
        )
    if proposition_support["P3"]["status"] != "supported":
        weaknesses.append(
            {
                "category": "weak_protocol_rank_shift",
                "details": protocol_sensitivity["pairwise_comparisons"].get("same_environment_vs_cep", {}),
            }
        )
    return {"weaknesses": weaknesses}


def _build_report_artifacts(
    *,
    cep_report_payload: dict[str, Any],
    protocol_report_payload: dict[str, Any],
    benchmark_health_payload: dict[str, Any],
    construct_validity: dict[str, Any],
    discriminative_power: dict[str, Any],
    protocol_sensitivity: dict[str, Any],
    proposition_support: dict[str, Any],
) -> dict[str, Any]:
    study_summary = {"proposition_support": proposition_support}
    return {
        "tables": {
            "familywise_main_results": cep_report_payload["familywise_summary"]["rows"],
            "severity_response": cep_report_payload["familywise_summary"]["by_severity"],
            "protocol_sensitivity_and_rank_correlation": protocol_sensitivity,
            "benchmark_health_summary": benchmark_health_payload,
            "construct_validity_summary": construct_validity,
            "discriminative_power_summary": discriminative_power,
        },
        "figures": {
            "familywise_degradation_curves": familywise_degradation_curves_figure_data(cep_report_payload),
            "protocol_ranking_comparison": protocol_ranking_comparison_figure_data(protocol_report_payload),
            "mapshift_changes_conclusion": mapshift_changes_conclusion_figure_data(protocol_report_payload, study_summary),
        },
    }


def _build_study_manifest(
    *,
    release_bundle: ReleaseBundle,
    study_config: MapShift2DStudyConfig,
    cep_report_payload: dict[str, Any],
    protocol_report_payload: dict[str, Any],
    artifact_paths: dict[str, str],
) -> StudyManifest:
    payload = json.dumps(study_config.to_dict(), sort_keys=True).encode("utf-8")
    config_hash = hashlib.sha1(payload).hexdigest()[:12]
    protocol_names = sorted(protocol_report_payload["protocol_reports"].keys())
    baseline_names = sorted(cep_report_payload["baseline_metadata"].keys())
    seeds = sorted({int(record["model_seed"]) for record in cep_report_payload["records"]})
    return StudyManifest(
        artifact_id=f"study-{study_config.study_name}",
        artifact_type="study",
        benchmark_version=release_bundle.root.benchmark_version,
        code_version="mapshift-2d-study-v1",
        config_hash=config_hash,
        parent_ids=[release_bundle.root.release_name],
        seed_values=seeds,
        study_name=study_config.study_name,
        release_name=release_bundle.root.release_name,
        protocol_names=protocol_names,
        baseline_names=baseline_names,
        artifact_paths=dict(sorted(artifact_paths.items())),
        metadata={
            "benchmark_config": study_config.benchmark_config,
            "baseline_run_configs": list(study_config.baseline_run_configs),
            "sample_count_per_motif": study_config.sample_count_per_motif,
            "task_samples_per_class": study_config.task_samples_per_class,
            "severity_levels": list(study_config.severity_levels),
        },
    )
