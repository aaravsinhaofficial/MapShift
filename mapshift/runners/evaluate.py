"""Evaluation runner and metrics/reporting helpers."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

from mapshift.analysis.bootstrap import bootstrap_grouped_metric
from mapshift.analysis.rank_stability import summarize_rank_samples
from mapshift.baselines.api import (
    BaselineContext,
    BaselineRunConfig,
    ExplorationResult,
    TaskEvaluationResult,
    build_run_manifest,
    instantiate_baseline,
    load_baseline_run_config,
    task_class_name,
)
from mapshift.core.schemas import ReleaseBundle
from mapshift.interventions import build_intervention
from mapshift.metrics.adaptation_metrics import adaptation_sample_efficiency, recovery_vs_budget, summarize_adaptation_curve
from mapshift.metrics.inference_metrics import accuracy_from_flags, change_detection_auroc, masked_state_inference_accuracy
from mapshift.metrics.planning_metrics import (
    counterfactual_planning_accuracy,
    long_horizon_rollout_consistency,
    mean_normalized_path_efficiency,
    success_rate,
)
from mapshift.metrics.ranking import kendall_tau, rank_by_metric, rank_positions, rank_reversals, ranking_spread, ranking_stddev
from mapshift.metrics.statistics import mean_or_zero, proportion_true
from mapshift.runners.explore import run_exploration
from mapshift.tasks.adaptation import AdaptationTask
from mapshift.tasks.inference import InferenceTask
from mapshift.tasks.planning import PlanningTask
from mapshift.tasks.samplers import TaskSampler, TaskSamplingRejected
from mapshift.envs.map2d.generator import Map2DGenerator


@dataclass(frozen=True)
class EvaluationProtocol:
    """Configuration of one evaluation protocol variant."""

    name: str
    environment_mode: str = "post_intervention"
    exploration_mode: str = "reward_free"
    horizon_multiplier: float = 1.0


@dataclass(frozen=True)
class EvaluationRecord:
    baseline_name: str
    protocol_name: str
    family: str
    severity: int
    split_name: str
    motif_tag: str
    task_class: str
    task_type: str
    environment_id: str
    base_environment_id: str
    task_id: str
    model_seed: int
    task_horizon_steps: int
    success: bool
    solvable: bool
    primary_score: float
    observed_length: float | None
    oracle_length: float | None
    path_efficiency: float
    oracle_gap: float | None
    oracle_success: bool
    oracle_primary_score: float
    predicted_answer: Any = None
    expected_answer: Any = None
    correct: bool | None = None
    adaptation_curve: tuple[float, ...] = ()
    metadata: dict[str, Any] = None  # type: ignore[assignment]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if payload["metadata"] is None:
            payload["metadata"] = {}
        return payload


@dataclass(frozen=True)
class CalibrationReport:
    release_name: str
    benchmark_version: str
    protocol_name: str
    sample_count_per_motif: int
    task_samples_per_class: int
    baseline_metadata: dict[str, dict[str, Any]]
    run_manifest_metadata: dict[str, dict[str, Any]]
    records: tuple[EvaluationRecord, ...]
    familywise_summary: dict[str, Any]
    secondary_summary: dict[str, Any]
    bootstrap_summary: dict[str, Any]
    ranking_summary: dict[str, Any]
    severity_summary: dict[str, Any]
    supplementary_summary: dict[str, Any]
    oracle_solvability_summary: dict[str, Any]
    weak_baseline_summary: dict[str, Any]
    saturation_summary: dict[str, Any]
    impossibility_summary: dict[str, Any]
    report_artifacts: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "release_name": self.release_name,
            "benchmark_version": self.benchmark_version,
            "protocol_name": self.protocol_name,
            "sample_count_per_motif": self.sample_count_per_motif,
            "task_samples_per_class": self.task_samples_per_class,
            "baseline_metadata": self.baseline_metadata,
            "run_manifest_metadata": self.run_manifest_metadata,
            "records": [record.to_dict() for record in self.records],
            "familywise_summary": self.familywise_summary,
            "secondary_summary": self.secondary_summary,
            "bootstrap_summary": self.bootstrap_summary,
            "ranking_summary": self.ranking_summary,
            "severity_summary": self.severity_summary,
            "supplementary_summary": self.supplementary_summary,
            "oracle_solvability_summary": self.oracle_solvability_summary,
            "weak_baseline_summary": self.weak_baseline_summary,
            "saturation_summary": self.saturation_summary,
            "impossibility_summary": self.impossibility_summary,
            "report_artifacts": self.report_artifacts,
        }


def default_post_intervention_protocol() -> EvaluationProtocol:
    """Return the canonical CEP evaluation protocol."""

    return EvaluationProtocol(name="cep", environment_mode="post_intervention", exploration_mode="reward_free", horizon_multiplier=1.0)


def run_evaluation(model: Any, environment: Any, task: Any, exploration: Any, context: BaselineContext) -> TaskEvaluationResult:
    """Run one evaluation pass through the shared baseline interface."""

    if task_class_name(task) == "adaptation":
        return model.adapt(environment, task, exploration, context)
    return model.evaluate(environment, task, exploration, context)


def load_run_configs(paths: Sequence[str | Path]) -> list[BaselineRunConfig]:
    """Load one or more baseline run configs."""

    return [load_baseline_run_config(path) for path in paths]


def run_calibration_suite(
    release_bundle: ReleaseBundle,
    baseline_run_configs: Sequence[BaselineRunConfig | str | Path],
    sample_count_per_motif: int = 1,
    task_samples_per_class: int = 1,
    severity_levels: Sequence[int] = (0, 1, 2, 3),
    *,
    protocol: EvaluationProtocol | None = None,
    motif_tags: Sequence[str] | None = None,
    family_names: Sequence[str] | None = None,
) -> CalibrationReport:
    """Evaluate calibration baselines on one configured MapShift protocol."""

    active_protocol = protocol or default_post_intervention_protocol()
    run_configs = [
        config if isinstance(config, BaselineRunConfig) else load_baseline_run_config(config)
        for config in baseline_run_configs
    ]
    models = {config.baseline_name: instantiate_baseline(config) for config in run_configs}
    run_contexts = {
        config.baseline_name: BaselineContext(
            model_name=config.baseline_name,
            exploration_budget_steps=config.exploration_budget_steps,
            seed=config.seed,
            release_name=release_bundle.root.release_name,
            protocol_name=active_protocol.name,
        )
        for config in run_configs
    }

    oracle_config = next((config for config in run_configs if config.baseline_name == "oracle_post_intervention_planner"), None)
    if oracle_config is None:
        oracle_config = BaselineRunConfig(
            schema_version="0.1.0",
            run_name="implicit_oracle_reference",
            baseline_name="oracle_post_intervention_planner",
            seed=0,
            exploration_budget_steps=release_bundle.env2d.exploration.canonical_budget_steps,
            parameters={},
        )
    if oracle_config.baseline_name not in models:
        models[oracle_config.baseline_name] = instantiate_baseline(oracle_config)
        run_contexts[oracle_config.baseline_name] = BaselineContext(
            model_name=oracle_config.baseline_name,
            exploration_budget_steps=oracle_config.exploration_budget_steps,
            seed=oracle_config.seed,
            release_name=release_bundle.root.release_name,
            protocol_name=active_protocol.name,
        )
    oracle_model = models[oracle_config.baseline_name]
    oracle_context = run_contexts[oracle_config.baseline_name]

    generator = Map2DGenerator(release_bundle.env2d)
    sampler = TaskSampler(release_bundle.tasks)
    oracle_explorations: dict[str, ExplorationResult] = {}
    explorations: dict[tuple[str, str], ExplorationResult] = {}
    records: list[EvaluationRecord] = []

    motifs = list(motif_tags or release_bundle.env2d.motif_families)
    families = list(family_names or release_bundle.interventions.canonical_family_order)
    for motif_index, motif in enumerate(motifs):
        for sample_index in range(sample_count_per_motif):
            seed = (motif_index + 1) * 100 + sample_index
            base_environment = generator.generate(seed=seed, motif_tag=motif).environment

            for model_name, model in models.items():
                explorations[(model_name, base_environment.environment_id)] = _exploration_for_protocol(
                    model,
                    base_environment,
                    run_contexts[model_name],
                    active_protocol,
                )
            oracle_explorations[base_environment.environment_id] = _exploration_for_protocol(
                oracle_model,
                base_environment,
                oracle_context,
                active_protocol,
            )

            for family in families:
                intervention = build_intervention(family, release_bundle.interventions.families[family])
                for severity in severity_levels:
                    sampled_environment = _environment_for_protocol(base_environment, family, severity, intervention, active_protocol, seed)
                    for task_class, class_config in release_bundle.tasks.classes.items():
                        if not class_config.enabled:
                            continue
                        for task_index in range(task_samples_per_class):
                            task_seed = seed * 1000 + severity * 100 + task_index * 10 + len(task_class)
                            try:
                                sampled = sampler.sample(
                                    base_environment=base_environment,
                                    intervened_environment=sampled_environment,
                                    family=family,
                                    seed=task_seed,
                                    task_class=task_class,
                                )
                            except TaskSamplingRejected:
                                continue

                            evaluation_task = _task_for_protocol(sampled.task, active_protocol)
                            evaluation_environment = sampled_environment if active_protocol.environment_mode == "post_intervention" else base_environment
                            oracle_result = run_evaluation(
                                oracle_model,
                                evaluation_environment,
                                evaluation_task,
                                oracle_explorations[base_environment.environment_id],
                                oracle_context,
                            )
                            for model_name, model in models.items():
                                result = run_evaluation(
                                    model,
                                    evaluation_environment,
                                    evaluation_task,
                                    explorations[(model_name, base_environment.environment_id)],
                                    run_contexts[model_name],
                                )
                                records.append(
                                    EvaluationRecord(
                                        baseline_name=model_name,
                                        protocol_name=active_protocol.name,
                                        family=family,
                                        severity=severity,
                                        split_name=base_environment.split_name,
                                        motif_tag=motif,
                                        task_class=task_class,
                                        task_type=evaluation_task.task_type,
                                        environment_id=evaluation_environment.environment_id,
                                        base_environment_id=base_environment.environment_id,
                                        task_id=sampled.manifest.task_id,
                                        model_seed=run_contexts[model_name].seed,
                                        task_horizon_steps=_task_horizon(evaluation_task),
                                        success=result.success,
                                        solvable=result.solvable,
                                        primary_score=result.primary_score,
                                        observed_length=result.observed_length,
                                        oracle_length=result.oracle_length,
                                        path_efficiency=result.path_efficiency,
                                        oracle_gap=result.oracle_gap,
                                        oracle_success=oracle_result.success,
                                        oracle_primary_score=oracle_result.primary_score,
                                        predicted_answer=result.predicted_answer,
                                        expected_answer=getattr(evaluation_task, "expected_answer", None),
                                        correct=result.correct,
                                        adaptation_curve=tuple(result.adaptation_curve),
                                        metadata=dict(result.metadata or {}),
                                    )
                                )

    baseline_metadata = {name: model.describe() for name, model in models.items()}
    run_manifest_metadata = {
        name: build_run_manifest(
            model=model,
            context=run_contexts[name],
            environment_ids=sorted({record.environment_id for record in records if record.baseline_name == name}),
            baseline_family=model.category,
        ).to_dict()
        for name, model in models.items()
    }

    long_horizon_threshold = _long_horizon_threshold(records)
    metric_tables = _build_metric_tables(records, long_horizon_threshold=long_horizon_threshold)
    bootstrap_summary = _build_bootstrap_summary(records, release_bundle, long_horizon_threshold=long_horizon_threshold)
    ranking_summary = _build_ranking_summary(records, metric_tables["familywise_main_results"], release_bundle)
    severity_summary = _build_severity_summary(metric_tables["severity_response"])
    supplementary_summary = _build_supplementary_summary(metric_tables["familywise_main_results"])
    report_artifacts = _build_report_artifacts(metric_tables, bootstrap_summary, ranking_summary, severity_summary, supplementary_summary)

    return CalibrationReport(
        release_name=release_bundle.root.release_name,
        benchmark_version=release_bundle.root.benchmark_version,
        protocol_name=active_protocol.name,
        sample_count_per_motif=sample_count_per_motif,
        task_samples_per_class=task_samples_per_class,
        baseline_metadata=baseline_metadata,
        run_manifest_metadata=run_manifest_metadata,
        records=tuple(records),
        familywise_summary={
            "rows": metric_tables["familywise_main_results"],
            "by_severity": metric_tables["severity_response"],
            "by_task_class": metric_tables["task_class_summary"],
            "by_split": metric_tables["split_summary"],
        },
        secondary_summary=metric_tables["secondary_diagnostics"],
        bootstrap_summary=bootstrap_summary,
        ranking_summary=ranking_summary,
        severity_summary=severity_summary,
        supplementary_summary=supplementary_summary,
        oracle_solvability_summary=_summarize_oracle(records),
        weak_baseline_summary=_summarize_weak_baseline(records),
        saturation_summary=_summarize_saturation(records),
        impossibility_summary=_summarize_impossibility(records),
        report_artifacts=report_artifacts,
    )


def _exploration_for_protocol(model: Any, base_environment: Any, context: BaselineContext, protocol: EvaluationProtocol) -> ExplorationResult:
    if protocol.exploration_mode == "reward_free":
        return run_exploration(model, base_environment, context)
    start_cell = base_environment.start_cell
    return ExplorationResult(
        baseline_name=model.name,
        environment_id=base_environment.environment_id,
        exploration_steps=0,
        visited_cells=(start_cell,),
        visited_node_ids=(),
        hidden_state=(),
        memory={"visited_ratio": 0.0, "remembered_goal_tokens": {}, "visited_node_ids": ()},
    )


def _environment_for_protocol(base_environment: Any, family: str, severity: int, intervention: Any, protocol: EvaluationProtocol, seed: int) -> Any:
    if protocol.environment_mode == "post_intervention":
        return intervention.apply(base_environment, severity=severity, seed=seed + severity).environment
    cloned = base_environment.clone(environment_id=f"{base_environment.environment_id}-{protocol.name}-s{severity}-{family}")
    cloned.history.append(f"protocol:{protocol.name}")
    return cloned


def _task_for_protocol(task: Any, protocol: EvaluationProtocol) -> Any:
    if protocol.horizon_multiplier == 1.0:
        return task
    if isinstance(task, PlanningTask):
        return replace(task, horizon_steps=max(1, int(round(task.horizon_steps * protocol.horizon_multiplier))))
    if isinstance(task, AdaptationTask):
        return replace(
            task,
            adaptation_budget_steps=max(1, int(round(task.adaptation_budget_steps * protocol.horizon_multiplier))),
            evaluation_horizon_steps=max(1, int(round(task.evaluation_horizon_steps * protocol.horizon_multiplier))),
        )
    if isinstance(task, InferenceTask):
        metadata = dict(task.metadata)
        horizon_steps = int(metadata.get("horizon_steps", 1))
        metadata["horizon_steps"] = max(1, int(round(horizon_steps * protocol.horizon_multiplier)))
        return replace(task, metadata=metadata)
    return task


def _task_horizon(task: Any) -> int:
    if isinstance(task, PlanningTask):
        return task.horizon_steps
    if isinstance(task, AdaptationTask):
        return task.evaluation_horizon_steps
    return int(task.metadata.get("horizon_steps", 1))


def _long_horizon_threshold(records: Sequence[EvaluationRecord]) -> int:
    horizons = sorted(record.task_horizon_steps for record in records if record.task_class == "planning")
    if not horizons:
        return 1
    return horizons[len(horizons) // 2]


def _build_metric_tables(records: Sequence[EvaluationRecord], *, long_horizon_threshold: int) -> dict[str, Any]:
    return {
        "familywise_main_results": _aggregate_metric_rows(records, ("protocol_name", "baseline_name", "family"), long_horizon_threshold),
        "severity_response": _aggregate_metric_rows(records, ("protocol_name", "baseline_name", "family", "severity"), long_horizon_threshold),
        "task_class_summary": _aggregate_metric_rows(records, ("protocol_name", "baseline_name", "family", "task_class"), long_horizon_threshold),
        "split_summary": _aggregate_metric_rows(records, ("protocol_name", "baseline_name", "family", "split_name"), long_horizon_threshold),
        "secondary_diagnostics": {
            "rows": _aggregate_metric_rows(records, ("protocol_name", "baseline_name", "family", "severity"), long_horizon_threshold),
        },
    }


def _aggregate_metric_rows(records: Sequence[EvaluationRecord], group_fields: Sequence[str], long_horizon_threshold: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[EvaluationRecord]] = {}
    for record in records:
        key = tuple(str(getattr(record, field)) for field in group_fields)
        grouped.setdefault(key, []).append(record)

    rows: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        row = {field: value for field, value in zip(group_fields, key)}
        row |= _metric_row(group, long_horizon_threshold=long_horizon_threshold)
        rows.append(row)
    return rows


def _metric_row(records: Sequence[EvaluationRecord], *, long_horizon_threshold: int) -> dict[str, Any]:
    planning_records = [record for record in records if record.task_class == "planning"]
    adaptation_records = [record for record in records if record.task_class == "adaptation"]
    inference_records = [record for record in records if record.task_class == "inference"]
    counterfactual_records = [record for record in planning_records if _is_counterfactual_planning_record(record)]
    change_detection_records = [record for record in inference_records if record.task_type == "detect_topology_change"]
    masked_inference_records = [record for record in inference_records if record.task_type == "predict_masked_region_after_intervention"]
    long_horizon_records = [record for record in planning_records if record.task_horizon_steps >= long_horizon_threshold]
    adaptation_curves = [record.adaptation_curve for record in adaptation_records if record.adaptation_curve]
    adaptation_curve_summaries = [summarize_adaptation_curve(record.adaptation_curve) for record in adaptation_records if record.adaptation_curve]

    planning_success = success_rate(record.success for record in planning_records)
    path_efficiency = mean_or_zero([record.path_efficiency for record in planning_records if record.path_efficiency > 0.0])
    adaptation_efficiency = mean_or_zero([adaptation_sample_efficiency(record.adaptation_curve) for record in adaptation_records if record.adaptation_curve])
    counterfactual_accuracy = (
        counterfactual_planning_accuracy(record.success for record in counterfactual_records)
        if counterfactual_records
        else 0.0
    )
    change_auc = (
        change_detection_auroc(
            [_inference_score(record) for record in change_detection_records],
            [1 if bool(record.expected_answer) else 0 for record in change_detection_records],
        )
        if change_detection_records
        else 0.0
    )
    masked_accuracy = masked_state_inference_accuracy(
        bool(record.correct) for record in masked_inference_records if record.correct is not None
    )
    inference_accuracy = accuracy_from_flags(bool(record.correct) for record in inference_records if record.correct is not None)
    rollout_consistency = long_horizon_rollout_consistency(
        [record.path_efficiency for record in long_horizon_records],
        [record.success for record in long_horizon_records],
    )
    family_primary_score = _family_primary_score_from_metrics(
        planning_success=planning_success,
        path_efficiency=path_efficiency,
        adaptation_efficiency=adaptation_efficiency,
        counterfactual_accuracy=counterfactual_accuracy,
    )

    return {
        "episode_count": len(records),
        "planning_episode_count": len(planning_records),
        "inference_episode_count": len(inference_records),
        "adaptation_episode_count": len(adaptation_records),
        "success_rate": planning_success,
        "normalized_path_efficiency": path_efficiency,
        "adaptation_sample_efficiency": adaptation_efficiency,
        "counterfactual_planning_accuracy": counterfactual_accuracy,
        "family_primary_score": family_primary_score,
        "change_detection_auroc": change_auc,
        "masked_state_inference_accuracy": masked_accuracy,
        "inference_accuracy": inference_accuracy,
        "long_horizon_rollout_consistency": rollout_consistency,
        "oracle_solvable_rate": proportion_true(record.solvable for record in records),
        "oracle_upper_score": mean_or_zero([record.oracle_primary_score for record in records]),
        "mean_primary_score": mean_or_zero([record.primary_score for record in records]),
        "mean_oracle_gap": mean_or_zero([record.oracle_gap for record in records if record.oracle_gap is not None]),
        "adaptation_curve_rows": [summary.to_dict() for summary in adaptation_curve_summaries],
        "adaptation_budget_curve": recovery_vs_budget(adaptation_curves),
    }


def _inference_score(record: EvaluationRecord) -> float:
    confidence = record.metadata.get("recurrent_confidence") if isinstance(record.metadata, dict) else None
    if isinstance(confidence, (int, float)):
        return float(confidence)
    if isinstance(record.predicted_answer, bool):
        return 1.0 if record.predicted_answer else 0.0
    return 1.0 if record.correct else 0.0


def _is_counterfactual_planning_record(record: EvaluationRecord) -> bool:
    return record.task_class == "planning" and record.task_type != "shortest_path_to_target"


def _family_primary_score_from_metrics(
    *,
    planning_success: float,
    path_efficiency: float,
    adaptation_efficiency: float,
    counterfactual_accuracy: float,
) -> float:
    values = [planning_success, path_efficiency, adaptation_efficiency, counterfactual_accuracy]
    return mean_or_zero(values)


def _family_primary_score_from_records(records: Sequence[EvaluationRecord], *, family: str | None = None, long_horizon_threshold: int) -> float:
    selected = [record for record in records if family is None or record.family == family]
    if not selected:
        return 0.0
    return _metric_row(selected, long_horizon_threshold=long_horizon_threshold)["family_primary_score"]


def _build_bootstrap_summary(records: Sequence[EvaluationRecord], release_bundle: ReleaseBundle, *, long_horizon_threshold: int) -> dict[str, Any]:
    bootstrap_config = release_bundle.analysis.bootstrap
    main_rows = []
    severity_rows = []
    for metric_name, statistic in (
        ("success_rate", lambda rows: _metric_row(rows, long_horizon_threshold=long_horizon_threshold)["success_rate"]),
        ("normalized_path_efficiency", lambda rows: _metric_row(rows, long_horizon_threshold=long_horizon_threshold)["normalized_path_efficiency"]),
        ("adaptation_sample_efficiency", lambda rows: _metric_row(rows, long_horizon_threshold=long_horizon_threshold)["adaptation_sample_efficiency"]),
        ("counterfactual_planning_accuracy", lambda rows: _metric_row(rows, long_horizon_threshold=long_horizon_threshold)["counterfactual_planning_accuracy"]),
        ("family_primary_score", lambda rows: _metric_row(rows, long_horizon_threshold=long_horizon_threshold)["family_primary_score"]),
    ):
        main_rows.extend(
            row.to_dict()
            for row in bootstrap_grouped_metric(
                records,
                group_fields=("protocol_name", "baseline_name", "family"),
                unit_field=bootstrap_config.paired_by,
                metric_name=metric_name,
                statistic=statistic,
                resamples=bootstrap_config.resamples,
                confidence_level=bootstrap_config.confidence_level,
                seed=0,
            )
        )
        severity_rows.extend(
            row.to_dict()
            for row in bootstrap_grouped_metric(
                records,
                group_fields=("protocol_name", "baseline_name", "family", "severity"),
                unit_field=bootstrap_config.paired_by,
                metric_name=metric_name,
                statistic=statistic,
                resamples=bootstrap_config.resamples,
                confidence_level=bootstrap_config.confidence_level,
                seed=101,
            )
        )
    return {
        "config": {
            "resamples": bootstrap_config.resamples,
            "confidence_level": bootstrap_config.confidence_level,
            "paired_by": bootstrap_config.paired_by,
        },
        "familywise_main_results": main_rows,
        "severity_response": severity_rows,
    }


def _build_ranking_summary(records: Sequence[EvaluationRecord], family_rows: Sequence[dict[str, Any]], release_bundle: ReleaseBundle) -> dict[str, Any]:
    protocols = sorted({record.protocol_name for record in records})
    long_horizon_threshold = _long_horizon_threshold(records)
    family_orders = []
    family_spreads = []
    pooled_orders = []
    average_family_orders = []
    leave_one_family_out = []
    rank_stability = []

    for protocol_name in protocols:
        protocol_rows = [row for row in family_rows if row["protocol_name"] == protocol_name]
        families = sorted({str(row["family"]) for row in protocol_rows})
        family_order_map: dict[str, list[str]] = {}
        for family in families:
            metric_values = {
                str(row["baseline_name"]): float(row["family_primary_score"])
                for row in protocol_rows
                if row["family"] == family
            }
            order = rank_by_metric(metric_values)
            family_order_map[family] = order
            family_orders.append(
                {
                    "protocol_name": protocol_name,
                    "family": family,
                    "order": order,
                    "kendall_self": kendall_tau(order, order),
                }
            )
            family_spreads.append(
                {
                    "protocol_name": protocol_name,
                    "family": family,
                    "spread": ranking_spread(metric_values),
                    "stddev": ranking_stddev(metric_values),
                }
            )
            sampled_orders = _bootstrap_rank_orders(
                records,
                protocol_name=protocol_name,
                family=family,
                resamples=release_bundle.analysis.bootstrap.resamples,
                seed=17,
                long_horizon_threshold=long_horizon_threshold,
            )
            rank_stability.append(
                summarize_rank_samples(
                    protocol_name=protocol_name,
                    family=family,
                    point_estimates=metric_values,
                    sampled_orders=sampled_orders,
                ).to_dict()
            )

        pooled_scores = _pooled_scores(protocol_rows)
        pooled_order = rank_by_metric(pooled_scores)
        pooled_orders.append(
            {
                "protocol_name": protocol_name,
                "order": pooled_order,
                "scores": pooled_scores,
            }
        )
        average_rank_scores = _average_family_rank_scores(family_order_map)
        average_family_order = rank_by_metric({name: -score for name, score in average_rank_scores.items()})
        average_family_orders.append(
            {
                "protocol_name": protocol_name,
                "order": average_family_order,
                "average_family_rank": average_rank_scores,
            }
        )
        for excluded_family in families:
            remaining_rows = [row for row in protocol_rows if row["family"] != excluded_family]
            leave_scores = _pooled_scores(remaining_rows)
            leave_one_family_out.append(
                {
                    "protocol_name": protocol_name,
                    "excluded_family": excluded_family,
                    "order": rank_by_metric(leave_scores),
                    "scores": leave_scores,
                }
            )
        sampled_orders = _bootstrap_rank_orders(
            records,
            protocol_name=protocol_name,
            family=None,
            resamples=release_bundle.analysis.bootstrap.resamples,
            seed=99,
            long_horizon_threshold=long_horizon_threshold,
        )
        rank_stability.append(
            summarize_rank_samples(
                protocol_name=protocol_name,
                family="pooled_supplementary",
                point_estimates=pooled_scores,
                sampled_orders=sampled_orders,
            ).to_dict()
        )

    return {
        "family_orders": family_orders,
        "family_spreads": family_spreads,
        "pooled_supplementary_orders": pooled_orders,
        "average_family_rank_orders": average_family_orders,
        "leave_one_family_out": leave_one_family_out,
        "rank_stability": rank_stability,
    }


def _bootstrap_rank_orders(
    records: Sequence[EvaluationRecord],
    *,
    protocol_name: str,
    family: str | None,
    resamples: int,
    seed: int,
    long_horizon_threshold: int,
) -> list[list[str]]:
    filtered = [record for record in records if record.protocol_name == protocol_name and (family is None or record.family == family)]
    grouped: dict[str, list[EvaluationRecord]] = {}
    for record in filtered:
        grouped.setdefault(record.base_environment_id, []).append(record)
    unit_ids = sorted(grouped)
    if not unit_ids:
        return []
    rng = random.Random(seed)
    orders: list[list[str]] = []
    for _ in range(resamples):
        sample: list[EvaluationRecord] = []
        for _unit_index in range(len(unit_ids)):
            sample.extend(grouped[unit_ids[rng.randrange(len(unit_ids))]])
        if family is None:
            scores = _pooled_scores_from_records(sample, long_horizon_threshold=long_horizon_threshold)
        else:
            scores = _scores_for_family(sample, family=family, long_horizon_threshold=long_horizon_threshold)
        orders.append(rank_by_metric(scores))
    return orders


def _scores_for_family(records: Sequence[EvaluationRecord], *, family: str, long_horizon_threshold: int) -> dict[str, float]:
    grouped: dict[str, list[EvaluationRecord]] = {}
    for record in records:
        if record.family != family:
            continue
        grouped.setdefault(record.baseline_name, []).append(record)
    return {
        baseline_name: _family_primary_score_from_records(group, family=family, long_horizon_threshold=long_horizon_threshold)
        for baseline_name, group in sorted(grouped.items())
    }


def _pooled_scores(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["baseline_name"]), []).append(float(row["family_primary_score"]))
    return {baseline_name: mean_or_zero(scores) for baseline_name, scores in sorted(grouped.items())}


def _pooled_scores_from_records(records: Sequence[EvaluationRecord], *, long_horizon_threshold: int) -> dict[str, float]:
    grouped: dict[tuple[str, str], list[EvaluationRecord]] = {}
    for record in records:
        grouped.setdefault((record.baseline_name, record.family), []).append(record)
    per_family_scores: dict[str, list[float]] = {}
    for (baseline_name, family), group in sorted(grouped.items()):
        per_family_scores.setdefault(baseline_name, []).append(
            _family_primary_score_from_records(group, family=family, long_horizon_threshold=long_horizon_threshold)
        )
    return {baseline_name: mean_or_zero(scores) for baseline_name, scores in sorted(per_family_scores.items())}


def _average_family_rank_scores(family_order_map: dict[str, list[str]]) -> dict[str, float]:
    rank_lists: dict[str, list[int]] = {}
    for order in family_order_map.values():
        for method, position in rank_positions(order).items():
            rank_lists.setdefault(method, []).append(position)
    return {method: mean_or_zero(positions) for method, positions in sorted(rank_lists.items())}


def _build_severity_summary(severity_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in severity_rows:
        grouped.setdefault((str(row["protocol_name"]), str(row["baseline_name"]), str(row["family"])), []).append(row)
    trends = []
    for (protocol_name, baseline_name, family), rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: int(row["severity"]))
        family_scores = [float(row["family_primary_score"]) for row in ordered]
        monotone = all(left >= right for left, right in zip(family_scores, family_scores[1:]))
        trends.append(
            {
                "protocol_name": protocol_name,
                "baseline_name": baseline_name,
                "family": family,
                "monotone_degradation": monotone,
                "severity_values": [int(row["severity"]) for row in ordered],
                "family_primary_scores": family_scores,
            }
        )
    return {"rows": trends}


def _build_supplementary_summary(family_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in family_rows:
        grouped.setdefault((str(row["protocol_name"]), str(row["baseline_name"])), []).append(row)
    rows = []
    for (protocol_name, baseline_name), group in sorted(grouped.items()):
        rows.append(
            {
                "protocol_name": protocol_name,
                "baseline_name": baseline_name,
                "supplementary_pooled_score": mean_or_zero([float(row["family_primary_score"]) for row in group]),
                "family_count": len(group),
                "label": "supplementary_only",
            }
        )
    return {"pooled_supplementary_rows": rows}


def _build_report_artifacts(
    metric_tables: dict[str, Any],
    bootstrap_summary: dict[str, Any],
    ranking_summary: dict[str, Any],
    severity_summary: dict[str, Any],
    supplementary_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "tables": {
            "familywise_main_results": metric_tables["familywise_main_results"],
            "severity_response": metric_tables["severity_response"],
            "task_class_summary": metric_tables["task_class_summary"],
            "split_summary": metric_tables["split_summary"],
            "bootstrap_familywise_main_results": bootstrap_summary["familywise_main_results"],
            "bootstrap_severity_response": bootstrap_summary["severity_response"],
            "ranking_spread_and_stability": ranking_summary,
            "supplementary_pooled_score": supplementary_summary["pooled_supplementary_rows"],
        },
        "figures": {
            "familywise_degradation_curves": metric_tables["severity_response"],
            "protocol_ranking_comparison": ranking_summary,
            "bootstrap_summary_table": bootstrap_summary,
        },
    }


def _summarize_oracle(records: Iterable[EvaluationRecord]) -> dict[str, Any]:
    oracle_records = [record for record in records if record.baseline_name == "oracle_post_intervention_planner"]
    grouped: dict[tuple[str, int], list[EvaluationRecord]] = {}
    for record in oracle_records:
        grouped.setdefault((record.family, record.severity), []).append(record)
    rows = []
    for (family, severity), group in sorted(grouped.items()):
        rows.append(
            {
                "family": family,
                "severity": severity,
                "count": len(group),
                "oracle_success_rate": proportion_true(record.success for record in group),
                "oracle_solvable_rate": proportion_true(record.solvable for record in group),
            }
        )
    return {"rows": rows}


def _summarize_weak_baseline(records: Iterable[EvaluationRecord]) -> dict[str, Any]:
    weak_records = [record for record in records if record.baseline_name == "weak_heuristic_baseline"]
    grouped: dict[tuple[str, int], list[EvaluationRecord]] = {}
    for record in weak_records:
        grouped.setdefault((record.family, record.severity), []).append(record)
    rows = []
    for (family, severity), group in sorted(grouped.items()):
        rows.append(
            {
                "family": family,
                "severity": severity,
                "count": len(group),
                "weak_success_rate": proportion_true(record.success for record in group),
                "weak_mean_primary_score": mean_or_zero([record.primary_score for record in group]),
            }
        )
    return {"rows": rows}


def _summarize_saturation(records: Iterable[EvaluationRecord]) -> dict[str, Any]:
    weak_records = [record for record in records if record.baseline_name == "weak_heuristic_baseline"]
    grouped: dict[tuple[str, int], list[EvaluationRecord]] = {}
    for record in weak_records:
        grouped.setdefault((record.family, record.severity), []).append(record)
    saturated_cells = []
    for (family, severity), group in sorted(grouped.items()):
        weak_rate = proportion_true(record.success for record in group)
        oracle_rate = mean_or_zero([record.oracle_primary_score for record in group])
        if weak_rate >= 0.8 and oracle_rate >= 0.9:
            saturated_cells.append(
                {
                    "family": family,
                    "severity": severity,
                    "weak_success_rate": weak_rate,
                    "oracle_primary_score": oracle_rate,
                }
            )
    return {"saturated_cells": saturated_cells, "saturated_cell_count": len(saturated_cells)}


def _summarize_impossibility(records: Iterable[EvaluationRecord]) -> dict[str, Any]:
    oracle_records = [record for record in records if record.baseline_name == "oracle_post_intervention_planner"]
    grouped: dict[tuple[str, int], list[EvaluationRecord]] = {}
    for record in oracle_records:
        grouped.setdefault((record.family, record.severity), []).append(record)
    impossible_cells = []
    for (family, severity), group in sorted(grouped.items()):
        oracle_rate = proportion_true(record.success for record in group)
        if oracle_rate <= 0.0:
            impossible_cells.append({"family": family, "severity": severity, "count": len(group)})
    return {"impossible_cells": impossible_cells, "impossible_cell_count": len(impossible_cells)}
