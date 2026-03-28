"""Oracle planner calibration baseline for MapShift-2D."""

from __future__ import annotations

from typing import Any

from mapshift.tasks.adaptation import AdaptationTask
from mapshift.tasks.inference import InferenceTask
from mapshift.tasks.planning import PlanningTask

from .api import (
    BaseBaselineModel,
    BaselineContext,
    BaselineRunConfig,
    ExplorationResult,
    TaskEvaluationResult,
    deterministic_exploration_trace,
    register_baseline,
    summarize_exploration_memory,
    task_class_name,
)
from .common import dynamics_cost_multiplier, horizon_allows_path, path_efficiency_from_lengths


class OraclePlannerBaseline(BaseBaselineModel):
    """Upper-reference planner with full access to the intervened substrate."""

    name = "oracle_post_intervention_planner"
    category = "oracle"
    parameter_count = 0
    trainable_parameter_count = 0

    def explore(self, environment: Any, context: BaselineContext) -> ExplorationResult:
        visited_cells, visited_node_ids = deterministic_exploration_trace(environment, context.exploration_budget_steps, context.seed)
        return ExplorationResult(
            baseline_name=self.name,
            environment_id=environment.environment_id,
            exploration_steps=len(visited_cells),
            visited_cells=visited_cells,
            visited_node_ids=visited_node_ids,
            memory=summarize_exploration_memory(environment, visited_cells, visited_node_ids),
        )

    def evaluate(self, environment: Any, task: Any, exploration: ExplorationResult, context: BaselineContext) -> TaskEvaluationResult:
        if isinstance(task, PlanningTask):
            return self._evaluate_planning(environment, task)
        if isinstance(task, InferenceTask):
            return self._evaluate_inference(task)
        if isinstance(task, AdaptationTask):
            return self._evaluate_adaptation(environment, task)
        raise TypeError(f"Unsupported task type for oracle baseline: {type(task).__name__}")

    def _evaluate_planning(self, environment: Any, task: PlanningTask) -> TaskEvaluationResult:
        goal_node = task.goal_node_id
        path = None if goal_node is None else environment.oracle_shortest_path(task.start_node_id, goal_node)
        oracle_length = None if goal_node is None else environment.shortest_path_length(task.start_node_id, goal_node)
        observed_length = None
        if oracle_length is not None:
            observed_length = oracle_length * dynamics_cost_multiplier(environment)
        success = goal_node is not None and path is not None and horizon_allows_path(task, observed_length, environment)
        return TaskEvaluationResult(
            baseline_name=self.name,
            task_class="planning",
            task_type=task.task_type,
            family=task.family,
            success=success,
            solvable=path is not None,
            observed_length=observed_length,
            oracle_length=oracle_length,
            path_efficiency=path_efficiency_from_lengths(oracle_length, observed_length),
            oracle_gap=0.0 if observed_length is not None and oracle_length is not None else None,
            path_cells=tuple(path or ()),
            metadata={
                "impossible_for_oracle": path is None,
                "dynamics_multiplier": dynamics_cost_multiplier(environment),
            },
        )

    def _evaluate_inference(self, task: InferenceTask) -> TaskEvaluationResult:
        return TaskEvaluationResult(
            baseline_name=self.name,
            task_class="inference",
            task_type=task.task_type,
            family=task.family,
            success=True,
            solvable=True,
            predicted_answer=task.expected_answer,
            correct=True,
            metadata={"oracle_answer": True},
        )

    def _evaluate_adaptation(self, environment: Any, task: AdaptationTask) -> TaskEvaluationResult:
        path = environment.oracle_shortest_path(task.start_node_id, task.goal_node_id)
        oracle_length = environment.shortest_path_length(task.start_node_id, task.goal_node_id)
        success = path is not None and horizon_allows_path(task, oracle_length, environment)
        curve = (1.0, 1.0, 1.0) if success else (0.0, 0.0, 0.0)
        return TaskEvaluationResult(
            baseline_name=self.name,
            task_class="adaptation",
            task_type=task.task_type,
            family=task.family,
            success=success,
            solvable=path is not None,
            observed_length=oracle_length,
            oracle_length=oracle_length,
            path_efficiency=1.0 if success else 0.0,
            oracle_gap=0.0 if success else None,
            adaptation_curve=curve,
            path_cells=tuple(path or ()),
            metadata={"adaptation_budget_steps": task.adaptation_budget_steps},
        )


class SameEnvironmentUpperBaseline(OraclePlannerBaseline):
    """Alias baseline used as a same-environment upper-reference placeholder."""

    name = "same_environment_upper_baseline"
    category = "oracle_reference"


register_baseline(OraclePlannerBaseline.name, OraclePlannerBaseline)
register_baseline(SameEnvironmentUpperBaseline.name, SameEnvironmentUpperBaseline)
