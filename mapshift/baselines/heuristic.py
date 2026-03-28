"""Weak heuristic calibration baseline for MapShift-2D."""

from __future__ import annotations

from typing import Any

from mapshift.tasks.adaptation import AdaptationTask
from mapshift.tasks.inference import InferenceTask
from mapshift.tasks.planning import PlanningTask

from .api import (
    BaseBaselineModel,
    BaselineContext,
    ExplorationResult,
    TaskEvaluationResult,
    deterministic_exploration_trace,
    register_baseline,
    summarize_exploration_memory,
)
from .common import (
    dynamics_cost_multiplier,
    family_shift_severity,
    goal_node_for_task,
    horizon_allows_path,
    orthogonal_heuristic_path,
    path_efficiency_from_lengths,
)


class WeakHeuristicBaseline(BaseBaselineModel):
    """Limited deterministic baseline without map revision."""

    name = "weak_heuristic_baseline"
    category = "heuristic"
    parameter_count = 0
    trainable_parameter_count = 0

    def explore(self, environment: Any, context: BaselineContext) -> ExplorationResult:
        visited_cells, visited_node_ids = deterministic_exploration_trace(environment, context.exploration_budget_steps, context.seed)
        memory = summarize_exploration_memory(environment, visited_cells, visited_node_ids)
        memory["heuristic_policy"] = "orthogonal_visible_path"
        return ExplorationResult(
            baseline_name=self.name,
            environment_id=environment.environment_id,
            exploration_steps=len(visited_cells),
            visited_cells=visited_cells,
            visited_node_ids=visited_node_ids,
            memory=memory,
        )

    def evaluate(self, environment: Any, task: Any, exploration: ExplorationResult, context: BaselineContext) -> TaskEvaluationResult:
        if isinstance(task, PlanningTask):
            return self._evaluate_planning(environment, task, exploration)
        if isinstance(task, InferenceTask):
            return self._evaluate_inference(task, exploration)
        if isinstance(task, AdaptationTask):
            return self._evaluate_adaptation(environment, task, exploration)
        raise TypeError(f"Unsupported task type for weak heuristic baseline: {type(task).__name__}")

    def _evaluate_planning(
        self,
        environment: Any,
        task: PlanningTask,
        exploration: ExplorationResult,
    ) -> TaskEvaluationResult:
        remembered_tokens = exploration.memory.get("remembered_goal_tokens", {})
        goal_node, goal_matches_task = goal_node_for_task(environment, task, remembered_goal_tokens=remembered_tokens)
        heuristic_path = None if goal_node is None else orthogonal_heuristic_path(environment, task.start_node_id, goal_node)
        oracle_length = None if task.goal_node_id is None else environment.shortest_path_length(task.start_node_id, task.goal_node_id)
        observed_length = None
        if heuristic_path is not None:
            observed_length = max(0, len(heuristic_path) - 1) * environment.occupancy_resolution_m * environment.geometry_scale
            observed_length *= dynamics_cost_multiplier(environment)
        success = (
            goal_node is not None
            and goal_matches_task
            and heuristic_path is not None
            and horizon_allows_path(task, observed_length, environment)
        )
        return TaskEvaluationResult(
            baseline_name=self.name,
            task_class="planning",
            task_type=task.task_type,
            family=task.family,
            success=success,
            solvable=oracle_length is not None,
            observed_length=observed_length,
            oracle_length=oracle_length,
            path_efficiency=path_efficiency_from_lengths(oracle_length, observed_length),
            oracle_gap=None if oracle_length is None or observed_length is None else max(0.0, observed_length - oracle_length),
            path_cells=tuple(heuristic_path or ()),
            metadata={
                "goal_matches_task": goal_matches_task,
                "used_stale_semantics": bool(task.goal_token and not goal_matches_task),
                "shift_severity": family_shift_severity(environment, task.family),
                "planner": "orthogonal_visible_path",
            },
        )

    def _evaluate_inference(self, task: InferenceTask, exploration: ExplorationResult) -> TaskEvaluationResult:
        predicted_answer: Any
        if task.task_type == "detect_topology_change":
            predicted_answer = False
        elif task.task_type == "counterfactual_reachability_query":
            remembered_tokens = exploration.memory.get("remembered_goal_tokens", {})
            token = ""
            for piece in task.query.split():
                if piece.startswith("target_"):
                    token = piece
                    break
            predicted_answer = remembered_tokens.get(token)
        else:
            predicted_answer = task.family
        correct = predicted_answer == task.expected_answer
        return TaskEvaluationResult(
            baseline_name=self.name,
            task_class="inference",
            task_type=task.task_type,
            family=task.family,
            success=bool(correct),
            solvable=True,
            predicted_answer=predicted_answer,
            correct=bool(correct),
            metadata={"heuristic_rule": task.task_type},
        )

    def _evaluate_adaptation(
        self,
        environment: Any,
        task: AdaptationTask,
        exploration: ExplorationResult,
    ) -> TaskEvaluationResult:
        proxy_planning = PlanningTask(
            task_type=task.task_type,
            horizon_steps=task.evaluation_horizon_steps,
            family=task.family,
            start_node_id=task.start_node_id,
            goal_node_id=task.goal_node_id,
            goal_token=None,
            goal_descriptor=task.goal_node_id,
            metadata=dict(task.metadata),
        )
        planning_result = self._evaluate_planning(environment, proxy_planning, exploration)
        family_gain = 0.25 if task.family in {"metric", "dynamics"} else 0.10
        if planning_result.success:
            curve = (1.0, 1.0, 1.0)
        else:
            mid_score = min(0.5, family_gain * max(1, task.adaptation_budget_steps) / max(1, task.evaluation_horizon_steps))
            final_score = min(0.75, mid_score + family_gain * 0.5)
            curve = (0.0, mid_score, final_score)
        return TaskEvaluationResult(
            baseline_name=self.name,
            task_class="adaptation",
            task_type=task.task_type,
            family=task.family,
            success=bool(curve[-1] >= 0.999),
            solvable=planning_result.solvable,
            observed_length=planning_result.observed_length,
            oracle_length=planning_result.oracle_length,
            path_efficiency=planning_result.path_efficiency,
            oracle_gap=planning_result.oracle_gap,
            adaptation_curve=curve,
            path_cells=planning_result.path_cells,
            metadata={"adaptation_gain": family_gain, "base_success": planning_result.success},
        )


register_baseline(WeakHeuristicBaseline.name, WeakHeuristicBaseline)
