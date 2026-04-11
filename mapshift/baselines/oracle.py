"""Oracle planner calibration baseline for MapShift-2D."""

from __future__ import annotations

from collections import deque
from typing import Any

from mapshift.envs.procthor.observation import observe_scene
from mapshift.envs.procthor.wrappers import ProcTHORPose, ProcTHORScene
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
    supported_tiers = ("mapshift_2d", "mapshift_3d")

    def explore(self, environment: Any, context: BaselineContext) -> ExplorationResult:
        if isinstance(environment, ProcTHORScene):
            return self._explore_procthor(environment, context)
        visited_cells, visited_node_ids = deterministic_exploration_trace(environment, context.exploration_budget_steps, context.seed)
        return ExplorationResult(
            baseline_name=self.name,
            environment_id=environment.environment_id,
            exploration_steps=len(visited_cells),
            visited_cells=visited_cells,
            visited_node_ids=visited_node_ids,
            memory=summarize_exploration_memory(environment, visited_cells, visited_node_ids),
        )

    def _explore_procthor(self, scene: ProcTHORScene, context: BaselineContext) -> ExplorationResult:
        frontier: deque[str] = deque([scene.start_node_id])
        seen = {scene.start_node_id}
        visited_rooms: list[str] = []
        visible_objects: set[str] = set()
        visible_categories: set[str] = set()
        remembered_goal_tokens: dict[str, str] = {}
        max_steps = max(1, min(context.exploration_budget_steps, scene.room_count() * 4))

        while frontier and len(visited_rooms) < max_steps:
            room_id = frontier.popleft()
            visited_rooms.append(room_id)
            frame = observe_scene(scene, ProcTHORPose(room_id=room_id, x_m=0.0, z_m=0.0))
            visible_objects.update(frame.visible_objects)
            visible_categories.update(frame.visible_categories)
            for object_id in frame.visible_objects:
                obj = scene.object_by_id(object_id)
                if obj.semantic_token:
                    remembered_goal_tokens[obj.semantic_token] = obj.room_id
            for neighbor in scene.neighbors(room_id):
                if neighbor not in seen:
                    seen.add(neighbor)
                    frontier.append(neighbor)

        remembered_graph = {
            room_id: tuple(sorted(neighbor for neighbor in scene.neighbors(room_id) if neighbor in seen))
            for room_id in sorted(seen)
        }
        return ExplorationResult(
            baseline_name=self.name,
            environment_id=scene.environment_id,
            exploration_steps=len(visited_rooms),
            visited_cells=(),
            visited_node_ids=tuple(visited_rooms),
            memory={
                "visited_ratio": len(seen) / max(1, scene.room_count()),
                "remembered_goal_tokens": dict(sorted(remembered_goal_tokens.items())),
                "remembered_room_graph": remembered_graph,
                "visible_objects": tuple(sorted(visible_objects)),
                "visible_categories": tuple(sorted(visible_categories)),
                "base_scene_payload": scene.to_dict(),
            },
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
        if isinstance(environment, ProcTHORScene):
            return self._evaluate_planning_procthor(environment, task)
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

    def _evaluate_planning_procthor(self, scene: ProcTHORScene, task: PlanningTask) -> TaskEvaluationResult:
        goal_node = task.goal_node_id
        path_rooms = None if goal_node is None else scene.shortest_path(task.start_node_id, goal_node)
        oracle_length = None if goal_node is None else scene.shortest_path_length(task.start_node_id, goal_node)
        observed_length = None if oracle_length is None else oracle_length * dynamics_cost_multiplier(scene)
        success = goal_node is not None and path_rooms is not None and horizon_allows_path(task, observed_length, scene)
        return TaskEvaluationResult(
            baseline_name=self.name,
            task_class="planning",
            task_type=task.task_type,
            family=task.family,
            success=success,
            solvable=path_rooms is not None,
            observed_length=observed_length,
            oracle_length=oracle_length,
            path_efficiency=path_efficiency_from_lengths(oracle_length, observed_length),
            oracle_gap=0.0 if observed_length is not None and oracle_length is not None else None,
            path_cells=(),
            metadata={
                "impossible_for_oracle": path_rooms is None,
                "dynamics_multiplier": dynamics_cost_multiplier(scene),
                "path_rooms": tuple(path_rooms or ()),
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
        if isinstance(environment, ProcTHORScene):
            return self._evaluate_adaptation_procthor(environment, task)
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

    def _evaluate_adaptation_procthor(self, scene: ProcTHORScene, task: AdaptationTask) -> TaskEvaluationResult:
        path_rooms = scene.shortest_path(task.start_node_id, task.goal_node_id)
        oracle_length = scene.shortest_path_length(task.start_node_id, task.goal_node_id)
        observed_length = None if oracle_length is None else oracle_length * dynamics_cost_multiplier(scene)
        success = path_rooms is not None and horizon_allows_path(task, observed_length, scene)
        curve = (1.0, 1.0, 1.0) if success else (0.0, 0.0, 0.0)
        return TaskEvaluationResult(
            baseline_name=self.name,
            task_class="adaptation",
            task_type=task.task_type,
            family=task.family,
            success=success,
            solvable=path_rooms is not None,
            observed_length=observed_length,
            oracle_length=oracle_length,
            path_efficiency=path_efficiency_from_lengths(oracle_length, observed_length),
            oracle_gap=0.0 if oracle_length is not None else None,
            adaptation_curve=curve,
            path_cells=(),
            metadata={
                "adaptation_budget_steps": task.adaptation_budget_steps,
                "path_rooms": tuple(path_rooms or ()),
            },
        )


class SameEnvironmentUpperBaseline(OraclePlannerBaseline):
    """Alias baseline used as a same-environment upper-reference placeholder."""

    name = "same_environment_upper_baseline"
    category = "oracle_reference"


register_baseline(OraclePlannerBaseline.name, OraclePlannerBaseline)
register_baseline(SameEnvironmentUpperBaseline.name, SameEnvironmentUpperBaseline)
