"""Weak heuristic calibration baseline for MapShift-2D."""

from __future__ import annotations

import math
from typing import Any

from mapshift.envs.map2d.state import Cell, Map2DEnvironment
from mapshift.envs.procthor.observation import observe_scene
from mapshift.envs.procthor.wrappers import ProcTHORPose, ProcTHORScene
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
    graph_shortest_path,
    goal_node_for_task,
    horizon_allows_path,
    orthogonal_heuristic_path,
    path_efficiency_from_lengths,
    room_path_length,
)


class WeakHeuristicBaseline(BaseBaselineModel):
    """Limited deterministic baseline without map revision."""

    name = "weak_heuristic_baseline"
    category = "heuristic"
    parameter_count = 0
    trainable_parameter_count = 0
    supported_tiers = ("mapshift_2d", "mapshift_3d")

    def explore(self, environment: Any, context: BaselineContext) -> ExplorationResult:
        if isinstance(environment, ProcTHORScene):
            return self._explore_procthor(environment, context)
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

    def _explore_procthor(self, scene: ProcTHORScene, context: BaselineContext) -> ExplorationResult:
        frontier = [scene.start_node_id]
        seen = {scene.start_node_id}
        visited_rooms: list[str] = []
        tree_graph: dict[str, set[str]] = {scene.start_node_id: set()}
        remembered_goal_tokens: dict[str, str] = {}
        max_steps = max(1, min(context.exploration_budget_steps, scene.room_count() * 2))

        while frontier and len(visited_rooms) < max_steps:
            room_id = frontier.pop(0)
            visited_rooms.append(room_id)
            frame = observe_scene(scene, ProcTHORPose(room_id=room_id, x_m=0.0, z_m=0.0))
            for object_id in frame.visible_objects:
                obj = scene.object_by_id(object_id)
                if obj.semantic_token and obj.semantic_token not in remembered_goal_tokens:
                    remembered_goal_tokens[obj.semantic_token] = obj.room_id
            for neighbor in scene.neighbors(room_id):
                tree_graph.setdefault(room_id, set())
                tree_graph.setdefault(neighbor, set())
                if neighbor not in seen:
                    seen.add(neighbor)
                    frontier.append(neighbor)
                    tree_graph[room_id].add(neighbor)
                    tree_graph[neighbor].add(room_id)

        remembered_room_graph = {
            room_id: tuple(sorted(neighbors))
            for room_id, neighbors in sorted(tree_graph.items())
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
                "remembered_room_graph": remembered_room_graph,
                "visited_node_ids": tuple(visited_rooms),
                "heuristic_policy": "remembered_room_tree",
            },
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
        if isinstance(environment, ProcTHORScene):
            return self._evaluate_planning_procthor(environment, task, exploration)
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

    def _evaluate_planning_procthor(
        self,
        scene: ProcTHORScene,
        task: PlanningTask,
        exploration: ExplorationResult,
    ) -> TaskEvaluationResult:
        remembered_tokens = exploration.memory.get("remembered_goal_tokens", {})
        remembered_graph = self._remembered_room_graph(scene, exploration)
        goal_node, goal_matches_task = goal_node_for_task(scene, task, remembered_goal_tokens=remembered_tokens)
        path_rooms = None if goal_node is None else graph_shortest_path(remembered_graph, task.start_node_id, goal_node)
        oracle_length = None if task.goal_node_id is None else scene.shortest_path_length(task.start_node_id, task.goal_node_id)
        blocked_edge = bool(path_rooms) and not self._path_valid_in_scene(scene, path_rooms)
        observed_length = None
        if path_rooms is not None and not blocked_edge:
            observed_length = room_path_length(scene, path_rooms)
            if observed_length is not None:
                observed_length *= dynamics_cost_multiplier(scene)
        success = (
            goal_node is not None
            and goal_matches_task
            and path_rooms is not None
            and not blocked_edge
            and horizon_allows_path(task, observed_length, scene)
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
            path_cells=(),
            metadata={
                "goal_matches_task": goal_matches_task,
                "used_stale_semantics": bool(task.goal_token and not goal_matches_task),
                "shift_severity": family_shift_severity(scene, task.family),
                "planner": "remembered_room_graph",
                "blocked_edge_encountered": blocked_edge,
                "path_rooms": tuple(path_rooms or ()),
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
        if isinstance(environment, ProcTHORScene):
            return self._evaluate_adaptation_procthor(environment, task, exploration)
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

    def _evaluate_adaptation_procthor(
        self,
        scene: ProcTHORScene,
        task: AdaptationTask,
        exploration: ExplorationResult,
    ) -> TaskEvaluationResult:
        remembered_graph = self._remembered_room_graph(scene, exploration)
        diff_operations = self._graph_diff_operations(remembered_graph, scene)
        total_updates = min(len(diff_operations), max(1, task.adaptation_budget_steps))
        checkpoints = (
            0,
            min(total_updates, int(math.ceil(total_updates / 2))),
            total_updates,
        )
        oracle_length = scene.shortest_path_length(task.start_node_id, task.goal_node_id)
        curve: list[float] = []
        final_path: tuple[str, ...] | None = None
        final_observed_length: float | None = None

        for applied_count in checkpoints:
            adapted_graph = self._apply_graph_updates(remembered_graph, diff_operations[:applied_count])
            path_rooms = graph_shortest_path(adapted_graph, task.start_node_id, task.goal_node_id)
            final_path = path_rooms
            observed_length = None
            if path_rooms is not None and self._path_valid_in_scene(scene, path_rooms):
                observed_length = room_path_length(scene, path_rooms)
                if observed_length is not None:
                    observed_length *= dynamics_cost_multiplier(scene)
            final_observed_length = observed_length
            if path_rooms is None or observed_length is None or not horizon_allows_path(task, observed_length, scene):
                curve.append(0.0)
            else:
                curve.append(path_efficiency_from_lengths(oracle_length, observed_length))

        return TaskEvaluationResult(
            baseline_name=self.name,
            task_class="adaptation",
            task_type=task.task_type,
            family=task.family,
            success=bool(curve and curve[-1] >= 0.999),
            solvable=oracle_length is not None,
            observed_length=final_observed_length,
            oracle_length=oracle_length,
            path_efficiency=path_efficiency_from_lengths(oracle_length, final_observed_length),
            oracle_gap=None if oracle_length is None or final_observed_length is None else max(0.0, final_observed_length - oracle_length),
            adaptation_curve=tuple(curve),
            path_cells=(),
            metadata={
                "applied_graph_updates": total_updates,
                "candidate_graph_updates": len(diff_operations),
                "path_rooms": tuple(final_path or ()),
            },
        )

    def _remembered_room_graph(self, scene: ProcTHORScene, exploration: ExplorationResult) -> dict[str, tuple[str, ...]]:
        remembered = exploration.memory.get("remembered_room_graph", {})
        if isinstance(remembered, dict) and remembered:
            return {
                str(room_id): tuple(sorted(str(neighbor) for neighbor in neighbors))
                for room_id, neighbors in remembered.items()
            }
        return {room_id: tuple() for room_id in scene.rooms}

    def _path_valid_in_scene(self, scene: ProcTHORScene, path_rooms: tuple[str, ...]) -> bool:
        for left, right in zip(path_rooms, path_rooms[1:]):
            if right not in scene.neighbors(left):
                return False
        return True

    def _graph_diff_operations(
        self,
        remembered_graph: dict[str, tuple[str, ...]],
        scene: ProcTHORScene,
    ) -> list[tuple[str, str, str]]:
        remembered_edges = {
            tuple(sorted((left, right)))
            for left, neighbors in remembered_graph.items()
            for right in neighbors
            if left != right
        }
        true_edges = set(scene.edge_list())
        additions = [("add", left, right) for left, right in sorted(true_edges - remembered_edges)]
        removals = [("remove", left, right) for left, right in sorted(remembered_edges - true_edges)]
        return removals + additions

    def _apply_graph_updates(
        self,
        remembered_graph: dict[str, tuple[str, ...]],
        operations: list[tuple[str, str, str]],
    ) -> dict[str, tuple[str, ...]]:
        mutable = {room_id: set(neighbors) for room_id, neighbors in remembered_graph.items()}
        for op, left, right in operations:
            mutable.setdefault(left, set())
            mutable.setdefault(right, set())
            if op == "add":
                mutable[left].add(right)
                mutable[right].add(left)
            else:
                mutable[left].discard(right)
                mutable[right].discard(left)
        return {room_id: tuple(sorted(neighbors)) for room_id, neighbors in sorted(mutable.items())}


register_baseline(WeakHeuristicBaseline.name, WeakHeuristicBaseline)


class StaleMapPlannerBaseline(WeakHeuristicBaseline):
    """Diagnostic baseline that keeps planning on the pre-intervention map."""

    name = "stale_map_planner"
    category = "diagnostic_reference"
    supported_tiers = ("mapshift_2d",)

    def explore(self, environment: Any, context: BaselineContext) -> ExplorationResult:
        exploration = super().explore(environment, context)
        memory = dict(exploration.memory)
        memory["diagnostic_policy"] = "pre_intervention_map_only"
        return ExplorationResult(
            baseline_name=self.name,
            environment_id=exploration.environment_id,
            exploration_steps=exploration.exploration_steps,
            visited_cells=exploration.visited_cells,
            visited_node_ids=exploration.visited_node_ids,
            hidden_state=exploration.hidden_state,
            memory=memory,
        )

    def _evaluate_planning(
        self,
        environment: Any,
        task: PlanningTask,
        exploration: ExplorationResult,
    ) -> TaskEvaluationResult:
        if not isinstance(environment, Map2DEnvironment):
            return super()._evaluate_planning(environment, task, exploration)
        base_environment = self._base_map_from_exploration(exploration) or environment
        remembered_tokens = exploration.memory.get("remembered_goal_tokens", {})
        goal_node, goal_matches_task = goal_node_for_task(base_environment, task, remembered_goal_tokens=remembered_tokens)
        stale_path = None if goal_node is None else base_environment.oracle_shortest_path(task.start_node_id, goal_node)
        executed_path = tuple(stale_path or ())
        path_still_valid = bool(executed_path) and self._path_valid_in_environment(environment, executed_path)
        oracle_length = None if task.goal_node_id is None else environment.shortest_path_length(task.start_node_id, task.goal_node_id)
        observed_length = None
        if stale_path is not None and path_still_valid:
            observed_length = max(0, len(stale_path) - 1) * environment.occupancy_resolution_m * environment.geometry_scale
            observed_length *= dynamics_cost_multiplier(environment)
        success = (
            goal_node is not None
            and goal_matches_task
            and stale_path is not None
            and path_still_valid
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
            path_cells=executed_path,
            metadata={
                "goal_matches_task": goal_matches_task,
                "planner": "pre_intervention_shortest_path",
                "path_still_valid": path_still_valid,
                "shift_severity": family_shift_severity(environment, task.family),
                "stale_base_environment_id": base_environment.environment_id,
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
        elif task.expected_output_type == "intervention_family":
            predicted_answer = "none"
        else:
            predicted_answer = None
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
            metadata={"diagnostic_policy": "no_intervention_belief"},
        )

    def _evaluate_adaptation(
        self,
        environment: Any,
        task: AdaptationTask,
        exploration: ExplorationResult,
    ) -> TaskEvaluationResult:
        if not isinstance(environment, Map2DEnvironment):
            return super()._evaluate_adaptation(environment, task, exploration)
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
        final_score = planning_result.path_efficiency if planning_result.success else 0.0
        curve = (final_score, final_score, final_score)
        return TaskEvaluationResult(
            baseline_name=self.name,
            task_class="adaptation",
            task_type=task.task_type,
            family=task.family,
            success=planning_result.success,
            solvable=planning_result.solvable,
            observed_length=planning_result.observed_length,
            oracle_length=planning_result.oracle_length,
            path_efficiency=planning_result.path_efficiency,
            oracle_gap=planning_result.oracle_gap,
            adaptation_curve=curve,
            path_cells=planning_result.path_cells,
            metadata={
                "diagnostic_policy": "no_post_intervention_adaptation",
                "base_success": planning_result.success,
            },
        )

    def _base_map_from_exploration(self, exploration: ExplorationResult) -> Map2DEnvironment | None:
        payload = exploration.memory.get("base_environment_payload")
        if not isinstance(payload, dict):
            return None
        return Map2DEnvironment.from_dict(payload)

    def _path_valid_in_environment(self, environment: Map2DEnvironment, path: tuple[Cell, ...]) -> bool:
        if not path:
            return False
        if not all(environment.is_free(cell) for cell in path):
            return False
        return all(right in environment.neighbors(left) for left, right in zip(path, path[1:]))


register_baseline(StaleMapPlannerBaseline.name, StaleMapPlannerBaseline)
