"""Relational graph calibration baseline for MapShift-2D."""

from __future__ import annotations

import math
from itertools import combinations
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
    path_efficiency_from_lengths,
    stable_bucket_score,
)


def _edge_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))


class RelationalGraphBaseline(BaseBaselineModel):
    """Graph-structured baseline with explicit node-edge relational summaries."""

    name = "relational_graph_world_model"
    category = "relational"

    def __init__(self, run_config: Any) -> None:
        super().__init__(run_config)
        self.hidden_size = max(6, int(self.parameters.get("hidden_size", 10)))
        self.message_passing_steps = max(1, int(self.parameters.get("message_passing_steps", 2)))
        self.family_penalties = {
            "metric": float(self.parameters.get("metric_penalty", 0.1)),
            "topology": float(self.parameters.get("topology_penalty", 0.08)),
            "dynamics": float(self.parameters.get("dynamics_penalty", 0.2)),
            "semantic": float(self.parameters.get("semantic_penalty", 0.29)),
        }
        self.topology_revision_bonus = float(self.parameters.get("topology_revision_bonus", 0.18))
        self.bottleneck_awareness_bonus = float(self.parameters.get("bottleneck_awareness_bonus", 0.1))
        self.metric_stability_bonus = float(self.parameters.get("metric_stability_bonus", 0.08))
        self.semantic_mismatch_penalty = float(self.parameters.get("semantic_mismatch_penalty", 0.16))
        self.adaptation_gain = float(self.parameters.get("adaptation_gain", 0.22))
        feature_dim = 8
        self.parameter_count = (self.hidden_size * feature_dim) + (self.message_passing_steps * self.hidden_size * 2) + self.hidden_size
        self.trainable_parameter_count = self.parameter_count

    def explore(self, environment: Any, context: BaselineContext) -> ExplorationResult:
        visited_cells, visited_node_ids = deterministic_exploration_trace(environment, context.exploration_budget_steps, context.seed)
        memory = summarize_exploration_memory(environment, visited_cells, visited_node_ids)
        graph_summary = self._build_relational_memory(environment, visited_cells, visited_node_ids)
        memory.update(graph_summary)
        return ExplorationResult(
            baseline_name=self.name,
            environment_id=environment.environment_id,
            exploration_steps=len(visited_cells),
            visited_cells=visited_cells,
            visited_node_ids=visited_node_ids,
            hidden_state=tuple(graph_summary["graph_embedding"]),
            memory=memory,
        )

    def evaluate(self, environment: Any, task: Any, exploration: ExplorationResult, context: BaselineContext) -> TaskEvaluationResult:
        if isinstance(task, PlanningTask):
            return self._evaluate_planning(environment, task, exploration)
        if isinstance(task, InferenceTask):
            return self._evaluate_inference(environment, task, exploration)
        if isinstance(task, AdaptationTask):
            return self._evaluate_adaptation(environment, task, exploration)
        raise TypeError(f"Unsupported task type for relational baseline: {type(task).__name__}")

    def _evaluate_planning(
        self,
        environment: Any,
        task: PlanningTask,
        exploration: ExplorationResult,
    ) -> TaskEvaluationResult:
        remembered_tokens = exploration.memory.get("remembered_goal_tokens", {})
        goal_node, goal_matches_task = goal_node_for_task(environment, task, remembered_goal_tokens=remembered_tokens)
        path = None if goal_node is None else environment.oracle_shortest_path(task.start_node_id, goal_node)
        oracle_length = None if task.goal_node_id is None else environment.shortest_path_length(task.start_node_id, task.goal_node_id)
        planned_length = None if goal_node is None else environment.shortest_path_length(task.start_node_id, goal_node)
        severity = family_shift_severity(environment, task.family)
        confidence = self._planning_confidence(environment, task, exploration, severity, goal_matches_task)
        observed_length = None
        if planned_length is not None:
            stretch = 1.0 + max(0.0, 1.0 - confidence) * 0.5
            observed_length = planned_length * stretch * dynamics_cost_multiplier(environment)
        success = (
            goal_node is not None
            and goal_matches_task
            and path is not None
            and confidence >= 0.35
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
            path_cells=tuple(path or ()),
            metadata={
                "goal_matches_task": goal_matches_task,
                "relational_confidence": confidence,
                "graph_node_coverage": exploration.memory.get("graph_node_coverage", 0.0),
                "graph_edge_coverage": exploration.memory.get("graph_edge_coverage", 0.0),
                "articulation_count": len(exploration.memory.get("graph_articulation_nodes", ())),
                "shift_severity": severity,
            },
        )

    def _evaluate_inference(
        self,
        environment: Any,
        task: InferenceTask,
        exploration: ExplorationResult,
    ) -> TaskEvaluationResult:
        structural_shift = self._structural_shift_detected(environment, exploration)
        changed_family = self._changed_family_from_signatures(environment, exploration)
        remembered_tokens = exploration.memory.get("remembered_goal_tokens", {})
        severity = family_shift_severity(environment, task.family)
        confidence = self._inference_confidence(environment, task.family, exploration, severity, structural_shift)

        if task.task_type == "detect_topology_change":
            predicted_answer = bool(structural_shift and confidence >= 0.45)
        elif task.task_type == "counterfactual_reachability_query":
            token = ""
            for piece in task.query.split():
                if piece.startswith("target_"):
                    token = piece
                    break
            predicted_answer = remembered_tokens.get(token)
        else:
            predicted_answer = changed_family if confidence >= 0.4 else "metric"
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
            metadata={
                "relational_confidence": confidence,
                "structural_shift_detected": structural_shift,
                "changed_family_guess": changed_family,
            },
        )

    def _evaluate_adaptation(
        self,
        environment: Any,
        task: AdaptationTask,
        exploration: ExplorationResult,
    ) -> TaskEvaluationResult:
        proxy_task = PlanningTask(
            task_type=task.task_type,
            horizon_steps=task.evaluation_horizon_steps,
            family=task.family,
            start_node_id=task.start_node_id,
            goal_node_id=task.goal_node_id,
            goal_token=None,
            goal_descriptor=task.goal_node_id,
            metadata=dict(task.metadata),
        )
        planning_result = self._evaluate_planning(environment, proxy_task, exploration)
        severity = family_shift_severity(environment, task.family)
        normalized_budget = min(1.0, task.adaptation_budget_steps / max(1, task.evaluation_horizon_steps))
        base_score = 1.0 if planning_result.success else max(0.0, 0.3 + planning_result.path_efficiency * 0.36)
        gain = self.adaptation_gain
        if task.family == "topology":
            gain += self.topology_revision_bonus * 0.8
        elif task.family == "metric":
            gain += self.metric_stability_bonus
        elif task.family == "semantic":
            gain -= self.semantic_mismatch_penalty * 0.4
        elif task.family == "dynamics":
            gain -= 0.05
        gain -= 0.025 * severity
        curve = (
            round(base_score, 6),
            round(min(1.0, base_score + gain * normalized_budget), 6),
            round(min(1.0, base_score + gain), 6),
        )
        return TaskEvaluationResult(
            baseline_name=self.name,
            task_class="adaptation",
            task_type=task.task_type,
            family=task.family,
            success=bool(curve[-1] >= 0.71 and planning_result.solvable),
            solvable=planning_result.solvable,
            observed_length=planning_result.observed_length,
            oracle_length=planning_result.oracle_length,
            path_efficiency=planning_result.path_efficiency,
            oracle_gap=planning_result.oracle_gap,
            adaptation_curve=curve,
            path_cells=planning_result.path_cells,
            metadata={"adaptation_gain": gain, "shift_severity": severity},
        )

    def _build_relational_memory(
        self,
        environment: Any,
        visited_cells: tuple[tuple[int, int], ...],
        visited_node_ids: tuple[str, ...],
    ) -> dict[str, Any]:
        visited_set = set(visited_cells)
        observed_nodes = tuple(sorted(set(visited_node_ids) | {environment.start_node_id}))
        observed_edges = []
        for left, right in environment.edge_list():
            if left not in observed_nodes or right not in observed_nodes:
                continue
            corridor = environment.edge_corridors.get("|".join(sorted((left, right))), [])
            if any(cell in visited_set for cell in corridor):
                observed_edges.append(_edge_key(left, right))
        if not observed_edges and len(observed_nodes) > 1:
            for left, right in combinations(observed_nodes, 2):
                if right in environment.adjacency.get(left, []):
                    observed_edges.append(_edge_key(left, right))
        adjacency = {node_id: [] for node_id in observed_nodes}
        for left, right in observed_edges:
            adjacency[left].append(right)
            adjacency[right].append(left)
        for neighbors in adjacency.values():
            neighbors.sort()

        articulation_nodes = tuple(self._articulation_nodes(observed_nodes, adjacency))
        closeness = self._closeness_scores(observed_nodes, adjacency)
        node_states = {
            node_id: self._initial_node_state(environment, node_id, adjacency, articulation_nodes, closeness)
            for node_id in observed_nodes
        }
        for step in range(self.message_passing_steps):
            node_states = self._message_passing_step(node_states, adjacency, step)
        graph_embedding = self._graph_embedding(node_states)
        graph_embedding_norm = sum(abs(value) for value in graph_embedding) / max(1, len(graph_embedding))
        return {
            "graph_node_ids": observed_nodes,
            "graph_edge_pairs": tuple(sorted(observed_edges)),
            "graph_node_coverage": len(observed_nodes) / max(1, environment.node_count()),
            "graph_edge_coverage": len(observed_edges) / max(1, environment.edge_count()),
            "graph_degree_profile": tuple(sorted(len(adjacency[node_id]) for node_id in observed_nodes)),
            "graph_articulation_nodes": articulation_nodes,
            "graph_closeness_scores": {node_id: round(value, 6) for node_id, value in sorted(closeness.items())},
            "graph_embedding": tuple(round(value, 6) for value in graph_embedding),
            "graph_embedding_norm": round(graph_embedding_norm, 6),
            "base_edge_signature": tuple(environment.edge_list()),
            "base_node_signature": tuple(sorted(observed_nodes)),
        }

    def _initial_node_state(
        self,
        environment: Any,
        node_id: str,
        adjacency: dict[str, list[str]],
        articulation_nodes: tuple[str, ...],
        closeness: dict[str, float],
    ) -> tuple[float, ...]:
        node = environment.nodes[node_id]
        features = [
            len(adjacency[node_id]) / max(1.0, float(max(1, environment.node_count() - 1))),
            closeness.get(node_id, 0.0),
            node.row / max(1.0, float(max(1, environment.height_cells - 1))),
            node.col / max(1.0, float(max(1, environment.width_cells - 1))),
            1.0 if node_id in articulation_nodes else 0.0,
            1.0 if node_id in environment.landmark_by_node else 0.0,
            1.0 if node_id in environment.goal_tokens.values() else 0.0,
            1.0 if node_id in {environment.start_node_id, environment.goal_node_id} else 0.0,
        ]
        state = []
        for dim in range(self.hidden_size):
            total = 0.0
            for feature_index, feature in enumerate(features):
                weight = (stable_bucket_score(self.name, "init", dim, feature_index) - 0.5) * 1.2
                total += feature * weight
            total += (stable_bucket_score(self.name, "bias", node_id, dim) - 0.5) * 0.4
            state.append(math.tanh(total))
        return tuple(state)

    def _message_passing_step(
        self,
        node_states: dict[str, tuple[float, ...]],
        adjacency: dict[str, list[str]],
        step: int,
    ) -> dict[str, tuple[float, ...]]:
        updated: dict[str, tuple[float, ...]] = {}
        for node_id, state in node_states.items():
            neighbors = adjacency.get(node_id, [])
            if neighbors:
                aggregate = [
                    sum(node_states[neighbor][dim] for neighbor in neighbors) / len(neighbors)
                    for dim in range(self.hidden_size)
                ]
            else:
                aggregate = list(state)
            new_state = []
            for dim, previous in enumerate(state):
                relation_gain = 0.24 + (stable_bucket_score(self.name, "relation", step, dim) * 0.16)
                bias = (stable_bucket_score(self.name, "step_bias", node_id, step, dim) - 0.5) * 0.15
                value = math.tanh((0.58 * previous) + (relation_gain * aggregate[dim]) + bias)
                new_state.append(value)
            updated[node_id] = tuple(new_state)
        return updated

    def _graph_embedding(self, node_states: dict[str, tuple[float, ...]]) -> tuple[float, ...]:
        if not node_states:
            return tuple(0.0 for _ in range(self.hidden_size))
        ordered_states = [node_states[node_id] for node_id in sorted(node_states)]
        return tuple(
            sum(state[dim] for state in ordered_states) / len(ordered_states)
            for dim in range(self.hidden_size)
        )

    def _articulation_nodes(self, node_ids: tuple[str, ...], adjacency: dict[str, list[str]]) -> list[str]:
        if len(node_ids) <= 2:
            return []
        base_components = self._component_count(node_ids, adjacency)
        articulation = []
        for removed in node_ids:
            remaining = tuple(node_id for node_id in node_ids if node_id != removed)
            if not remaining:
                continue
            reduced = {
                node_id: [neighbor for neighbor in adjacency[node_id] if neighbor != removed]
                for node_id in remaining
            }
            if self._component_count(remaining, reduced) > base_components:
                articulation.append(removed)
        return articulation

    def _component_count(self, node_ids: tuple[str, ...], adjacency: dict[str, list[str]]) -> int:
        unseen = set(node_ids)
        components = 0
        while unseen:
            components += 1
            stack = [next(iter(unseen))]
            while stack:
                current = stack.pop()
                if current not in unseen:
                    continue
                unseen.remove(current)
                stack.extend(neighbor for neighbor in adjacency.get(current, []) if neighbor in unseen)
        return components

    def _closeness_scores(self, node_ids: tuple[str, ...], adjacency: dict[str, list[str]]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for node_id in node_ids:
            distances = self._graph_distances(node_id, adjacency)
            if len(distances) <= 1:
                scores[node_id] = 0.0
                continue
            total_distance = sum(distance for other, distance in distances.items() if other != node_id)
            scores[node_id] = 0.0 if total_distance <= 0 else (len(distances) - 1) / total_distance
        return scores

    def _graph_distances(self, source: str, adjacency: dict[str, list[str]]) -> dict[str, int]:
        frontier = [source]
        distances = {source: 0}
        while frontier:
            current = frontier.pop(0)
            for neighbor in adjacency.get(current, []):
                if neighbor in distances:
                    continue
                distances[neighbor] = distances[current] + 1
                frontier.append(neighbor)
        return distances

    def _planning_confidence(
        self,
        environment: Any,
        task: PlanningTask,
        exploration: ExplorationResult,
        severity: int,
        goal_matches_task: bool,
    ) -> float:
        node_coverage = float(exploration.memory.get("graph_node_coverage", 0.0))
        edge_coverage = float(exploration.memory.get("graph_edge_coverage", 0.0))
        embedding_norm = float(exploration.memory.get("graph_embedding_norm", 0.0))
        articulation_count = len(exploration.memory.get("graph_articulation_nodes", ()))
        articulation_ratio = articulation_count / max(1.0, float(len(exploration.memory.get("graph_node_ids", ()))))
        confidence = 0.46 + (node_coverage * 0.14) + (edge_coverage * 0.18) + (embedding_norm * 0.14)
        confidence += (stable_bucket_score(self.name, task.family, task.task_type, self.seed) - 0.5) * 0.08
        confidence -= self.family_penalties.get(task.family, 0.2)
        confidence -= 0.05 * severity
        if task.family == "topology" and bool(task.metadata.get("route_changed", False)):
            confidence += self.topology_revision_bonus + (self.bottleneck_awareness_bonus * articulation_ratio)
        if task.family == "metric":
            confidence += self.metric_stability_bonus
        if task.family == "semantic" and task.goal_token is not None:
            confidence -= self.semantic_mismatch_penalty
        if not goal_matches_task:
            confidence -= 0.24
        if task.family == "dynamics":
            confidence -= max(0.0, dynamics_cost_multiplier(environment) - 1.0) * 0.12
        return max(0.0, min(1.0, confidence))

    def _inference_confidence(
        self,
        environment: Any,
        family: str,
        exploration: ExplorationResult,
        severity: int,
        structural_shift: bool,
    ) -> float:
        node_coverage = float(exploration.memory.get("graph_node_coverage", 0.0))
        edge_coverage = float(exploration.memory.get("graph_edge_coverage", 0.0))
        embedding_norm = float(exploration.memory.get("graph_embedding_norm", 0.0))
        confidence = 0.42 + (node_coverage * 0.12) + (edge_coverage * 0.16) + (embedding_norm * 0.1)
        if family == "topology" and structural_shift:
            confidence += self.topology_revision_bonus * 0.75
        confidence -= self.family_penalties.get(family, 0.2) * 0.45
        confidence -= 0.03 * severity
        if family == "dynamics":
            confidence -= max(0.0, dynamics_cost_multiplier(environment) - 1.0) * 0.08
        return max(0.0, min(1.0, confidence))

    def _structural_shift_detected(self, environment: Any, exploration: ExplorationResult) -> bool:
        base_edges = tuple(tuple(item) for item in exploration.memory.get("base_edge_signature", ()))
        current_edges = tuple(environment.edge_list())
        return current_edges != base_edges

    def _changed_family_from_signatures(self, environment: Any, exploration: ExplorationResult) -> str:
        base_geometry = exploration.memory.get("base_geometry_signature")
        base_semantic = exploration.memory.get("base_semantic_signature")
        base_dynamics = exploration.memory.get("base_dynamics_signature")
        base_edges = tuple(tuple(item) for item in exploration.memory.get("base_edge_signature", ()))
        if tuple(environment.edge_list()) != base_edges:
            return "topology"
        if environment.semantic_signature() != base_semantic:
            return "semantic"
        if environment.dynamics_signature() != base_dynamics:
            return "dynamics"
        if environment.geometry_signature() != base_geometry:
            return "metric"
        return "metric"


register_baseline(RelationalGraphBaseline.name, RelationalGraphBaseline)
