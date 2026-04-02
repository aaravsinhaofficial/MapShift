"""Shared learned graph baseline scaffold for MapShift-2D."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from torch import nn

from mapshift.envs.map2d.state import Map2DEnvironment
from mapshift.tasks.adaptation import AdaptationTask
from mapshift.tasks.inference import InferenceTask
from mapshift.tasks.planning import PlanningTask

from .api import (
    BaseBaselineModel,
    BaselineContext,
    ExplorationResult,
    TaskEvaluationResult,
    deterministic_exploration_trace,
    summarize_exploration_memory,
)
from .common import dynamics_cost_multiplier, horizon_allows_path, path_efficiency_from_lengths
from .learned_common import checkpoint_path, count_parameters, load_checkpoint, save_checkpoint, set_torch_seed
from .learned_graph import (
    GraphTrainingData,
    build_graph_training_data,
    edge_probability_map,
    execute_belief_route,
    infer_goal_node_from_semantics,
    plan_on_predicted_graph,
    semantic_label_for_node,
    token_symbol,
    traversal_cost_map,
)


class LearnedGraphBaseline(BaseBaselineModel, ABC):
    """Common learned-baseline implementation driven by predicted node graphs."""

    learnable = True
    implementation_kind = "self_supervised_torch_world_model"

    def __init__(self, run_config: Any) -> None:
        super().__init__(run_config)
        self.training_epochs = max(1, int(self.parameters.get("training_epochs", 8)))
        self.learning_rate = float(self.parameters.get("learning_rate", 0.01))
        self.edge_threshold = float(self.parameters.get("edge_threshold", 0.5))
        self.checkpoint_dir = str(self.parameters.get("checkpoint_dir", "/tmp/mapshift_learned_baselines"))
        self.adaptation_step_divisor = max(1, int(self.parameters.get("adaptation_step_divisor", 16)))
        self.validation_fraction = min(0.4, max(0.1, float(self.parameters.get("validation_fraction", 0.2))))
        self.early_stopping_patience = max(1, int(self.parameters.get("early_stopping_patience", 3)))
        self.max_probe_steps = max(1, int(self.parameters.get("max_probe_steps", 8)))
        self._model_cache: dict[str, nn.Module] = {}
        self._training_cache: dict[str, dict[str, Any]] = {}

    @property
    @abstractmethod
    def model_class(self) -> type[nn.Module]:
        raise NotImplementedError

    @abstractmethod
    def build_model(self, graph_data: GraphTrainingData) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def forward_outputs(self, model: nn.Module, graph_data: GraphTrainingData) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def training_loss(
        self,
        outputs: dict[str, torch.Tensor],
        graph_data: GraphTrainingData,
        *,
        pair_mask: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        edge_terms = nn.functional.binary_cross_entropy_with_logits(
            outputs["edge_logits"],
            graph_data.edge_labels,
            reduction="none",
        )
        geometry_terms = nn.functional.mse_loss(
            outputs["geometry_costs"],
            graph_data.geometry_cost_labels,
            reduction="none",
        )
        traversal_terms = nn.functional.mse_loss(
            outputs["traversal_costs"],
            graph_data.traversal_cost_labels,
            reduction="none",
        )
        token_terms = nn.functional.binary_cross_entropy_with_logits(
            outputs["token_logits"],
            graph_data.token_labels,
            reduction="none",
        )

        def _reduce(vector: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
            if mask is None:
                return vector.mean()
            active = vector[mask]
            return active.mean() if active.numel() else vector.mean()

        def _reduce_nodes(matrix: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
            reduced = matrix.mean(dim=-1)
            if mask is None:
                return reduced.mean()
            active = reduced[mask]
            return active.mean() if active.numel() else reduced.mean()

        edge_loss = _reduce(edge_terms, pair_mask)
        geometry_loss = _reduce(geometry_terms, pair_mask)
        traversal_loss = _reduce(traversal_terms, pair_mask)
        token_loss = _reduce_nodes(token_terms, node_mask)
        total = edge_loss + geometry_loss + traversal_loss + token_loss
        return total, {
            "edge_loss": float(edge_loss.item()),
            "geometry_loss": float(geometry_loss.item()),
            "traversal_loss": float(traversal_loss.item()),
            "token_loss": float(token_loss.item()),
        }

    def explore(self, environment: Any, context: BaselineContext) -> ExplorationResult:
        visited_cells, visited_node_ids = deterministic_exploration_trace(environment, context.exploration_budget_steps, context.seed)
        graph_data = build_graph_training_data(environment, visited_node_ids)
        checkpoint = checkpoint_path(
            checkpoint_dir=self.checkpoint_dir,
            baseline_name=self.name,
            environment_id=environment.environment_id,
            seed=context.seed,
            parameters=self.parameters,
        )
        model, training_summary = self._train_or_load_model(environment.environment_id, graph_data, checkpoint, context.seed)
        outputs = self.forward_outputs(model, graph_data)
        hidden = tuple(float(value) for value in outputs["node_hidden"].mean(dim=0).detach().cpu().tolist())
        memory = summarize_exploration_memory(environment, visited_cells, visited_node_ids)
        memory["training_summary"] = dict(training_summary)
        memory["node_order"] = tuple(graph_data.node_order)
        memory["token_order"] = tuple(graph_data.token_order)
        memory["belief_graph_node_count"] = len(graph_data.node_order)
        memory["belief_graph_edge_count"] = len(environment.edge_list())
        return ExplorationResult(
            baseline_name=self.name,
            environment_id=environment.environment_id,
            exploration_steps=len(visited_cells),
            visited_cells=visited_cells,
            visited_node_ids=visited_node_ids,
            hidden_state=hidden,
            memory=memory,
        )

    def evaluate(self, environment: Any, task: Any, exploration: ExplorationResult, context: BaselineContext) -> TaskEvaluationResult:
        if isinstance(task, PlanningTask):
            return self._evaluate_planning(environment, task, exploration)
        if isinstance(task, InferenceTask):
            return self._evaluate_inference(environment, task, exploration)
        if isinstance(task, AdaptationTask):
            return self._evaluate_adaptation(environment, task, exploration)
        raise TypeError(f"Unsupported task type for {self.name}: {type(task).__name__}")

    def _train_or_load_model(
        self,
        environment_id: str,
        graph_data: GraphTrainingData,
        checkpoint: Path,
        seed: int,
    ) -> tuple[nn.Module, dict[str, Any]]:
        if environment_id in self._model_cache:
            return self._model_cache[environment_id], dict(self._training_cache[environment_id])

        model = self.build_model(graph_data)
        total_parameters, trainable_parameters = count_parameters(model)
        self.parameter_count = total_parameters
        self.trainable_parameter_count = trainable_parameters

        if checkpoint.is_file():
            payload = load_checkpoint(checkpoint)
            try:
                model.load_state_dict(payload["state_dict"])
            except RuntimeError:
                checkpoint.unlink(missing_ok=True)
            else:
                model.eval()
                summary = dict(payload.get("training_summary", {}))
                summary.setdefault("parameter_count", total_parameters)
                summary.setdefault("trainable_parameter_count", trainable_parameters)
                self._model_cache[environment_id] = model
                self._training_cache[environment_id] = summary
                return model, summary

        set_torch_seed(seed)
        pair_train_mask, pair_val_mask = self._pair_split_masks(graph_data)
        node_train_mask, node_val_mask = self._node_split_masks(graph_data)

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        train_curve: list[float] = []
        val_curve: list[float] = []
        best_epoch = 0
        best_val_loss = float("inf")
        best_state = copy.deepcopy(model.state_dict())
        patience = 0
        last_breakdown: dict[str, float] = {}

        for epoch in range(self.training_epochs):
            optimizer.zero_grad()
            outputs = self.forward_outputs(model, graph_data)
            train_loss, train_breakdown = self.training_loss(
                outputs,
                graph_data,
                pair_mask=pair_train_mask,
                node_mask=node_train_mask,
            )
            train_loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                validation_outputs = self.forward_outputs(model, graph_data)
                validation_loss, validation_breakdown = self.training_loss(
                    validation_outputs,
                    graph_data,
                    pair_mask=pair_val_mask,
                    node_mask=node_val_mask,
                )
            model.train()

            train_curve.append(float(train_loss.item()))
            val_curve.append(float(validation_loss.item()))
            last_breakdown = {
                **{f"train_{key}": round(value, 6) for key, value in sorted(train_breakdown.items())},
                **{f"val_{key}": round(value, 6) for key, value in sorted(validation_breakdown.items())},
            }
            if validation_loss.item() + 1e-8 < best_val_loss:
                best_val_loss = float(validation_loss.item())
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
                patience = 0
            else:
                patience += 1
                if patience >= self.early_stopping_patience:
                    break

        model.load_state_dict(best_state)
        model.eval()
        summary = {
            "training_epochs": len(train_curve),
            "configured_epochs": self.training_epochs,
            "learning_rate": self.learning_rate,
            "best_epoch": best_epoch,
            "best_validation_loss": round(best_val_loss if best_val_loss != float("inf") else 0.0, 6),
            "final_train_loss": round(train_curve[-1] if train_curve else 0.0, 6),
            "train_loss_curve": tuple(round(value, 6) for value in train_curve),
            "validation_loss_curve": tuple(round(value, 6) for value in val_curve),
            "loss_breakdown": dict(last_breakdown),
            "checkpoint_path": str(checkpoint),
            "parameter_count": total_parameters,
            "trainable_parameter_count": trainable_parameters,
        }
        save_checkpoint(checkpoint, {"state_dict": model.state_dict(), "training_summary": summary})
        self._model_cache[environment_id] = model
        self._training_cache[environment_id] = summary
        return model, summary

    def _pair_split_masks(self, graph_data: GraphTrainingData) -> tuple[torch.Tensor, torch.Tensor]:
        pair_count = int(graph_data.pair_index.shape[0])
        mask = torch.ones(pair_count, dtype=torch.bool)
        if pair_count >= 5:
            for index in range(pair_count):
                if index % max(2, int(round(1 / self.validation_fraction))) == 0:
                    mask[index] = False
        validation = ~mask
        if not validation.any():
            validation[-1] = True
            mask[-1] = False
        if not mask.any():
            mask[0] = True
        return mask, validation

    def _node_split_masks(self, graph_data: GraphTrainingData) -> tuple[torch.Tensor, torch.Tensor]:
        node_count = int(graph_data.node_features.shape[0])
        mask = torch.ones(node_count, dtype=torch.bool)
        if node_count >= 4:
            for index in range(node_count):
                if index % max(2, int(round(1 / self.validation_fraction))) == 0:
                    mask[index] = False
        validation = ~mask
        if not validation.any():
            validation[-1] = True
            mask[-1] = False
        if not mask.any():
            mask[0] = True
        return mask, validation

    def _base_environment(self, exploration: ExplorationResult) -> Map2DEnvironment:
        payload = exploration.memory.get("base_environment_payload")
        if not isinstance(payload, dict):
            raise ValueError(f"Exploration memory for {self.name} is missing base_environment_payload")
        return Map2DEnvironment.from_dict(payload)

    def _base_model_outputs(
        self,
        exploration: ExplorationResult,
    ) -> tuple[Map2DEnvironment, GraphTrainingData, dict[str, torch.Tensor]]:
        model = self._model_cache.get(exploration.environment_id)
        if model is None:
            raise ValueError(f"No cached model found for {exploration.environment_id}")
        base_environment = self._base_environment(exploration)
        graph_data = build_graph_training_data(base_environment, tuple(exploration.visited_node_ids))
        outputs = self.forward_outputs(model, graph_data)
        return base_environment, graph_data, outputs

    def _goal_node_from_token(self, outputs: dict[str, torch.Tensor], graph_data: GraphTrainingData, goal_token: str | None) -> str | None:
        if goal_token is None or goal_token not in graph_data.token_order:
            return None
        token_index = graph_data.token_order.index(goal_token)
        token_scores = torch.sigmoid(outputs["token_logits"][:, token_index]).detach().cpu().tolist()
        if not token_scores:
            return None
        best_index = max(range(len(token_scores)), key=lambda index: token_scores[index])
        return graph_data.node_order[best_index]

    def _ordered_edge_probes(self, base_environment: Map2DEnvironment, start_node_id: str) -> list[tuple[str, str]]:
        def _distance(edge: tuple[str, str]) -> tuple[float, tuple[str, str]]:
            left, right = edge
            left_distance = base_environment.shortest_path_length(start_node_id, left) or 1e9
            right_distance = base_environment.shortest_path_length(start_node_id, right) or 1e9
            return min(left_distance, right_distance), edge

        return sorted(base_environment.edge_list(), key=_distance)

    def _ordered_shortcut_probes(self, base_environment: Map2DEnvironment) -> list[tuple[str, str]]:
        return [(left, right) for left, right, _delta in base_environment.candidate_shortcuts()]

    def _ordered_semantic_probes(self, base_environment: Map2DEnvironment, start_node_id: str, candidate_node_ids: tuple[str, ...]) -> list[str]:
        def _distance(node_id: str) -> tuple[float, str]:
            return base_environment.shortest_path_length(start_node_id, node_id) or 1e9, node_id

        return sorted(candidate_node_ids, key=_distance)

    def _probe_structural_updates(
        self,
        *,
        base_environment: Map2DEnvironment,
        current_environment: Map2DEnvironment,
        start_node_id: str,
        probe_budget_steps: int,
    ) -> tuple[set[tuple[str, str]], dict[tuple[str, str], tuple[Any, ...]], dict[str, Any]]:
        active_budget = max(0, min(self.max_probe_steps, probe_budget_steps))
        blocked_edges: set[tuple[str, str]] = set()
        opened_corridors: dict[tuple[str, str], tuple[Any, ...]] = {}
        probed_known_edges: list[tuple[str, str]] = []
        probed_shortcuts: list[tuple[str, str]] = []

        if active_budget <= 0:
            return blocked_edges, opened_corridors, {
                "structural_shift_detected": False,
                "probed_known_edges": [],
                "probed_shortcuts": [],
            }

        edge_budget = max(1, active_budget)
        shortcut_budget = max(0, active_budget // 2)
        for edge in self._ordered_edge_probes(base_environment, start_node_id)[:edge_budget]:
            left, right = edge
            corridor = base_environment.corridor_for_edge(left, right)
            probed_known_edges.append(edge)
            if not corridor or not current_environment.corridor_is_traversable(corridor):
                blocked_edges.add(tuple(sorted(edge)))

        for edge in self._ordered_shortcut_probes(base_environment)[:shortcut_budget]:
            left, right = edge
            corridor = current_environment.corridor_for_edge(left, right)
            probed_shortcuts.append(edge)
            if corridor and current_environment.corridor_is_traversable(corridor):
                opened_corridors[tuple(sorted(edge))] = tuple(corridor)

        return blocked_edges, opened_corridors, {
            "structural_shift_detected": bool(blocked_edges or opened_corridors),
            "probed_known_edges": [list(edge) for edge in probed_known_edges],
            "probed_shortcuts": [list(edge) for edge in probed_shortcuts],
        }

    def _probe_semantic_goal(
        self,
        *,
        current_environment: Map2DEnvironment,
        candidate_node_ids: tuple[str, ...],
        start_node_id: str,
        goal_token: str | None,
        probe_budget_steps: int,
    ) -> tuple[str | None, dict[str, Any]]:
        if not goal_token:
            return None, {"semantic_probe_used": False, "semantic_probe_candidates": []}
        budget = max(1, min(len(candidate_node_ids), max(2, probe_budget_steps)))
        probe_nodes = tuple(self._ordered_semantic_probes(current_environment, start_node_id, candidate_node_ids)[:budget])
        return infer_goal_node_from_semantics(current_environment, probe_nodes, goal_token)

    def _predicted_graph(
        self,
        graph_data: GraphTrainingData,
        outputs: dict[str, torch.Tensor],
    ) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
        edge_probabilities = edge_probability_map(graph_data.node_order, graph_data.pair_index, outputs["edge_logits"])
        traversal_costs = traversal_cost_map(
            graph_data.node_order,
            graph_data.pair_index,
            outputs["traversal_costs"],
            graph_data.max_traversal_cost,
        )
        return edge_probabilities, traversal_costs

    def _evaluate_planning(
        self,
        environment: Any,
        task: PlanningTask,
        exploration: ExplorationResult,
        *,
        probe_budget_steps: int = 0,
    ) -> TaskEvaluationResult:
        try:
            base_environment, graph_data, outputs = self._base_model_outputs(exploration)
        except ValueError:
            return self._empty_planning_result(task)

        semantic_goal_node, semantic_metadata = self._probe_semantic_goal(
            current_environment=environment,
            candidate_node_ids=graph_data.node_order,
            start_node_id=task.start_node_id,
            goal_token=task.goal_token,
            probe_budget_steps=probe_budget_steps,
        )
        model_goal_node = self._goal_node_from_token(outputs, graph_data, task.goal_token) if task.goal_token else task.goal_node_id
        goal_node = semantic_goal_node or model_goal_node
        goal_matches_task = goal_node == task.goal_node_id if task.goal_node_id is not None else goal_node is not None
        if goal_node is None:
            return self._empty_planning_result(task, goal_matches_task=goal_matches_task)

        blocked_edges, opened_corridors, structural_metadata = self._probe_structural_updates(
            base_environment=base_environment,
            current_environment=environment,
            start_node_id=task.start_node_id,
            probe_budget_steps=probe_budget_steps,
        )
        edge_probabilities, traversal_costs = self._predicted_graph(graph_data, outputs)
        initial_route = plan_on_predicted_graph(
            node_order=graph_data.node_order,
            start_node_id=task.start_node_id,
            goal_node_id=goal_node,
            edge_probabilities={
                **edge_probabilities,
                **{edge: 1.0 for edge in opened_corridors},
                **{edge: 0.0 for edge in blocked_edges},
            },
            traversal_costs=traversal_costs,
            edge_threshold=self.edge_threshold,
        )
        belief_result = execute_belief_route(
            base_environment=base_environment,
            current_environment=environment,
            node_order=graph_data.node_order,
            start_node_id=task.start_node_id,
            goal_node_id=goal_node,
            edge_probabilities=edge_probabilities,
            traversal_costs=traversal_costs,
            edge_threshold=self.edge_threshold,
            blocked_edges=blocked_edges,
            opened_corridors=opened_corridors,
            max_replans=max(4, probe_budget_steps + 2),
        )
        oracle_length = None if task.goal_node_id is None else environment.shortest_path_length(task.start_node_id, task.goal_node_id)
        success = belief_result.reached_goal and goal_matches_task and horizon_allows_path(task, belief_result.observed_length, environment)
        return TaskEvaluationResult(
            baseline_name=self.name,
            task_class="planning",
            task_type=task.task_type,
            family=task.family,
            success=success,
            solvable=oracle_length is not None,
            observed_length=belief_result.observed_length,
            oracle_length=oracle_length,
            path_efficiency=path_efficiency_from_lengths(oracle_length, belief_result.observed_length),
            oracle_gap=None
            if oracle_length is None or belief_result.observed_length is None
            else max(0.0, belief_result.observed_length - oracle_length),
            path_cells=belief_result.path_cells,
            metadata={
                "goal_matches_task": goal_matches_task,
                "predicted_goal_node": goal_node,
                "model_goal_node": model_goal_node,
                "semantic_goal_node": semantic_goal_node,
                "predicted_node_route": list(initial_route or []),
                "executed_route": belief_result.success,
                "blocked_edges": [list(edge) for edge in belief_result.blocked_edges],
                "discovered_edges": [list(edge) for edge in belief_result.discovered_edges],
                "replans": belief_result.replans,
                "probe_budget_steps": probe_budget_steps,
                **semantic_metadata,
                **structural_metadata,
                **belief_result.metadata,
            },
        )

    def _empty_planning_result(self, task: PlanningTask, goal_matches_task: bool = False) -> TaskEvaluationResult:
        return TaskEvaluationResult(
            baseline_name=self.name,
            task_class="planning",
            task_type=task.task_type,
            family=task.family,
            success=False,
            solvable=bool(task.goal_node_id),
            observed_length=None,
            oracle_length=None,
            path_efficiency=0.0,
            oracle_gap=None,
            path_cells=(),
            metadata={"goal_matches_task": goal_matches_task, "predicted_node_route": [], "executed_route": False},
        )

    def _evaluate_inference(self, environment: Any, task: InferenceTask, exploration: ExplorationResult) -> TaskEvaluationResult:
        try:
            base_environment, graph_data, outputs = self._base_model_outputs(exploration)
        except ValueError:
            return TaskEvaluationResult(
                baseline_name=self.name,
                task_class="inference",
                task_type=task.task_type,
                family=task.family,
                success=False,
                solvable=True,
                predicted_answer=None,
                correct=False,
                metadata={},
            )

        blocked_edges, opened_corridors, structural_metadata = self._probe_structural_updates(
            base_environment=base_environment,
            current_environment=environment,
            start_node_id=base_environment.start_node_id,
            probe_budget_steps=self.max_probe_steps,
        )
        semantic_probe_candidates = tuple(graph_data.node_order)
        semantic_changed_nodes = [
            node_id
            for node_id in semantic_probe_candidates
            if semantic_label_for_node(base_environment, node_id) != semantic_label_for_node(environment, node_id)
        ]
        semantic_ratio = len(semantic_changed_nodes) / max(1, len(semantic_probe_candidates))
        topology_ratio = (len(blocked_edges) + len(opened_corridors)) / max(
            1,
            len(structural_metadata.get("probed_known_edges", [])) + len(structural_metadata.get("probed_shortcuts", [])),
        )
        metric_samples: list[float] = []
        traversal_samples: list[float] = []
        for edge in structural_metadata.get("probed_known_edges", []):
            left, right = edge
            corridor = base_environment.corridor_for_edge(str(left), str(right))
            if not corridor or not environment.corridor_is_traversable(corridor):
                continue
            base_geometry = max(1e-6, len(corridor) - 1) * base_environment.occupancy_resolution_m * base_environment.geometry_scale
            current_geometry = max(1e-6, len(corridor) - 1) * environment.occupancy_resolution_m * environment.geometry_scale
            metric_samples.append(abs(current_geometry - base_geometry) / base_geometry)
            base_traversal = base_geometry * dynamics_cost_multiplier(base_environment)
            current_traversal = current_geometry * dynamics_cost_multiplier(environment)
            traversal_samples.append(abs(current_traversal - base_traversal) / max(1e-6, base_traversal))
        metric_ratio = sum(metric_samples) / len(metric_samples) if metric_samples else 0.0
        traversal_ratio = sum(traversal_samples) / len(traversal_samples) if traversal_samples else 0.0
        dynamics_ratio = max(0.0, traversal_ratio - metric_ratio)

        if task.task_type == "detect_topology_change":
            predicted_answer = bool(structural_metadata["structural_shift_detected"])
        elif task.task_type == "counterfactual_reachability_query":
            token = next((piece for piece in task.query.split() if piece.startswith("target_")), "")
            predicted_answer, semantic_probe_metadata = infer_goal_node_from_semantics(environment, semantic_probe_candidates, token)
            structural_metadata = {**structural_metadata, **semantic_probe_metadata}
        else:
            family_scores = {
                "topology": topology_ratio,
                "semantic": semantic_ratio,
                "metric": metric_ratio,
                "dynamics": dynamics_ratio,
            }
            predicted_answer = max(sorted(family_scores), key=lambda family: family_scores[family])
            structural_metadata["family_scores"] = family_scores

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
                "topology_ratio": topology_ratio,
                "semantic_ratio": semantic_ratio,
                "metric_ratio": metric_ratio,
                "dynamics_ratio": dynamics_ratio,
                "structural_shift_detected": bool(structural_metadata.get("structural_shift_detected")),
                "semantic_changed_nodes": sorted(semantic_changed_nodes),
                **structural_metadata,
            },
        )

    def _evaluate_adaptation(self, environment: Any, task: AdaptationTask, exploration: ExplorationResult) -> TaskEvaluationResult:
        step_count = max(1, task.adaptation_budget_steps // self.adaptation_step_divisor)
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

        before = self._evaluate_planning(environment, proxy_task, exploration, probe_budget_steps=0)
        midpoint = self._evaluate_planning(environment, proxy_task, exploration, probe_budget_steps=max(1, step_count // 2))
        final = self._evaluate_planning(environment, proxy_task, exploration, probe_budget_steps=step_count)
        curve = (
            round(before.primary_score, 6),
            round(midpoint.primary_score, 6),
            round(final.primary_score, 6),
        )
        return TaskEvaluationResult(
            baseline_name=self.name,
            task_class="adaptation",
            task_type=task.task_type,
            family=task.family,
            success=bool(final.success),
            solvable=bool(final.solvable),
            observed_length=final.observed_length,
            oracle_length=final.oracle_length,
            path_efficiency=final.path_efficiency,
            oracle_gap=final.oracle_gap,
            adaptation_curve=curve,
            path_cells=final.path_cells,
            metadata={
                "adaptation_steps": step_count,
                "midpoint_probe_steps": max(1, step_count // 2),
                "final_probe_steps": step_count,
            },
        )
