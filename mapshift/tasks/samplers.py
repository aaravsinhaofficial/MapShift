"""Config-driven task sampling for the first executable benchmark path."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from typing import Any

from mapshift.core.manifests import TaskManifest
from mapshift.core.schemas import TaskConfig
from mapshift.envs.map2d.state import Map2DEnvironment
from mapshift.envs.procthor.wrappers import ProcTHORScene
from mapshift.splits.motifs import stable_template_hash, task_template_metadata

from .adaptation import AdaptationTask
from .inference import InferenceTask
from .planning import PlanningTask


@dataclass(frozen=True)
class TaskSamplingResult:
    """Container returned by the task sampler."""

    manifest: TaskManifest
    task: PlanningTask | InferenceTask | AdaptationTask


@dataclass(frozen=True)
class TaskRejection:
    """Structured record of a rejected sampled task."""

    family: str
    task_class: str
    task_type: str
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


class TaskSamplingRejected(ValueError):
    """Raised when a sampled task fails eligibility checks."""

    def __init__(self, rejection: TaskRejection) -> None:
        super().__init__(f"Rejected {rejection.task_class}/{rejection.task_type} for {rejection.family}: {rejection.reason}")
        self.rejection = rejection


def _config_hash(config: TaskConfig) -> str:
    config_blob = json.dumps(asdict(config), sort_keys=True).encode("utf-8")
    return hashlib.sha1(config_blob).hexdigest()[:12]


class TaskSampler:
    """Task sampler conditioned on base and intervened environments."""

    def __init__(self, config: TaskConfig | Any) -> None:
        self.config = config
        self.rejection_log: list[TaskRejection] = []

    def sample(
        self,
        base_environment: Map2DEnvironment | ProcTHORScene,
        intervened_environment: Map2DEnvironment | ProcTHORScene,
        family: str,
        seed: int,
        task_class: str | None = None,
    ) -> TaskSamplingResult:
        rng = random.Random(seed)
        candidate_classes = [task_class] if task_class is not None else self._ordered_task_classes(rng)
        last_rejection: TaskSamplingRejected | None = None

        for selected_class in candidate_classes:
            try:
                task = self._sample_for_class(base_environment, intervened_environment, family, rng, selected_class)
            except TaskSamplingRejected as exc:
                self.rejection_log.append(exc.rejection)
                last_rejection = exc
                continue

            manifest = TaskManifest(
                artifact_id=f"task-{intervened_environment.environment_id}-{selected_class}-{seed}",
                artifact_type="task",
                benchmark_version="0.1-draft",
                code_version="occupancy-grid-v1",
                config_hash=_config_hash(self.config),
                parent_ids=[base_environment.environment_id, intervened_environment.environment_id],
                seed_values=[seed],
                metadata={
                    "family": family,
                    "task_template_id": task.metadata.get("task_template_id"),
                    "query_template_id": task.metadata.get("query_template_id"),
                    "budget_template_id": task.metadata.get("budget_template_id"),
                    "start_goal_template_id": task.metadata.get("start_goal_template_id"),
                    "start_role_template_id": task.metadata.get("start_role_template_id"),
                    "goal_role_template_id": task.metadata.get("goal_role_template_id"),
                    "distance_steps": task.metadata.get("distance_steps"),
                    "path_changed": task.metadata.get("path_changed"),
                    "route_changed": task.metadata.get("route_changed"),
                    "distance_delta": task.metadata.get("distance_delta"),
                },
                task_id=f"{selected_class}-{intervened_environment.environment_id}-{seed}",
                task_class=selected_class,
                task_type=task.task_type,
                base_environment_id=base_environment.environment_id,
                intervened_environment_id=intervened_environment.environment_id,
                horizon_steps=self._task_horizon(task),
            )
            return TaskSamplingResult(manifest=manifest, task=task)

        if last_rejection is None:
            raise ValueError("No enabled task classes are available in the task config.")
        raise last_rejection

    def rejection_summary(self) -> dict[str, int]:
        """Return counts of task rejections by reason."""

        counts: dict[str, int] = {}
        for rejection in self.rejection_log:
            counts[rejection.reason] = counts.get(rejection.reason, 0) + 1
        return dict(sorted(counts.items()))

    def _ordered_task_classes(self, rng: random.Random) -> list[str]:
        enabled = [name for name, cfg in self.config.classes.items() if cfg.enabled]
        if not enabled:
            return []
        rng.shuffle(enabled)
        return enabled

    def _sample_for_class(
        self,
        base_environment: Map2DEnvironment | ProcTHORScene,
        intervened_environment: Map2DEnvironment | ProcTHORScene,
        family: str,
        rng: random.Random,
        selected_class: str,
    ) -> PlanningTask | InferenceTask | AdaptationTask:
        if selected_class == "planning":
            return self._sample_planning(base_environment, intervened_environment, family, rng)
        if selected_class == "inference":
            return self._sample_inference(base_environment, intervened_environment, family, rng)
        if selected_class == "adaptation":
            return self._sample_adaptation(base_environment, intervened_environment, family, rng)
        raise ValueError(f"Unsupported task class: {selected_class}")

    def _sample_planning(
        self,
        base_environment: Map2DEnvironment | ProcTHORScene,
        intervened_environment: Map2DEnvironment | ProcTHORScene,
        family: str,
        rng: random.Random,
    ) -> PlanningTask:
        cfg = self.config.classes["planning"]
        horizon = rng.choice(cfg.canonical_horizon_steps)
        start_node = intervened_environment.start_node_id
        goal_node = intervened_environment.goal_node_id
        comparison_goal_node = goal_node
        goal_token: str | None = None

        topology_changed = base_environment.edge_list() != intervened_environment.edge_list()
        dynamics_changed = base_environment.dynamics_signature() != intervened_environment.dynamics_signature()
        semantic_changed = base_environment.semantic_signature() != intervened_environment.semantic_signature()
        changed_token = self._changed_goal_token(base_environment, intervened_environment)
        metric_changed = self._metric_changed(base_environment, intervened_environment)
        route_changed = False

        if family == "topology" and topology_changed:
            topology_pair = self._select_topology_planning_pair(base_environment, intervened_environment)
            if topology_pair is None:
                self._reject(
                    family=family,
                    task_class="planning",
                    task_type="reroute_after_blockage",
                    reason="topology_no_reroute_required",
                    details={
                        "environment_id": intervened_environment.environment_id,
                        "applied_operations": list(intervened_environment.metadata.get("topology_shift", {}).get("applied_operations", [])),
                    },
                )
            start_node = str(topology_pair["start_node_id"])
            goal_node = str(topology_pair["goal_node_id"])
            comparison_goal_node = goal_node
            route_changed = bool(topology_pair["route_changed"])
            task_type = self._topology_planning_task_type(cfg, intervened_environment, topology_pair)
        elif family == "dynamics" and dynamics_changed and "reach_target_changed_dynamics" in cfg.task_types:
            task_type = "reach_target_changed_dynamics"
        elif family == "semantic" and semantic_changed and changed_token and "navigate_changed_cue_semantics" in cfg.task_types:
            task_type = "navigate_changed_cue_semantics"
            goal_token = changed_token
            goal_node = self._goal_target_for_token(intervened_environment, goal_token)
            comparison_goal_node = self._goal_target_for_token(base_environment, goal_token, fallback=goal_node)
            if goal_node == start_node:
                alternative_tokens = [
                    token
                    for token in sorted(intervened_environment.goal_tokens)
                    if self._goal_target_for_token(intervened_environment, token) != start_node
                ]
                if alternative_tokens:
                    goal_token = alternative_tokens[0]
                    goal_node = self._goal_target_for_token(intervened_environment, goal_token)
                    comparison_goal_node = self._goal_target_for_token(base_environment, goal_token, fallback=goal_node)
                else:
                    self._reject(
                        family=family,
                        task_class="planning",
                        task_type=task_type,
                        reason="semantic_goal_collapses_to_start",
                        details={"environment_id": intervened_environment.environment_id},
                    )
        else:
            task_type = "shortest_path_to_target"

        before_path = base_environment.shortest_path(start_node, comparison_goal_node)
        after_path = intervened_environment.shortest_path(start_node, goal_node)
        before = base_environment.shortest_path_length(start_node, comparison_goal_node)
        after = intervened_environment.shortest_path_length(start_node, goal_node)
        if family != "topology":
            route_changed = tuple(before_path or ()) != tuple(after_path or ())
        self._ensure_planning_eligible(
            family=family,
            task_type=task_type,
            before_distance=before,
            after_distance=after,
            topology_changed=topology_changed,
            semantic_changed=semantic_changed,
            dynamics_changed=dynamics_changed,
            metric_changed=metric_changed,
            route_changed=route_changed,
        )
        template_metadata = self._task_template_metadata(
            environment=intervened_environment,
            task_class="planning",
            task_type=task_type,
            family=family,
            start_node_id=start_node,
            goal_node_id=goal_node,
            goal_token=goal_token,
            horizon_steps=horizon,
        )

        return PlanningTask(
            task_type=task_type,
            horizon_steps=horizon,
            family=family,
            start_node_id=start_node,
            goal_node_id=goal_node,
            goal_token=goal_token,
            goal_descriptor=f"reach {goal_token or goal_node}",
            metadata={
                "base_path_length": before,
                "intervened_path_length": after,
                "path_changed": before != after,
                "route_changed": route_changed,
                "comparison_goal_node_id": comparison_goal_node,
                "distance_delta": None if before is None or after is None else after - before,
                **template_metadata,
            },
        )

    def _sample_inference(
        self,
        base_environment: Map2DEnvironment | ProcTHORScene,
        intervened_environment: Map2DEnvironment | ProcTHORScene,
        family: str,
        rng: random.Random,
    ) -> InferenceTask:
        cfg = self.config.classes["inference"]
        horizon_steps = int(rng.choice(cfg.canonical_horizon_steps))
        topology_changed = base_environment.edge_list() != intervened_environment.edge_list()
        semantic_changed = base_environment.semantic_signature() != intervened_environment.semantic_signature()
        changed_token = self._changed_goal_token(base_environment, intervened_environment)

        if family == "topology" and topology_changed and "detect_topology_change" in cfg.task_types:
            task_type = "detect_topology_change"
            query = "Did the connectivity structure change between the explored and intervened environment?"
            expected_answer = True
            expected_output_type = "boolean"
        elif family == "semantic" and semantic_changed and changed_token and "counterfactual_reachability_query" in cfg.task_types:
            task_type = "counterfactual_reachability_query"
            if isinstance(intervened_environment, ProcTHORScene):
                query = f"Which room does token {changed_token} refer to after the semantic intervention?"
                expected_answer = self._goal_target_for_token(intervened_environment, changed_token)
                expected_output_type = "room_id"
            else:
                query = f"Which node does token {changed_token} refer to after the semantic intervention?"
                expected_answer = intervened_environment.goal_tokens[changed_token]
                expected_output_type = "node_id"
        else:
            task_type = "predict_masked_region_after_intervention"
            query = f"What measurable parameter changed under the {family} intervention?"
            expected_answer = family
            expected_output_type = "category"

        self._ensure_inference_eligible(
            family=family,
            task_type=task_type,
            topology_changed=topology_changed,
            semantic_changed=semantic_changed,
            changed_token=changed_token,
        )
        template_metadata = self._task_template_metadata(
            environment=intervened_environment,
            task_class="inference",
            task_type=task_type,
            family=family,
            start_node_id=None,
            goal_node_id=None,
            expected_output_type=expected_output_type,
            horizon_steps=horizon_steps,
        )

        return InferenceTask(
            task_type=task_type,
            family=family,
            query=query,
            expected_output_type=expected_output_type,
            expected_answer=expected_answer,
            metadata={"horizon_steps": horizon_steps, **template_metadata},
        )

    def _sample_adaptation(
        self,
        base_environment: Map2DEnvironment | ProcTHORScene,
        intervened_environment: Map2DEnvironment | ProcTHORScene,
        family: str,
        rng: random.Random,
    ) -> AdaptationTask:
        cfg = self.config.classes["adaptation"]
        budget = rng.choice(cfg.canonical_horizon_steps)
        dynamics_changed = base_environment.dynamics_signature() != intervened_environment.dynamics_signature()
        metric_changed = self._metric_changed(base_environment, intervened_environment)
        topology_changed = base_environment.edge_list() != intervened_environment.edge_list()

        if family in {"metric", "dynamics"} and (metric_changed or dynamics_changed) and "few_shot_replanning_after_action_gain_change" in cfg.task_types:
            task_type = "few_shot_replanning_after_action_gain_change"
        else:
            task_type = "limited_post_shift_interaction_then_replan"

        goal_node = intervened_environment.goal_node_id
        distance = intervened_environment.shortest_path_length(intervened_environment.start_node_id, goal_node)
        self._ensure_adaptation_eligible(
            family=family,
            task_type=task_type,
            distance=distance,
            topology_changed=topology_changed,
            metric_changed=metric_changed,
            dynamics_changed=dynamics_changed,
        )
        template_metadata = self._task_template_metadata(
            environment=intervened_environment,
            task_class="adaptation",
            task_type=task_type,
            family=family,
            start_node_id=intervened_environment.start_node_id,
            goal_node_id=goal_node,
            adaptation_budget_steps=budget,
            horizon_steps=budget,
        )

        return AdaptationTask(
            task_type=task_type,
            family=family,
            adaptation_budget_steps=budget,
            evaluation_horizon_steps=budget,
            start_node_id=intervened_environment.start_node_id,
            goal_node_id=goal_node,
            metadata={"base_environment_id": base_environment.environment_id, **template_metadata},
        )

    def _changed_goal_token(
        self,
        base_environment: Map2DEnvironment | ProcTHORScene,
        intervened_environment: Map2DEnvironment | ProcTHORScene,
    ) -> str | None:
        for token, _base_target in base_environment.goal_tokens.items():
            if self._goal_target_for_token(intervened_environment, token, fallback="") != self._goal_target_for_token(base_environment, token, fallback=""):
                return token
        return None

    def _metric_changed(
        self,
        base_environment: Map2DEnvironment | ProcTHORScene,
        intervened_environment: Map2DEnvironment | ProcTHORScene,
    ) -> bool:
        if isinstance(base_environment, ProcTHORScene) and isinstance(intervened_environment, ProcTHORScene):
            return base_environment.metric_signature() != intervened_environment.metric_signature()
        return (
            abs(base_environment.geometry_scale - intervened_environment.geometry_scale) > 1e-9
            or abs(base_environment.observation_radius_m - intervened_environment.observation_radius_m) > 1e-9
        )

    def _select_topology_planning_pair(
        self,
        base_environment: Map2DEnvironment | ProcTHORScene,
        intervened_environment: Map2DEnvironment | ProcTHORScene,
    ) -> dict[str, Any] | None:
        candidates = self._topology_planning_candidates(base_environment, intervened_environment)
        if not candidates:
            return None

        topology_metadata = intervened_environment.metadata.get("topology_shift", {})
        applied_operations = [str(item) for item in topology_metadata.get("applied_operations", [])]
        prefer_degraded_paths = any(
            operation.startswith("remove_edge:") or operation.startswith("remove_aux_edge:")
            for operation in applied_operations
        )
        if prefer_degraded_paths:
            degraded = [
                candidate
                for candidate in candidates
                if candidate["distance_delta"] is not None and float(candidate["distance_delta"]) > 1e-9
            ]
            if degraded:
                candidates = degraded
        candidates.sort(
            key=lambda item: (
                -int(bool(item["connectivity_gain"])),
                -float(item["absolute_distance_delta"]),
                -int(item["path_cell_delta"]),
                str(item["start_node_id"]),
                str(item["goal_node_id"]),
            )
        )
        return candidates[0]

    def _topology_planning_candidates(
        self,
        base_environment: Map2DEnvironment | ProcTHORScene,
        intervened_environment: Map2DEnvironment | ProcTHORScene,
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        node_ids = self._node_ids(intervened_environment)
        for start_index, start_node in enumerate(node_ids):
            for goal_node in node_ids[start_index + 1 :]:
                before_path = base_environment.shortest_path(start_node, goal_node)
                after_path = intervened_environment.shortest_path(start_node, goal_node)
                if after_path is None:
                    continue
                route_changed = tuple(before_path or ()) != tuple(after_path or ())
                if not route_changed:
                    continue
                before_distance = base_environment.shortest_path_length(start_node, goal_node)
                after_distance = intervened_environment.shortest_path_length(start_node, goal_node)
                if after_distance is None or after_distance <= 0.0:
                    continue
                distance_delta = None
                absolute_distance_delta = 0.0
                if before_distance is not None:
                    distance_delta = after_distance - before_distance
                    absolute_distance_delta = abs(distance_delta)
                elif after_distance is not None:
                    absolute_distance_delta = float(after_distance)
                path_cell_delta = len(set(before_path or ()) ^ set(after_path or ()))
                candidates.append(
                    {
                        "start_node_id": start_node,
                        "goal_node_id": goal_node,
                        "before_distance": before_distance,
                        "after_distance": after_distance,
                        "distance_delta": distance_delta,
                        "absolute_distance_delta": absolute_distance_delta,
                        "route_changed": route_changed,
                        "path_cell_delta": path_cell_delta,
                        "connectivity_gain": before_path is None and after_path is not None,
                    }
                )
        return candidates

    def _topology_planning_task_type(
        self,
        planning_config: Any,
        intervened_environment: Map2DEnvironment | ProcTHORScene,
        topology_pair: dict[str, Any],
    ) -> str:
        applied_operations = [str(item) for item in intervened_environment.metadata.get("topology_shift", {}).get("applied_operations", [])]
        has_removal = any(
            operation.startswith("remove_edge:") or operation.startswith("remove_aux_edge:")
            for operation in applied_operations
        )
        if has_removal and topology_pair["route_changed"] and "reroute_after_blockage" in planning_config.task_types:
            return "reroute_after_blockage"
        return "shortest_path_to_target"

    def _ensure_planning_eligible(
        self,
        family: str,
        task_type: str,
        before_distance: float | None,
        after_distance: float | None,
        topology_changed: bool,
        semantic_changed: bool,
        dynamics_changed: bool,
        metric_changed: bool,
        route_changed: bool,
    ) -> None:
        if after_distance is None:
            self._reject(family, "planning", task_type, "impossible_path", {"after_distance": after_distance})

        threshold = max(1e-9, after_distance if after_distance is not None else 0.0)
        trivial_threshold = max(1e-9, threshold * 0.0)
        if after_distance is not None and after_distance <= trivial_threshold:
            self._reject(family, "planning", task_type, "trivial_goal_already_solved", {"after_distance": after_distance})

        if task_type == "reroute_after_blockage" and not route_changed:
            self._reject(
                family,
                "planning",
                task_type,
                "topology_no_reroute_required",
                {"before_distance": before_distance, "after_distance": after_distance},
            )
        if task_type == "navigate_changed_cue_semantics" and not semantic_changed:
            self._reject(family, "planning", task_type, "semantic_no_counterfactual_change", {})
        if task_type == "reach_target_changed_dynamics" and not dynamics_changed:
            self._reject(family, "planning", task_type, "dynamics_no_transition_change", {})
        if family == "metric" and task_type != "shortest_path_to_target" and not metric_changed:
            self._reject(family, "planning", task_type, "metric_no_geometry_change", {})
        if task_type == "shortest_path_to_target" and family == "topology" and topology_changed and not route_changed:
            self._reject(family, "planning", task_type, "uninformative_no_material_change", {"before_distance": before_distance, "after_distance": after_distance})

    def _ensure_inference_eligible(
        self,
        family: str,
        task_type: str,
        topology_changed: bool,
        semantic_changed: bool,
        changed_token: str | None,
    ) -> None:
        if task_type == "detect_topology_change" and not topology_changed:
            self._reject(family, "inference", task_type, "topology_no_connectivity_change", {})
        if task_type == "counterfactual_reachability_query" and (not semantic_changed or changed_token is None):
            self._reject(family, "inference", task_type, "semantic_no_counterfactual_change", {})

    def _ensure_adaptation_eligible(
        self,
        family: str,
        task_type: str,
        distance: float | None,
        topology_changed: bool,
        metric_changed: bool,
        dynamics_changed: bool,
    ) -> None:
        if distance is None:
            self._reject(family, "adaptation", task_type, "impossible_path", {})
        if distance is not None and distance <= 0.0:
            self._reject(family, "adaptation", task_type, "trivial_goal_already_solved", {"distance": distance})
        if task_type == "few_shot_replanning_after_action_gain_change" and not (metric_changed or dynamics_changed):
            self._reject(family, "adaptation", task_type, "dynamics_no_transition_change", {})

    def _reject(self, family: str, task_class: str, task_type: str, reason: str, details: dict[str, Any]) -> None:
        raise TaskSamplingRejected(TaskRejection(family=family, task_class=task_class, task_type=task_type, reason=reason, details=details))

    def _task_horizon(self, task: PlanningTask | InferenceTask | AdaptationTask) -> int:
        if isinstance(task, PlanningTask):
            return task.horizon_steps
        if isinstance(task, AdaptationTask):
            return task.evaluation_horizon_steps
        return int(task.metadata.get("horizon_steps", 1))

    def _goal_target_for_token(
        self,
        environment: Map2DEnvironment | ProcTHORScene,
        token: str,
        *,
        fallback: str | None = None,
    ) -> str:
        if isinstance(environment, ProcTHORScene):
            object_id = environment.goal_tokens.get(token)
            if object_id is None:
                return fallback or ""
            return environment.object_by_id(object_id).room_id
        return environment.goal_tokens.get(token, fallback or "")

    def _node_ids(self, environment: Map2DEnvironment | ProcTHORScene) -> list[str]:
        if isinstance(environment, ProcTHORScene):
            return sorted(environment.rooms)
        return sorted(environment.nodes)

    def _task_template_metadata(
        self,
        environment: Map2DEnvironment | ProcTHORScene,
        task_class: str,
        task_type: str,
        family: str,
        start_node_id: str | None,
        goal_node_id: str | None,
        *,
        goal_token: str | None = None,
        horizon_steps: int | None = None,
        expected_output_type: str | None = None,
        adaptation_budget_steps: int | None = None,
    ) -> dict[str, Any]:
        if isinstance(environment, ProcTHORScene):
            return self._task_template_metadata_3d(
                environment=environment,
                task_class=task_class,
                task_type=task_type,
                family=family,
                start_node_id=start_node_id,
                goal_node_id=goal_node_id,
                goal_token=goal_token,
                horizon_steps=horizon_steps,
                expected_output_type=expected_output_type,
                adaptation_budget_steps=adaptation_budget_steps,
            )
        return task_template_metadata(
            environment=environment,
            task_class=task_class,
            task_type=task_type,
            family=family,
            start_node_id=start_node_id,
            goal_node_id=goal_node_id,
            goal_token=goal_token,
            horizon_steps=horizon_steps,
            expected_output_type=expected_output_type,
            adaptation_budget_steps=adaptation_budget_steps,
        )

    def _task_template_metadata_3d(
        self,
        environment: ProcTHORScene,
        task_class: str,
        task_type: str,
        family: str,
        start_node_id: str | None,
        goal_node_id: str | None,
        *,
        goal_token: str | None = None,
        horizon_steps: int | None = None,
        expected_output_type: str | None = None,
        adaptation_budget_steps: int | None = None,
    ) -> dict[str, Any]:
        distance_steps = None
        start_goal_template_id = None
        start_role_template_id = None
        goal_role_template_id = None
        if start_node_id is not None:
            start_role_template_id = stable_template_hash({"room_id": start_node_id, "degree": len(environment.neighbors(start_node_id))})
            if goal_node_id is not None:
                goal_role_template_id = stable_template_hash({"room_id": goal_node_id, "degree": len(environment.neighbors(goal_node_id))})
                move_step = max(float(environment.control.get("move_step_m", 0.25)), 1e-9)
                distance = environment.shortest_path_length(start_node_id, goal_node_id)
                distance_steps = -1 if distance is None else int(round(distance / move_step))
                start_goal_template_id = stable_template_hash(
                    {
                        "start_role": start_role_template_id,
                        "goal_role": goal_role_template_id,
                        "distance_steps": distance_steps,
                    }
                )
            else:
                start_goal_template_id = stable_template_hash({"start_role": start_role_template_id, "goal_role": None})

        query_template_id = stable_template_hash(
            {
                "task_class": task_class,
                "task_type": task_type,
                "family": family,
                "expected_output_type": expected_output_type,
            }
        )
        budget_template_id = stable_template_hash(
            {
                "task_class": task_class,
                "task_type": task_type,
                "horizon_steps": horizon_steps,
                "adaptation_budget_steps": adaptation_budget_steps,
            }
        )
        task_template_id = stable_template_hash(
            {
                "task_class": task_class,
                "task_type": task_type,
                "family": family,
                "goal_token": goal_token,
                "query_template_id": query_template_id,
                "budget_template_id": budget_template_id,
                "start_goal_template_id": start_goal_template_id,
                "scene_fingerprint": stable_template_hash(
                    {
                        "motif_tag": environment.motif_tag,
                        "structural_signature": list(environment.structural_signature()),
                    }
                ),
            }
        )
        return {
            "task_template_id": task_template_id,
            "query_template_id": query_template_id,
            "budget_template_id": budget_template_id,
            "start_goal_template_id": start_goal_template_id,
            "start_role_template_id": start_role_template_id,
            "goal_role_template_id": goal_role_template_id,
            "distance_steps": distance_steps,
        }
