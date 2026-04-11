"""Shared baseline API and config loading for MapShift calibration systems."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from mapshift.core.manifests import RunManifest
from mapshift.core.registry import Registry
from mapshift.envs.map2d.observation import observe_egocentric
from mapshift.envs.map2d.state import AgentPose2D, Cell, Map2DEnvironment


_BASELINE_FACTORIES: Registry[Callable[["BaselineRunConfig"], "BaseBaselineModel"]] = Registry("baseline_factory")
_KNOWN_BASELINE_TIERS: dict[str, tuple[str, ...]] = {
    "oracle_post_intervention_planner": ("mapshift_2d", "mapshift_3d"),
    "same_environment_upper_baseline": ("mapshift_2d", "mapshift_3d"),
    "weak_heuristic_baseline": ("mapshift_2d", "mapshift_3d"),
    "monolithic_recurrent_world_model": ("mapshift_2d",),
    "persistent_memory_world_model": ("mapshift_2d",),
    "relational_graph_world_model": ("mapshift_2d",),
}
_TORCH_OPTIONAL_BASELINES = {
    "monolithic_recurrent_world_model",
    "persistent_memory_world_model",
    "relational_graph_world_model",
}


@dataclass(frozen=True)
class BaselineRunConfig:
    """Minimal config used to instantiate one calibration baseline."""

    schema_version: str
    run_name: str
    baseline_name: str
    seed: int
    exploration_budget_steps: int
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BaselineContext:
    """Context shared across baseline exploration/evaluation calls."""

    model_name: str
    exploration_budget_steps: int
    seed: int
    release_name: str = ""
    split_name: str = ""
    tier: str = "mapshift_2d"
    protocol_name: str = "post_intervention"


@dataclass(frozen=True)
class ExplorationResult:
    """Artifact returned by the reward-free exploration interface."""

    baseline_name: str
    environment_id: str
    exploration_steps: int
    visited_cells: tuple[Cell, ...]
    visited_node_ids: tuple[str, ...]
    hidden_state: tuple[float, ...] = field(default_factory=tuple)
    memory: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskEvaluationResult:
    """Normalized output from planning, inference, or adaptation evaluation."""

    baseline_name: str
    task_class: str
    task_type: str
    family: str
    success: bool
    solvable: bool
    observed_length: float | None = None
    oracle_length: float | None = None
    path_efficiency: float = 0.0
    oracle_gap: float | None = None
    predicted_answer: Any = None
    correct: bool | None = None
    adaptation_curve: tuple[float, ...] = field(default_factory=tuple)
    path_cells: tuple[Cell, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def primary_score(self) -> float:
        if self.task_class == "inference":
            return 1.0 if self.correct else 0.0
        if self.task_class == "adaptation" and self.adaptation_curve:
            return float(self.adaptation_curve[-1])
        return 1.0 if self.success else 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaselineModel(Protocol):
    """Common baseline interface used by evaluation runners."""

    name: str
    category: str
    learnable: bool
    implementation_kind: str
    parameter_count: int
    trainable_parameter_count: int
    supported_tiers: tuple[str, ...]

    def describe(self) -> dict[str, Any]:
        ...

    def explore(self, environment: Any, context: BaselineContext) -> ExplorationResult:
        ...

    def evaluate(self, environment: Any, task: Any, exploration: ExplorationResult, context: BaselineContext) -> TaskEvaluationResult:
        ...

    def adapt(self, environment: Any, task: Any, exploration: ExplorationResult, context: BaselineContext) -> TaskEvaluationResult:
        ...


class BaseBaselineModel:
    """Base implementation with default metadata and adaptation behavior."""

    name = "baseline"
    category = "generic"
    learnable = False
    implementation_kind = "deterministic_calibration_wrapper"
    parameter_count = 0
    trainable_parameter_count = 0
    supported_tiers = ("mapshift_2d",)

    def __init__(self, run_config: BaselineRunConfig) -> None:
        self.run_config = run_config
        self.seed = run_config.seed
        self.parameters = dict(run_config.parameters)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "learnable": self.learnable,
            "implementation_kind": self.implementation_kind,
            "parameter_count": self.parameter_count,
            "trainable_parameter_count": self.trainable_parameter_count,
            "supported_tiers": list(self.supported_tiers),
            "parameter_count_semantics": "trainable_model_parameters" if self.learnable else "deterministic_wrapper_complexity_proxy",
            "seed": self.seed,
            "parameters": dict(self.parameters),
        }

    def adapt(self, environment: Any, task: Any, exploration: ExplorationResult, context: BaselineContext) -> TaskEvaluationResult:
        return self.evaluate(environment, task, exploration, context)


def register_baseline(name: str, factory: Callable[[BaselineRunConfig], BaseBaselineModel]) -> None:
    """Register one baseline factory."""

    if name in _BASELINE_FACTORIES:
        return
    _BASELINE_FACTORIES.register(name, factory)


def supported_tiers_for_baseline_name(name: str) -> tuple[str, ...]:
    """Return the declared supported tiers for a registered or known baseline."""

    return _KNOWN_BASELINE_TIERS.get(name, ("mapshift_2d",))


def _ensure_builtin_registrations() -> None:
    """Import built-in calibration baselines so they self-register."""

    from . import heuristic, oracle  # noqa: F401

    for module_name in ("memory", "recurrent", "relational"):
        try:
            __import__(f"{__package__}.{module_name}", fromlist=[module_name])
        except ModuleNotFoundError as exc:
            if exc.name != "torch":
                raise


def instantiate_baseline(run_config: BaselineRunConfig) -> BaseBaselineModel:
    """Instantiate a registered baseline from a run config."""

    _ensure_builtin_registrations()
    if run_config.baseline_name not in _BASELINE_FACTORIES and run_config.baseline_name in _TORCH_OPTIONAL_BASELINES:
        raise ModuleNotFoundError(
            f"Baseline {run_config.baseline_name!r} requires the optional 'torch' dependency, which is not available."
        )
    return _BASELINE_FACTORIES.get(run_config.baseline_name)(run_config)


def load_baseline_run_config(path: str | Path) -> BaselineRunConfig:
    """Load and minimally validate a baseline run config."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Baseline run config must be an object: {path}")
    for field_name in ("schema_version", "run_name", "baseline_name"):
        if not isinstance(payload.get(field_name), str) or not str(payload[field_name]).strip():
            raise ValueError(f"Missing or invalid {field_name} in baseline run config: {path}")
    seed = payload.get("seed")
    budget = payload.get("exploration_budget_steps")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"seed must be an integer in baseline run config: {path}")
    if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
        raise ValueError(f"exploration_budget_steps must be a positive integer in baseline run config: {path}")
    parameters = payload.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError(f"parameters must be an object in baseline run config: {path}")
    return BaselineRunConfig(
        schema_version=str(payload["schema_version"]),
        run_name=str(payload["run_name"]),
        baseline_name=str(payload["baseline_name"]),
        seed=seed,
        exploration_budget_steps=budget,
        parameters=dict(parameters),
    )


def task_class_name(task: Any) -> str:
    """Infer the task class from a task container."""

    return type(task).__name__.replace("Task", "").lower()


def build_run_manifest(
    model: BaselineModel,
    context: BaselineContext,
    environment_ids: list[str],
    baseline_family: str,
) -> RunManifest:
    """Build a simple run manifest for one evaluation pass."""

    return RunManifest(
        artifact_id=f"run-{context.model_name}-{context.seed}",
        artifact_type="run",
        benchmark_version="0.1-draft",
        code_version="calibration-baselines-v1",
        config_hash=f"{context.model_name}-{context.seed}",
        run_id=f"{context.model_name}-{context.seed}",
        model_name=context.model_name,
        protocol_name=context.protocol_name,
        baseline_family=baseline_family,
        environment_ids=environment_ids,
        metadata=model.describe(),
    )


def deterministic_exploration_trace(environment: Map2DEnvironment, budget_steps: int, seed: int) -> tuple[tuple[Cell, ...], tuple[str, ...]]:
    """Return a deterministic reward-free exploration trace over free cells."""

    start = environment.start_cell
    frontier = [start]
    visited = {start}
    order: list[Cell] = []
    while frontier and len(order) < budget_steps:
        current = frontier.pop(0)
        order.append(current)
        neighbors = sorted(environment.neighbors(current))
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append(neighbor)

    if not order:
        order = [start]
    rng = random.Random(seed)
    while len(order) < min(budget_steps, max(1, len(order) * 2)):
        current = order[-1]
        neighbors = sorted(environment.neighbors(current))
        if not neighbors:
            break
        order.append(neighbors[(len(order) + seed + rng.randrange(len(neighbors))) % len(neighbors)])

    visited_nodes = tuple(
        sorted(node_id for node_id, node in environment.nodes.items() if node.cell in set(order))
    )
    return tuple(order[:budget_steps]), visited_nodes


def summarize_exploration_memory(environment: Map2DEnvironment, visited_cells: tuple[Cell, ...], visited_node_ids: tuple[str, ...]) -> dict[str, Any]:
    """Summarize exploration-time memory for calibration baselines."""

    visible_landmarks: set[str] = set()
    for cell in visited_cells[: min(len(visited_cells), 8)]:
        frame = observe_egocentric(
            environment,
            pose=AgentPose2D(x=float(cell[1]), y=float(cell[0]), theta_deg=0.0),
        )
        visible_landmarks.update(frame.visible_landmarks)
    return {
        "visited_ratio": len(set(visited_cells)) / max(1, environment.free_cell_count()),
        "remembered_goal_tokens": dict(environment.goal_tokens),
        "remembered_landmarks": dict(environment.landmark_by_node),
        "base_geometry_signature": environment.geometry_signature(),
        "base_edge_signature": tuple(environment.edge_list()),
        "base_semantic_signature": environment.semantic_signature(),
        "base_dynamics_signature": environment.dynamics_signature(),
        "visible_landmarks": tuple(sorted(visible_landmarks)),
        "visited_node_ids": visited_node_ids,
        "base_environment_payload": environment.to_dict(),
    }
