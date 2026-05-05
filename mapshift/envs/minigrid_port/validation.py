"""Invariant checks for MiniGrid matched intervention pairs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .state import MiniGridPortEnvironment


@dataclass(frozen=True)
class MiniGridPortValidationResult:
    family: str
    severity: int
    issues: tuple[str, ...]
    metrics: dict[str, Any]

    @property
    def ok(self) -> bool:
        return not self.issues


def validate_minigrid_intervention_pair(
    base_environment: MiniGridPortEnvironment,
    transformed_environment: MiniGridPortEnvironment,
    *,
    family: str,
    severity: int,
) -> MiniGridPortValidationResult:
    issues: list[str] = []
    if severity == 0 and base_environment.state_signature() != transformed_environment.state_signature():
        issues.append("severity-0 intervention does not preserve MiniGrid port state")

    if not transformed_environment.shortest_path():
        issues.append("transformed MiniGrid port environment is not solvable from start to active goal")

    topology_changed = base_environment.topology_signature() != transformed_environment.topology_signature()
    semantic_changed = base_environment.semantic_signature() != transformed_environment.semantic_signature()
    dynamics_changed = base_environment.dynamics_signature() != transformed_environment.dynamics_signature()
    metric_changed = base_environment.metric_signature() != transformed_environment.metric_signature()

    if severity > 0:
        if family == "metric":
            if not metric_changed:
                issues.append("metric intervention did not change movement-cost scale")
            if topology_changed or semantic_changed or dynamics_changed:
                issues.append("metric intervention changed non-target MiniGrid factors")
        elif family == "topology":
            if not topology_changed:
                issues.append("topology intervention did not change wall layout")
            if semantic_changed or dynamics_changed or metric_changed:
                issues.append("topology intervention changed non-target MiniGrid factors")
        elif family == "dynamics":
            if not dynamics_changed:
                issues.append("dynamics intervention did not change slip probability")
            if topology_changed or semantic_changed or metric_changed:
                issues.append("dynamics intervention changed non-target MiniGrid factors")
        elif family == "semantic":
            if not semantic_changed:
                issues.append("semantic intervention did not change active goal token")
            if topology_changed or dynamics_changed or metric_changed:
                issues.append("semantic intervention changed non-target MiniGrid factors")
        else:
            issues.append(f"unsupported MiniGrid intervention family: {family}")

    metrics = {
        "topology_changed": topology_changed,
        "semantic_changed": semantic_changed,
        "dynamics_changed": dynamics_changed,
        "metric_changed": metric_changed,
        "base_path_length": max(0, len(base_environment.shortest_path()) - 1),
        "transformed_path_length": max(0, len(transformed_environment.shortest_path()) - 1),
        "movement_cost_scale": transformed_environment.movement_cost_scale,
        "slip_probability": transformed_environment.slip_probability,
        "active_token": transformed_environment.active_token,
    }
    return MiniGridPortValidationResult(family=family, severity=severity, issues=tuple(issues), metrics=metrics)
