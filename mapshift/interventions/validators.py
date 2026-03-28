"""Validator hooks for intervention isolation checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from mapshift.analysis.severity import severity_is_monotone
from mapshift.core.schemas import FamilyInterventionConfig
from mapshift.envs.map2d.state import Map2DEnvironment


def _node_reachability_matrix(environment: Map2DEnvironment) -> tuple[tuple[str, str, bool], ...]:
    node_ids = sorted(environment.nodes)
    entries = []
    for left in node_ids:
        for right in node_ids:
            entries.append((left, right, environment.reachable(left, right)))
    return tuple(entries)


def _core_state_signature(environment: Map2DEnvironment) -> tuple[Any, ...]:
    return (
        environment.geometry_signature(),
        environment.semantic_signature(),
        environment.dynamics_signature(),
        environment.observation_radius_m,
        environment.field_of_view_deg,
        environment.semantic_channels,
        _node_reachability_matrix(environment),
    )


def _normalized_state_payload(environment: Map2DEnvironment) -> dict[str, Any]:
    payload = environment.to_dict()
    payload.pop("environment_id", None)
    payload.pop("history", None)
    payload.pop("metadata", None)
    return payload


def intervention_magnitude(
    base_environment: Map2DEnvironment,
    transformed_environment: Map2DEnvironment,
    family: str,
) -> float:
    """Return a family-specific intervention magnitude on the realized substrate."""

    if family == "metric":
        return abs(transformed_environment.geometry_scale - base_environment.geometry_scale) + abs(
            transformed_environment.observation_radius_m - base_environment.observation_radius_m
        )
    if family == "topology":
        base_edges = set(base_environment.edge_list())
        transformed_edges = set(transformed_environment.edge_list())
        return float(len(base_edges.symmetric_difference(transformed_edges)))
    if family == "dynamics":
        return sum(abs(right - left) for left, right in zip(base_environment.dynamics_signature(), transformed_environment.dynamics_signature()))
    if family == "semantic":
        goal_changes = sum(
            1
            for token, node_id in base_environment.goal_tokens.items()
            if transformed_environment.goal_tokens.get(token) != node_id
        )
        landmark_changes = sum(
            1
            for node_id, label in base_environment.landmark_by_node.items()
            if transformed_environment.landmark_by_node.get(node_id) != label
        )
        return float(goal_changes + landmark_changes)
    raise ValueError(f"Unsupported family for intervention magnitude: {family}")


@dataclass(frozen=True)
class InterventionValidationResult:
    family: str
    severity: int
    issues: tuple[str, ...]
    metrics: dict[str, Any]

    @property
    def ok(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "severity": self.severity,
            "issues": list(self.issues),
            "metrics": dict(self.metrics),
            "ok": self.ok,
        }


def validate_intervention_invariants(issues: Iterable[str]) -> list[str]:
    """Normalize a collection of intervention validation issues."""

    return [issue for issue in issues if issue]


def validate_severity_config(family_config: FamilyInterventionConfig) -> list[str]:
    """Validate monotone severity magnitude for one family config."""

    if severity_is_monotone(family_config):
        return []
    return [f"{family_config.family} severity ladder is not monotone away from severity 0"]


def validate_intervention_pair(
    base_environment: Map2DEnvironment,
    transformed_environment: Map2DEnvironment,
    family: str,
    severity: int,
    family_config: FamilyInterventionConfig,
) -> InterventionValidationResult:
    """Validate intervention-family invariants on a base/transformed pair."""

    issues: list[str] = list(validate_severity_config(family_config))
    base_edges = base_environment.edge_list()
    transformed_edges = transformed_environment.edge_list()
    base_reachability = _node_reachability_matrix(base_environment)
    transformed_reachability = _node_reachability_matrix(transformed_environment)

    if severity == 0 and _normalized_state_payload(base_environment) != _normalized_state_payload(transformed_environment):
        issues.append("no-op intervention does not reproduce the base environment state")

    if family == "semantic":
        if base_environment.geometry_signature() != transformed_environment.geometry_signature():
            issues.append("semantic shift changed geometry")
        if base_reachability != transformed_reachability:
            issues.append("semantic shift changed reachability")
        if base_environment.dynamics_signature() != transformed_environment.dynamics_signature():
            issues.append("semantic shift changed dynamics")

    if family == "topology":
        if severity > 0 and base_edges == transformed_edges:
            issues.append("topology shift did not modify connectivity structure")
        if base_environment.semantic_signature() != transformed_environment.semantic_signature():
            issues.append("topology shift changed semantic assignments")
        if base_environment.dynamics_signature() != transformed_environment.dynamics_signature():
            issues.append("topology shift changed dynamics")

    if family == "metric":
        if base_edges != transformed_edges:
            issues.append("metric shift changed topology labels")
        if len(base_environment.connected_components()) != len(transformed_environment.connected_components()):
            issues.append("metric shift changed connected-component structure")
        if base_environment.semantic_signature() != transformed_environment.semantic_signature():
            issues.append("metric shift changed semantic assignments")

    if family == "dynamics":
        if base_environment.geometry_signature() != transformed_environment.geometry_signature():
            issues.append("dynamics shift changed geometry")
        if base_environment.semantic_signature() != transformed_environment.semantic_signature():
            issues.append("dynamics shift changed semantic assignments")

    metrics = {
        "base_edge_count": base_environment.edge_count(),
        "transformed_edge_count": transformed_environment.edge_count(),
        "base_path_length": base_environment.shortest_path_length(base_environment.start_node_id, base_environment.goal_node_id),
        "transformed_path_length": transformed_environment.shortest_path_length(transformed_environment.start_node_id, transformed_environment.goal_node_id),
        "geometry_changed": base_environment.geometry_signature() != transformed_environment.geometry_signature(),
        "semantics_changed": base_environment.semantic_signature() != transformed_environment.semantic_signature(),
        "dynamics_changed": base_environment.dynamics_signature() != transformed_environment.dynamics_signature(),
        "reachability_changed": base_reachability != transformed_reachability,
        "magnitude": intervention_magnitude(base_environment, transformed_environment, family),
    }
    return InterventionValidationResult(family=family, severity=severity, issues=tuple(validate_intervention_invariants(issues)), metrics=metrics)


def validate_intervention_manifest_roundtrip(metadata: dict[str, Any]) -> list[str]:
    """Validate that an intervention manifest contains a replayable environment payload."""

    issues: list[str] = []
    payload = metadata.get("serialized_environment")
    if not isinstance(payload, dict):
        return ["intervention manifest missing serialized_environment payload"]
    replayed = Map2DEnvironment.from_dict(payload)
    if replayed.to_dict() != payload:
        issues.append("serialized intervention environment is not stable under replay")
    return issues
