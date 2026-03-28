"""Semantic intervention implementation for the first executable path."""

from __future__ import annotations

from mapshift.envs.map2d.state import Map2DEnvironment
from mapshift.splits.motifs import semantic_remap_template_id

from .base import BaseIntervention, InterventionResult


def _rotate_assignments(values: list[str], active_count: int) -> list[str]:
    if active_count <= 1 or len(values) <= 1:
        return list(values)
    bounded = min(len(values), active_count)
    rotated_prefix = values[1:bounded] + values[:1]
    return rotated_prefix + values[bounded:]


class SemanticIntervention(BaseIntervention):
    family = "semantic"

    def apply(self, environment: Map2DEnvironment, severity: int, seed: int) -> InterventionResult:
        severity_value, operations = self._severity_config(severity)
        transformed = environment.clone(environment_id=f"{environment.environment_id}-semantic-s{severity}")
        self._invalidate_cached_metadata(transformed, invalidate_semantic=True)

        token_names = sorted(transformed.goal_tokens)
        token_active_count = min(len(token_names), severity + 1)
        if severity > 0 and token_names and token_active_count > 1:
            goal_nodes = [transformed.goal_tokens[name] for name in token_names]
            rotated_nodes = _rotate_assignments(goal_nodes, token_active_count)
            transformed.goal_tokens = {token: rotated_nodes[index] for index, token in enumerate(token_names)}

        landmark_nodes = sorted(transformed.landmark_by_node)
        landmark_active_count = min(len(landmark_nodes), (severity * 2) if severity > 0 else 0)
        if severity > 0 and len(landmark_nodes) > 1 and landmark_active_count > 1:
            labels = [transformed.landmark_by_node[node_id] for node_id in landmark_nodes]
            labels = _rotate_assignments(labels, landmark_active_count)
            transformed.landmark_by_node = {node_id: labels[index] for index, node_id in enumerate(landmark_nodes)}

        transformed.history.append(f"semantic:{','.join(operations)}")
        transformed.metadata["semantic_shift"] = {
            "severity": severity,
            "value": severity_value,
            "operations": list(operations),
            "semantic_remap_template_id": semantic_remap_template_id(environment, transformed),
        }

        manifest = self._build_manifest(environment, transformed, severity, severity_value, seed)
        return InterventionResult(
            manifest=manifest,
            environment=transformed,
            preserved_attributes=self.family_config.preserve,
        )
