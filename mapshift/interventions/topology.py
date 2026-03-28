"""Topology intervention implementation for the first executable path."""

from __future__ import annotations

from mapshift.envs.map2d.state import Map2DEnvironment

from .base import BaseIntervention, InterventionResult


class TopologyIntervention(BaseIntervention):
    family = "topology"

    def apply(self, environment: Map2DEnvironment, severity: int, seed: int) -> InterventionResult:
        severity_value, operations = self._severity_config(severity)
        transformed = environment.clone(environment_id=f"{environment.environment_id}-topology-s{severity}")
        applied_operations: list[str] = []
        protected_edges: set[tuple[str, str]] = set()

        if severity >= 1:
            shortcuts = transformed.candidate_shortcuts()
            if shortcuts:
                left, right, _distance = shortcuts[0]
                transformed.add_edge(left, right)
                applied_operations.append(f"add_shortcut:{left}:{right}")
                protected_edges.add(tuple(sorted((left, right))))
            else:
                components = [sorted(component) for component in transformed.connected_components()]
                if len(components) > 1:
                    left = components[0][0]
                    right = components[1][0]
                    transformed.add_edge(left, right)
                    applied_operations.append(f"bridge_components:{left}:{right}")
                    protected_edges.add(tuple(sorted((left, right))))

        if severity >= 2:
            removable = transformed.removable_edges()
            removable = [edge for edge in removable if tuple(sorted((edge[0], edge[1]))) not in protected_edges]
            if removable:
                left, right, _delta = removable[0]
                transformed.remove_edge(left, right)
                if transformed.reachable(transformed.start_node_id, transformed.goal_node_id):
                    applied_operations.append(f"remove_edge:{left}:{right}")
                else:
                    transformed.add_edge(left, right)
                    protected_edges.add(tuple(sorted((left, right))))

        if severity >= 3:
            removable = transformed.removable_edges()
            removable = [edge for edge in removable if tuple(sorted((edge[0], edge[1]))) not in protected_edges]
            if removable:
                left, right, _delta = removable[-1]
                transformed.remove_edge(left, right)
                if transformed.reachable(transformed.start_node_id, transformed.goal_node_id):
                    applied_operations.append(f"remove_aux_edge:{left}:{right}")
                else:
                    transformed.add_edge(left, right)

        transformed.history.append(f"topology:{','.join(operations)}")
        transformed.metadata["topology_shift"] = {
            "severity": severity,
            "value": severity_value,
            "configured_operations": list(operations),
            "applied_operations": applied_operations,
            "edge_count": transformed.edge_count(),
        }

        manifest = self._build_manifest(environment, transformed, severity, severity_value, seed)
        return InterventionResult(
            manifest=manifest,
            environment=transformed,
            preserved_attributes=self.family_config.preserve,
        )
