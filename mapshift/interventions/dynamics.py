"""Dynamics intervention implementation for the first executable path."""

from __future__ import annotations

from mapshift.envs.map2d.state import Map2DEnvironment
from mapshift.envs.procthor.wrappers import ProcTHORScene
from mapshift.splits.motifs import stable_template_hash

from .base import BaseIntervention, InterventionResult


class DynamicsIntervention(BaseIntervention):
    family = "dynamics"

    def apply(self, environment: Map2DEnvironment | ProcTHORScene, severity: int, seed: int) -> InterventionResult:
        severity_value, operations = self._severity_config(severity)
        if isinstance(environment, Map2DEnvironment):
            transformed = environment.clone(environment_id=f"{environment.environment_id}-dynamics-s{severity}")
            transformed.dynamics.friction = severity_value
            transformed.dynamics.inertial_lag = max(0.0, 1.0 - severity_value)
            transformed.dynamics.action_asymmetry = 0.0 if severity == 0 else min(0.5, 0.1 * severity)
            transformed.history.append(f"dynamics:{','.join(operations)}")
            payload = {
                "friction": round(transformed.dynamics.friction, 6),
                "inertial_lag": round(transformed.dynamics.inertial_lag, 6),
                "action_asymmetry": round(transformed.dynamics.action_asymmetry, 6),
            }
        else:
            transformed = environment.clone(scene_id=f"{environment.scene_id}-dynamics-s{severity}")
            transformed.dynamics["friction"] = round(severity_value, 6)
            transformed.dynamics["inertial_lag"] = round(max(0.0, 1.0 - severity_value), 6)
            transformed.dynamics["action_asymmetry"] = round(0.0 if severity == 0 else min(0.5, 0.1 * severity), 6)
            transformed.metadata.setdefault("history", []).append(f"dynamics:{','.join(operations)}")
            payload = dict(sorted(transformed.dynamics.items()))

        transformed.metadata["dynamics_shift"] = {
            "severity": severity,
            "value": severity_value,
            "operations": list(operations),
            "dynamics_template_id": stable_template_hash({"family": "dynamics", "severity": severity, **payload}),
        }

        manifest = self._build_manifest(environment, transformed, severity, severity_value, seed)
        return InterventionResult(
            manifest=manifest,
            environment=transformed,
            preserved_attributes=self.family_config.preserve,
        )
