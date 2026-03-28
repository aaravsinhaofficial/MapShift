"""Dynamics intervention implementation for the first executable path."""

from __future__ import annotations

from mapshift.envs.map2d.state import Map2DEnvironment

from .base import BaseIntervention, InterventionResult


class DynamicsIntervention(BaseIntervention):
    family = "dynamics"

    def apply(self, environment: Map2DEnvironment, severity: int, seed: int) -> InterventionResult:
        severity_value, operations = self._severity_config(severity)
        transformed = environment.clone(environment_id=f"{environment.environment_id}-dynamics-s{severity}")

        transformed.dynamics.friction = severity_value
        transformed.dynamics.inertial_lag = max(0.0, 1.0 - severity_value)
        transformed.dynamics.action_asymmetry = 0.0 if severity == 0 else min(0.5, 0.1 * severity)
        transformed.history.append(f"dynamics:{','.join(operations)}")
        transformed.metadata["dynamics_shift"] = {
            "severity": severity,
            "value": severity_value,
            "operations": list(operations),
        }

        manifest = self._build_manifest(environment, transformed, severity, severity_value, seed)
        return InterventionResult(
            manifest=manifest,
            environment=transformed,
            preserved_attributes=self.family_config.preserve,
        )
