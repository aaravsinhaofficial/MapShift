"""Metric intervention implementation for the first executable path."""

from __future__ import annotations

from mapshift.envs.map2d.state import Map2DEnvironment

from .base import BaseIntervention, InterventionResult


class MetricIntervention(BaseIntervention):
    family = "metric"

    def apply(self, environment: Map2DEnvironment, severity: int, seed: int) -> InterventionResult:
        severity_value, operations = self._severity_config(severity)
        transformed = environment.clone(environment_id=f"{environment.environment_id}-metric-s{severity}")

        transformed.geometry_scale = severity_value
        transformed.dynamics.forward_gain = severity_value
        transformed.dynamics.odometry_bias_deg = max(0.0, (severity_value - 1.0) * 12.0)
        transformed.observation_radius_m = max(0.5, environment.observation_radius_m / max(severity_value, 1e-6))
        transformed.history.append(f"metric:{','.join(operations)}")
        transformed.metadata["metric_shift"] = {
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
