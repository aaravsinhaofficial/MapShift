"""Metric intervention implementation for the first executable path."""

from __future__ import annotations

from mapshift.envs.map2d.state import Map2DEnvironment
from mapshift.envs.procthor.wrappers import ProcTHORScene
from mapshift.splits.motifs import stable_template_hash

from .base import BaseIntervention, InterventionResult


class MetricIntervention(BaseIntervention):
    family = "metric"

    def apply(self, environment: Map2DEnvironment | ProcTHORScene, severity: int, seed: int) -> InterventionResult:
        severity_value, operations = self._severity_config(severity)
        if isinstance(environment, Map2DEnvironment):
            transformed = environment.clone(environment_id=f"{environment.environment_id}-metric-s{severity}")
            transformed.geometry_scale = severity_value
            transformed.dynamics.forward_gain = severity_value
            transformed.dynamics.odometry_bias_deg = max(0.0, (severity_value - 1.0) * 12.0)
            transformed.observation_radius_m = max(0.5, environment.observation_radius_m / max(severity_value, 1e-6))
            transformed.history.append(f"metric:{','.join(operations)}")
            metric_payload = {
                "geometry_scale": round(transformed.geometry_scale, 6),
                "forward_gain": round(transformed.dynamics.forward_gain, 6),
                "observation_radius_m": round(transformed.observation_radius_m, 6),
            }
        else:
            transformed = environment.clone(scene_id=f"{environment.scene_id}-metric-s{severity}")
            transformed.control["move_step_m"] = round(float(environment.control.get("move_step_m", 0.25)) * severity_value, 6)
            transformed.observation["field_of_view_deg"] = round(
                max(45.0, float(environment.observation.get("field_of_view_deg", 90.0)) / max(severity_value, 1e-6)),
                6,
            )
            transformed.metadata.setdefault("history", []).append(f"metric:{','.join(operations)}")
            metric_payload = {
                "move_step_m": round(float(transformed.control.get("move_step_m", 0.25)), 6),
                "field_of_view_deg": round(float(transformed.observation.get("field_of_view_deg", 90.0)), 6),
            }

        transformed.metadata["metric_shift"] = {
            "severity": severity,
            "value": severity_value,
            "operations": list(operations),
            "metric_template_id": stable_template_hash({"family": "metric", "severity": severity, **metric_payload}),
        }

        manifest = self._build_manifest(environment, transformed, severity, severity_value, seed)
        return InterventionResult(
            manifest=manifest,
            environment=transformed,
            preserved_attributes=self.family_config.preserve,
        )
