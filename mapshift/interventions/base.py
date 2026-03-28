"""Base classes for structured benchmark interventions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from mapshift.core.manifests import InterventionManifest
from mapshift.core.schemas import FamilyInterventionConfig

from mapshift.envs.map2d.state import Map2DEnvironment


@dataclass(frozen=True)
class InterventionResult:
    manifest: InterventionManifest
    environment: Any | None = None
    preserved_attributes: tuple[str, ...] = field(default_factory=tuple)


class BaseIntervention:
    """Common interface for benchmark interventions."""

    family: str = "unknown"

    def __init__(self, family_config: FamilyInterventionConfig) -> None:
        self.family_config = family_config
        self.family = family_config.family

    def apply(self, environment: Any, severity: int, seed: int) -> InterventionResult:
        raise NotImplementedError(f"{self.__class__.__name__}.apply is scaffolded but not implemented yet.")

    def _severity_config(self, severity: int) -> tuple[float, tuple[str, ...]]:
        key = str(severity)
        if key not in self.family_config.severity_levels:
            raise ValueError(f"Unsupported severity {severity} for family {self.family}")
        level = self.family_config.severity_levels[key]
        return level.value, level.interventions

    def _config_hash(self) -> str:
        config_blob = json.dumps(asdict(self.family_config), sort_keys=True).encode("utf-8")
        return hashlib.sha1(config_blob).hexdigest()[:12]

    def _build_manifest(
        self,
        environment: Map2DEnvironment,
        transformed_environment: Map2DEnvironment,
        severity: int,
        severity_value: float,
        seed: int,
    ) -> InterventionManifest:
        return InterventionManifest(
            artifact_id=f"intervention-{transformed_environment.environment_id}",
            artifact_type="intervention",
            benchmark_version="0.1-draft",
            code_version="occupancy-grid-v1",
            config_hash=self._config_hash(),
            parent_ids=[environment.environment_id],
            seed_values=[seed],
            metadata={
                "transformed_environment_id": transformed_environment.environment_id,
                "serialized_environment": transformed_environment.to_dict(),
            },
            base_environment_id=environment.environment_id,
            intervention_family=self.family,
            severity_level=severity,
            severity_parameter=self.family_config.severity_parameter,
            severity_value=severity_value,
        )
