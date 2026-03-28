"""Base classes for structured benchmark interventions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from mapshift.core.manifests import InterventionManifest
from mapshift.core.schemas import FamilyInterventionConfig
from mapshift.splits.motifs import semantic_template_metadata, stable_template_hash, structural_signature_for_environment

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

    def _invalidate_cached_metadata(
        self,
        environment: Map2DEnvironment,
        *,
        invalidate_structural: bool = False,
        invalidate_semantic: bool = False,
    ) -> None:
        """Drop derived tags/signatures that are no longer valid after an edit."""

        if invalidate_structural:
            for key in ("motif_tags", "structural_signature", "node_role_template_ids"):
                environment.metadata.pop(key, None)
        if invalidate_semantic:
            for key in (
                "landmark_layout_template_id",
                "goal_token_template_id",
                "semantic_template_id",
                "landmark_layout_template",
                "goal_token_template",
            ):
                environment.metadata.pop(key, None)

    def _build_manifest(
        self,
        environment: Map2DEnvironment,
        transformed_environment: Map2DEnvironment,
        severity: int,
        severity_value: float,
        seed: int,
    ) -> InterventionManifest:
        source_structural = environment.metadata.get("structural_signature")
        if not isinstance(source_structural, dict):
            source_structural = structural_signature_for_environment(environment).to_dict()
        target_structural = structural_signature_for_environment(transformed_environment).to_dict()
        target_semantics = semantic_template_metadata(transformed_environment)
        transformed_environment.metadata.update(
            {
                "motif_tags": list(target_structural["motif_tags"]),
                "structural_signature": target_structural,
                **target_semantics,
            }
        )
        family_shift_metadata = dict(transformed_environment.metadata.get(f"{self.family}_shift", {}))
        intervention_template_id = stable_template_hash(
            {
                "family": self.family,
                "severity": severity,
                "severity_value": round(severity_value, 6),
                "source_fingerprint": source_structural.get("normalized_fingerprint", ""),
                "target_fingerprint": target_structural.get("normalized_fingerprint", ""),
                "family_shift": family_shift_metadata,
            }
        )
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
                "source_structural_signature": source_structural,
                "target_structural_signature": target_structural,
                "source_semantic_template_id": environment.metadata.get("semantic_template_id", ""),
                "target_semantic_template_id": target_semantics["semantic_template_id"],
                "intervention_template_id": intervention_template_id,
                "family_shift_metadata": family_shift_metadata,
                "serialized_environment": transformed_environment.to_dict(),
            },
            base_environment_id=environment.environment_id,
            intervention_family=self.family,
            severity_level=severity,
            severity_parameter=self.family_config.severity_parameter,
            severity_value=severity_value,
        )
