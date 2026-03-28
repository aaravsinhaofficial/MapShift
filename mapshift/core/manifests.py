"""Manifest dataclasses for release artifacts and evaluation runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List


def utc_timestamp() -> str:
    """Return a stable UTC timestamp string."""

    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ArtifactManifest:
    """Base provenance container for benchmark artifacts."""

    artifact_id: str
    artifact_type: str
    benchmark_version: str
    code_version: str
    config_hash: str
    created_at: str = field(default_factory=utc_timestamp)
    parent_ids: List[str] = field(default_factory=list)
    seed_values: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EnvironmentManifest(ArtifactManifest):
    """Provenance for a base or intervened environment instance."""

    environment_id: str = ""
    tier: str = ""
    motif_tags: List[str] = field(default_factory=list)
    split_name: str = ""


@dataclass(frozen=True)
class InterventionManifest(ArtifactManifest):
    """Provenance for a structured environment intervention."""

    base_environment_id: str = ""
    intervention_family: str = ""
    severity_level: int = 0
    severity_parameter: str = ""
    severity_value: float = 0.0


@dataclass(frozen=True)
class TaskManifest(ArtifactManifest):
    """Provenance for a post-intervention evaluation task."""

    task_id: str = ""
    task_class: str = ""
    task_type: str = ""
    base_environment_id: str = ""
    intervened_environment_id: str = ""
    horizon_steps: int = 0


@dataclass(frozen=True)
class RunManifest(ArtifactManifest):
    """Provenance for a model run under a benchmark protocol."""

    run_id: str = ""
    model_name: str = ""
    protocol_name: str = ""
    baseline_family: str = ""
    environment_ids: List[str] = field(default_factory=list)
