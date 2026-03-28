"""Base classes for structured benchmark interventions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mapshift.core.manifests import InterventionManifest


@dataclass(frozen=True)
class InterventionResult:
    manifest: InterventionManifest
    environment: Any | None = None
    preserved_attributes: tuple[str, ...] = field(default_factory=tuple)


class BaseIntervention:
    """Common interface for benchmark interventions."""

    family: str = "unknown"

    def apply(self, environment: Any, severity: int, seed: int) -> InterventionResult:
        raise NotImplementedError(f"{self.__class__.__name__}.apply is scaffolded but not implemented yet.")
