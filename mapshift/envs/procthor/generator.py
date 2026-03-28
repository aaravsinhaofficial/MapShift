"""ProcTHOR scene sampling scaffold for MapShift-3D."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mapshift.core.manifests import EnvironmentManifest


@dataclass(frozen=True)
class ProcTHORGenerationResult:
    manifest: EnvironmentManifest
    scene: Any | None = None


class ProcTHORGenerator:
    """Placeholder generator wrapper for ProcTHOR-backed scenes."""

    def __init__(self, config: Any) -> None:
        self.config = config

    def sample(self, seed: int) -> ProcTHORGenerationResult:
        raise NotImplementedError("ProcTHORGenerator.sample is scaffolded but not implemented yet.")
