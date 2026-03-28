"""Procedural generation scaffold for MapShift-2D."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mapshift.core.manifests import EnvironmentManifest


@dataclass(frozen=True)
class Map2DGenerationResult:
    """Container returned by the 2D generator."""

    manifest: EnvironmentManifest
    environment: Any | None = None


class Map2DGenerator:
    """Placeholder generator for the 2D benchmark substrate."""

    def __init__(self, config: Any) -> None:
        self.config = config

    def generate(self, seed: int) -> Map2DGenerationResult:
        raise NotImplementedError("Map2DGenerator.generate is scaffolded but not implemented yet.")
