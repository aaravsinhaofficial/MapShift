"""State containers for MapShift-2D."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AgentPose2D:
    x: float
    y: float
    theta_deg: float


@dataclass(frozen=True)
class Map2DState:
    pose: AgentPose2D
    timestep: int = 0
    semantic_tags: tuple[str, ...] = field(default_factory=tuple)
