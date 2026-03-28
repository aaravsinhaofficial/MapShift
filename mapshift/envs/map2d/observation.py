"""Observation containers for MapShift-2D."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ObservationFrame2D:
    geometry_patch: tuple[tuple[int, ...], ...] = field(default_factory=tuple)
    visible_landmarks: tuple[str, ...] = field(default_factory=tuple)
    agent_heading_deg: float = 0.0
