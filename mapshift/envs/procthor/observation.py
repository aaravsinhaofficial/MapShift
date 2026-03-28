"""Observation scaffold for MapShift-3D."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ObservationFrame3D:
    frame_size: tuple[int, int]
    visible_objects: tuple[str, ...] = field(default_factory=tuple)
    pose_token: str = ""
