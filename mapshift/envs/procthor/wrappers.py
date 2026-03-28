"""ProcTHOR wrapper types used by MapShift-3D."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProcTHORWrapperConfig:
    scene_sampler: str
    observation_mode: str
