"""Dynamics parameter scaffold for MapShift-2D."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DynamicsParameters2D:
    forward_gain: float = 1.0
    turn_gain: float = 1.0
    friction: float = 1.0
    inertial_lag: float = 0.0
