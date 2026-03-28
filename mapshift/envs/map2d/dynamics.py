"""Dynamics parameter containers for MapShift-2D."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DynamicsParameters2D:
    forward_gain: float = 1.0
    turn_gain: float = 1.0
    friction: float = 1.0
    inertial_lag: float = 0.0
    action_asymmetry: float = 0.0
    odometry_bias_deg: float = 0.0

    def clone(self) -> "DynamicsParameters2D":
        """Return a detached copy of the dynamics parameters."""

        return DynamicsParameters2D(
            forward_gain=self.forward_gain,
            turn_gain=self.turn_gain,
            friction=self.friction,
            inertial_lag=self.inertial_lag,
            action_asymmetry=self.action_asymmetry,
            odometry_bias_deg=self.odometry_bias_deg,
        )
