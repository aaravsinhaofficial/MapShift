"""Planning task containers for MapShift."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlanningTask:
    task_type: str
    horizon_steps: int
    goal_descriptor: str
