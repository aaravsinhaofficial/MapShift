"""Planning task containers for MapShift."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PlanningTask:
    task_type: str
    horizon_steps: int
    family: str
    start_node_id: str
    goal_node_id: str | None
    goal_token: str | None
    goal_descriptor: str
    metadata: dict[str, Any] = field(default_factory=dict)
