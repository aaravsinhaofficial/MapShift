"""Adaptation task containers for MapShift."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AdaptationTask:
    task_type: str
    family: str
    adaptation_budget_steps: int
    evaluation_horizon_steps: int
    start_node_id: str
    goal_node_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
