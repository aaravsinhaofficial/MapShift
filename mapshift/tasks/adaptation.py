"""Adaptation task containers for MapShift."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdaptationTask:
    task_type: str
    adaptation_budget_steps: int
    evaluation_horizon_steps: int
