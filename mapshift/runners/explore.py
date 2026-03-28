"""Exploration runner helpers."""

from __future__ import annotations

from typing import Any

from mapshift.baselines.api import BaselineContext, ExplorationResult


def run_exploration(model: Any, environment: Any, context: BaselineContext) -> ExplorationResult:
    """Run a reward-free exploration pass."""

    return model.explore(environment, context)
