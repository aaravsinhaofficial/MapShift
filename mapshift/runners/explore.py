"""Exploration runner scaffold."""

from __future__ import annotations

from typing import Any


def run_exploration(model: Any, environment: Any, context: Any) -> Any:
    """Run a reward-free exploration pass."""

    return model.explore(environment, context)
