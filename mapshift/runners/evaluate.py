"""Evaluation runner scaffold."""

from __future__ import annotations

from typing import Any


def run_evaluation(model: Any, environment: Any, task: Any, context: Any) -> Any:
    """Run a post-intervention evaluation pass."""

    return model.evaluate(environment, task, context)
