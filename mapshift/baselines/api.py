"""Shared baseline API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class BaselineContext:
    model_name: str
    exploration_budget_steps: int
    seed: int


class BaselineModel(Protocol):
    """Common baseline interface used by evaluation runners."""

    def explore(self, environment: Any, context: BaselineContext) -> Any:
        ...

    def evaluate(self, environment: Any, task: Any, context: BaselineContext) -> Any:
        ...
