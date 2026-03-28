"""Task sampling scaffold."""

from __future__ import annotations

from typing import Any


class TaskSampler:
    """Placeholder task sampler conditioned on base and intervened environments."""

    def __init__(self, config: Any) -> None:
        self.config = config

    def sample(self, base_environment: Any, intervened_environment: Any, family: str, seed: int) -> Any:
        raise NotImplementedError("TaskSampler.sample is scaffolded but not implemented yet.")
