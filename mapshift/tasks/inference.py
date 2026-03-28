"""Inference task containers for MapShift."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class InferenceTask:
    task_type: str
    family: str
    query: str
    expected_output_type: str
    expected_answer: Any
    metadata: dict[str, Any] = field(default_factory=dict)
