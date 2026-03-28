"""Inference task containers for MapShift."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceTask:
    task_type: str
    query: str
    expected_output_type: str
