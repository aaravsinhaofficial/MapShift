"""Batch execution containers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BatchRunSpec:
    name: str
    seed_count: int
