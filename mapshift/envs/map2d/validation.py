"""Validation hooks for MapShift-2D instances."""

from __future__ import annotations

from typing import Any


def validate_map2d_instance(environment: Any) -> list[str]:
    """Return a list of validation issues for a generated environment."""

    if environment is None:
        return ["environment is None"]
    return []
