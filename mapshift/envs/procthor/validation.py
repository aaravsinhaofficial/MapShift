"""Validation hooks for MapShift-3D scenes."""

from __future__ import annotations

from typing import Any


def validate_procthor_scene(scene: Any) -> list[str]:
    """Return a list of validation issues for a sampled scene."""

    if scene is None:
        return ["scene is None"]
    return []
