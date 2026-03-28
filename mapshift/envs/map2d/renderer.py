"""Debug rendering utilities for MapShift-2D."""

from __future__ import annotations

from typing import Any


def render_debug_view(environment: Any) -> str:
    """Return a minimal textual placeholder for a debug rendering."""

    return f"MapShift-2D debug render unavailable for object type {type(environment).__name__}"
