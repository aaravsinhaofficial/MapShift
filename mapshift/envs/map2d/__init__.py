"""2D benchmark substrate for MapShift."""

from __future__ import annotations

from .state import Map2DEnvironment, Map2DNode

__all__ = [
    "Map2DEnvironment",
    "Map2DGenerationResult",
    "Map2DGenerator",
    "Map2DNode",
    "ObservationFrame2D",
    "observe_egocentric",
    "render_ascii",
    "render_debug_view",
]


def __getattr__(name: str):
    if name in {"Map2DGenerationResult", "Map2DGenerator"}:
        from .generator import Map2DGenerationResult, Map2DGenerator

        return {"Map2DGenerationResult": Map2DGenerationResult, "Map2DGenerator": Map2DGenerator}[name]
    if name in {"ObservationFrame2D", "observe_egocentric"}:
        from .observation import ObservationFrame2D, observe_egocentric

        return {"ObservationFrame2D": ObservationFrame2D, "observe_egocentric": observe_egocentric}[name]
    if name in {"render_ascii", "render_debug_view"}:
        from .renderer import render_ascii, render_debug_view

        return {"render_ascii": render_ascii, "render_debug_view": render_debug_view}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
