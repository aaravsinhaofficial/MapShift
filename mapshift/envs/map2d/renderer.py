"""ASCII debug rendering utilities for MapShift-2D."""

from __future__ import annotations

from typing import Iterable

from .state import Cell, Map2DEnvironment


def _content_bounds(environment: Map2DEnvironment) -> tuple[int, int, int, int]:
    free_cells = [(row, col) for row in range(environment.height_cells) for col in range(environment.width_cells) if environment.is_free((row, col))]
    if not free_cells:
        return 0, environment.height_cells - 1, 0, environment.width_cells - 1

    rows = [cell[0] for cell in free_cells]
    cols = [cell[1] for cell in free_cells]
    margin = 2
    return (
        max(0, min(rows) - margin),
        min(environment.height_cells - 1, max(rows) + margin),
        max(0, min(cols) - margin),
        min(environment.width_cells - 1, max(cols) + margin),
    )


def render_ascii(environment: Map2DEnvironment, path: Iterable[Cell] | None = None, crop_to_content: bool = True) -> str:
    """Render an occupancy-grid environment as ASCII."""

    highlight = set(path or [])
    if crop_to_content:
        row_min, row_max, col_min, col_max = _content_bounds(environment)
    else:
        row_min, row_max, col_min, col_max = 0, environment.height_cells - 1, 0, environment.width_cells - 1

    rows: list[str] = []
    for row in range(row_min, row_max + 1):
        chars: list[str] = []
        for col in range(col_min, col_max + 1):
            cell = (row, col)
            char = "#" if not environment.is_free(cell) else "."
            if cell in highlight and environment.is_free(cell):
                char = "*"
            semantic = environment.semantic_label_for_cell(cell)
            if semantic:
                char = semantic
            chars.append(char)
        rows.append("".join(chars))
    return "\n".join(rows)


def render_debug_view(environment: Map2DEnvironment) -> str:
    """Return an ASCII rendering plus key metadata."""

    path_length = environment.shortest_path_length(environment.start_node_id, environment.goal_node_id)
    header = (
        f"id={environment.environment_id} "
        f"motif={environment.motif_tag} "
        f"free_cells={environment.free_cell_count()} "
        f"path_length={path_length}"
    )
    return header + "\n" + render_ascii(environment)
