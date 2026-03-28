"""Observation utilities for MapShift-2D."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .state import AgentPose2D, Cell, Map2DEnvironment


def _bresenham_line(start: Cell, goal: Cell) -> list[Cell]:
    """Return integer cells on the line between start and goal."""

    row0, col0 = start
    row1, col1 = goal
    d_row = abs(row1 - row0)
    d_col = abs(col1 - col0)
    step_row = 1 if row0 < row1 else -1
    step_col = 1 if col0 < col1 else -1

    error = d_col - d_row
    row, col = row0, col0
    cells = []
    while True:
        cells.append((row, col))
        if row == row1 and col == col1:
            break
        twice_error = 2 * error
        if twice_error > -d_row:
            error -= d_row
            col += step_col
        if twice_error < d_col:
            error += d_col
            row += step_row
    return cells


def _relative_heading_deg(origin: Cell, target: Cell) -> float:
    d_row = target[0] - origin[0]
    d_col = target[1] - origin[1]
    angle = math.degrees(math.atan2(d_row, d_col))
    return angle


@dataclass(frozen=True)
class ObservationFrame2D:
    geometry_patch: tuple[tuple[int, ...], ...] = field(default_factory=tuple)
    semantic_patch: tuple[tuple[str, ...], ...] = field(default_factory=tuple)
    visible_landmarks: tuple[str, ...] = field(default_factory=tuple)
    agent_heading_deg: float = 0.0
    origin_row: int = 0
    origin_col: int = 0


def observe_egocentric(environment: Map2DEnvironment, pose: AgentPose2D | None = None) -> ObservationFrame2D:
    """Return a config-driven egocentric local observation."""

    active_pose = pose or environment.agent_pose
    if active_pose is None:
        start_row, start_col = environment.start_cell
        active_pose = AgentPose2D(x=float(start_col), y=float(start_row), theta_deg=0.0)

    center = environment.resolve_location(active_pose)
    radius_cells = max(1, int(round(environment.observation_radius_m / max(environment.occupancy_resolution_m, 1e-6))))
    visible_landmarks: set[str] = set()
    geometry_rows: list[tuple[int, ...]] = []
    semantic_rows: list[tuple[str, ...]] = []

    for row in range(center[0] - radius_cells, center[0] + radius_cells + 1):
        geometry_row: list[int] = []
        semantic_row: list[str] = []
        for col in range(center[1] - radius_cells, center[1] + radius_cells + 1):
            cell = (row, col)
            if not environment.in_bounds(cell):
                geometry_row.append(-1)
                semantic_row.append("")
                continue

            distance = math.dist((center[0], center[1]), (row, col))
            if distance > radius_cells:
                geometry_row.append(-1)
                semantic_row.append("")
                continue

            heading = _relative_heading_deg(center, cell)
            relative = (heading - active_pose.theta_deg + 180.0) % 360.0 - 180.0
            if abs(relative) > environment.field_of_view_deg / 2.0:
                geometry_row.append(-1)
                semantic_row.append("")
                continue

            line = _bresenham_line(center, cell)
            visible = True
            for intermediate in line[1:-1]:
                if not environment.is_free(intermediate):
                    visible = False
                    break
            if not visible:
                geometry_row.append(-1)
                semantic_row.append("")
                continue

            geometry_row.append(1 if environment.is_free(cell) else 0)
            label = environment.semantic_label_for_cell(cell) if environment.semantic_channels else ""
            if label and environment.semantic_channels:
                semantic_row.append(label)
                if cell in (node.cell for node in environment.nodes.values()):
                    for node_id, node in environment.nodes.items():
                        if node.cell == cell and node_id in environment.landmark_by_node:
                            visible_landmarks.add(environment.landmark_by_node[node_id])
                            break
            else:
                semantic_row.append("")

        geometry_rows.append(tuple(geometry_row))
        semantic_rows.append(tuple(semantic_row))

    origin_row = center[0] - radius_cells
    origin_col = center[1] - radius_cells
    return ObservationFrame2D(
        geometry_patch=tuple(geometry_rows),
        semantic_patch=tuple(semantic_rows),
        visible_landmarks=tuple(sorted(visible_landmarks)),
        agent_heading_deg=active_pose.theta_deg,
        origin_row=origin_row,
        origin_col=origin_col,
    )
