"""Shared helper utilities for MapShift calibration baselines."""

from __future__ import annotations

import math
from typing import Any

from mapshift.envs.map2d.state import Cell, Map2DEnvironment


def goal_node_for_task(environment: Map2DEnvironment, task: Any, remembered_goal_tokens: dict[str, str] | None = None) -> tuple[str | None, bool]:
    """Resolve the goal node a baseline will target and whether it matches the task."""

    goal_node_id = getattr(task, "goal_node_id", None)
    goal_token = getattr(task, "goal_token", None)
    if goal_token and remembered_goal_tokens and goal_token in remembered_goal_tokens:
        chosen = remembered_goal_tokens[goal_token]
        return chosen, chosen == goal_node_id
    return goal_node_id, True


def dynamics_cost_multiplier(environment: Map2DEnvironment) -> float:
    """Return a simple path-cost multiplier induced by current dynamics."""

    return max(
        1.0,
        environment.dynamics.friction
        + (environment.dynamics.inertial_lag * 0.5)
        + abs(environment.dynamics.action_asymmetry) * 0.5
        + abs(environment.dynamics.odometry_bias_deg) / 90.0,
    )


def family_shift_severity(environment: Map2DEnvironment, family: str) -> int:
    """Return the realized severity for one intervention family."""

    shift = environment.metadata.get(f"{family}_shift", {})
    if not isinstance(shift, dict):
        return 0
    severity = shift.get("severity", 0)
    return int(severity) if isinstance(severity, int) and not isinstance(severity, bool) else 0


def horizon_allows_path(task: Any, observed_length: float | None, environment: Map2DEnvironment) -> bool:
    """Return whether the task horizon can accommodate the observed path."""

    if observed_length is None:
        return False
    horizon_steps = int(getattr(task, "horizon_steps", getattr(task, "evaluation_horizon_steps", 0)))
    if horizon_steps <= 0:
        return True
    max_length = horizon_steps * environment.occupancy_resolution_m * max(environment.geometry_scale, 1e-6)
    return observed_length <= max_length


def cells_path_length(environment: Map2DEnvironment, path: list[Cell] | tuple[Cell, ...] | None) -> float | None:
    """Return geometric path length for a cell path."""

    if path is None:
        return None
    if not path:
        return 0.0
    return max(0, len(path) - 1) * environment.occupancy_resolution_m * environment.geometry_scale


def orthogonal_heuristic_path(environment: Map2DEnvironment, start: str | Cell, goal: str | Cell) -> tuple[Cell, ...] | None:
    """Return a direct horizontal/vertical heuristic path if unobstructed."""

    start_cell = environment.resolve_location(start)
    goal_cell = environment.resolve_location(goal)
    row, col = start_cell
    goal_row, goal_col = goal_cell

    def attempt(horizontal_first: bool) -> tuple[Cell, ...] | None:
        current_row, current_col = row, col
        path: list[Cell] = [(current_row, current_col)]
        if horizontal_first:
            step = 1 if goal_col >= current_col else -1
            for new_col in range(current_col + step, goal_col + step, step):
                if not environment.is_free((current_row, new_col)):
                    return None
                current_col = new_col
                path.append((current_row, current_col))
            step = 1 if goal_row >= current_row else -1
            for new_row in range(current_row + step, goal_row + step, step):
                if not environment.is_free((new_row, current_col)):
                    return None
                current_row = new_row
                path.append((current_row, current_col))
        else:
            step = 1 if goal_row >= current_row else -1
            for new_row in range(current_row + step, goal_row + step, step):
                if not environment.is_free((new_row, current_col)):
                    return None
                current_row = new_row
                path.append((current_row, current_col))
            step = 1 if goal_col >= current_col else -1
            for new_col in range(current_col + step, goal_col + step, step):
                if not environment.is_free((current_row, new_col)):
                    return None
                current_col = new_col
                path.append((current_row, current_col))
        return tuple(path)

    return attempt(horizontal_first=True) or attempt(horizontal_first=False)


def hidden_score(hidden_state: tuple[float, ...]) -> float:
    """Return a simple aggregate score from a hidden state."""

    if not hidden_state:
        return 0.0
    return sum(hidden_state) / len(hidden_state)


def path_efficiency_from_lengths(optimal_length: float | None, observed_length: float | None) -> float:
    """Return clipped path efficiency."""

    if optimal_length is None or observed_length is None or optimal_length <= 0.0 or observed_length <= 0.0:
        return 0.0
    return max(0.0, min(1.0, optimal_length / observed_length))


def stable_bucket_score(*parts: Any) -> float:
    """Return a deterministic score in [0, 1) for arbitrary hashable parts."""

    text = "|".join(str(part) for part in parts)
    total = 0
    for index, char in enumerate(text):
        total += (index + 1) * ord(char)
    return (total % 1000) / 1000.0
