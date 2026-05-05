"""Portable MiniGrid state representation for MapShift intervention wrappers."""

from __future__ import annotations

import json
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Iterable

GridPos = tuple[int, int]

PASSABLE_CELLS = frozenset({"floor", "goal", "key", "lava"})


class MiniGridPortDependencyError(ImportError):
    """Raised when the optional MiniGrid dependency is required but unavailable."""


def _pos_from_value(value: Iterable[int]) -> GridPos:
    row, col = tuple(value)
    return int(row), int(col)


def _pos_to_list(pos: GridPos) -> list[int]:
    return [int(pos[0]), int(pos[1])]


@dataclass
class MiniGridPortEnvironment:
    """Serializable matched-pair substrate that can instantiate a MiniGrid env.

    Coordinates are stored as ``(row, col)`` for consistency with MapShift-2D.
    The optional MiniGrid backend uses ``(x=col, y=row)`` internally.
    """

    environment_id: str
    motif_tag: str
    split_name: str
    seed: int
    width: int
    height: int
    grid: tuple[tuple[str, ...], ...]
    start_pos: GridPos
    agent_dir: int
    token_positions: dict[str, GridPos]
    active_token: str
    movement_cost_scale: float = 1.0
    slip_probability: float = 0.0
    observation_radius: int = 7
    mission: str = "go to the target"
    history: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def goal_pos(self) -> GridPos:
        return self.token_positions[self.active_token]

    def clone(self, environment_id: str | None = None) -> "MiniGridPortEnvironment":
        payload = self.to_dict()
        payload["environment_id"] = environment_id or self.environment_id
        return MiniGridPortEnvironment.from_dict(payload)

    def cell_type(self, pos: GridPos) -> str:
        row, col = pos
        return self.grid[row][col]

    def with_cell(self, pos: GridPos, cell_type: str, *, environment_id: str | None = None) -> "MiniGridPortEnvironment":
        rows = [list(row) for row in self.grid]
        row, col = pos
        rows[row][col] = cell_type
        payload = self.to_dict()
        payload["grid"] = rows
        if environment_id is not None:
            payload["environment_id"] = environment_id
        return MiniGridPortEnvironment.from_dict(payload)

    def passable(self, pos: GridPos) -> bool:
        row, col = pos
        if row < 0 or col < 0 or row >= self.height or col >= self.width:
            return False
        return self.grid[row][col] in PASSABLE_CELLS

    def neighbors(self, pos: GridPos) -> tuple[GridPos, ...]:
        row, col = pos
        candidates = ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))
        return tuple(candidate for candidate in candidates if self.passable(candidate))

    def shortest_path(self, start: GridPos | None = None, goal: GridPos | None = None) -> tuple[GridPos, ...]:
        start_pos = start or self.start_pos
        goal_pos = goal or self.goal_pos
        frontier: deque[GridPos] = deque([start_pos])
        parent: dict[GridPos, GridPos | None] = {start_pos: None}
        while frontier:
            current = frontier.popleft()
            if current == goal_pos:
                break
            for neighbor in self.neighbors(current):
                if neighbor not in parent:
                    parent[neighbor] = current
                    frontier.append(neighbor)
        if goal_pos not in parent:
            return tuple()
        path: list[GridPos] = []
        cursor: GridPos | None = goal_pos
        while cursor is not None:
            path.append(cursor)
            cursor = parent[cursor]
        return tuple(reversed(path))

    def topology_signature(self) -> tuple[str, ...]:
        return tuple("".join("1" if cell == "wall" else "0" for cell in row) for row in self.grid)

    def semantic_signature(self) -> tuple[Any, ...]:
        return (
            self.active_token,
            tuple(sorted((token, pos) for token, pos in self.token_positions.items())),
        )

    def dynamics_signature(self) -> tuple[float, ...]:
        return (round(float(self.slip_probability), 6),)

    def metric_signature(self) -> tuple[float, ...]:
        return (round(float(self.movement_cost_scale), 6),)

    def state_signature(self) -> tuple[Any, ...]:
        return (
            self.topology_signature(),
            self.semantic_signature(),
            self.dynamics_signature(),
            self.metric_signature(),
            self.start_pos,
            self.agent_dir,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment_id": self.environment_id,
            "motif_tag": self.motif_tag,
            "split_name": self.split_name,
            "seed": self.seed,
            "width": self.width,
            "height": self.height,
            "grid": [list(row) for row in self.grid],
            "start_pos": _pos_to_list(self.start_pos),
            "agent_dir": self.agent_dir,
            "token_positions": {token: _pos_to_list(pos) for token, pos in sorted(self.token_positions.items())},
            "active_token": self.active_token,
            "movement_cost_scale": self.movement_cost_scale,
            "slip_probability": self.slip_probability,
            "observation_radius": self.observation_radius,
            "mission": self.mission,
            "history": list(self.history),
            "metadata": deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MiniGridPortEnvironment":
        return cls(
            environment_id=str(payload["environment_id"]),
            motif_tag=str(payload["motif_tag"]),
            split_name=str(payload["split_name"]),
            seed=int(payload["seed"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
            grid=tuple(tuple(str(cell) for cell in row) for row in payload["grid"]),
            start_pos=_pos_from_value(payload["start_pos"]),
            agent_dir=int(payload["agent_dir"]),
            token_positions={str(token): _pos_from_value(pos) for token, pos in payload["token_positions"].items()},
            active_token=str(payload["active_token"]),
            movement_cost_scale=float(payload.get("movement_cost_scale", 1.0)),
            slip_probability=float(payload.get("slip_probability", 0.0)),
            observation_radius=int(payload.get("observation_radius", 7)),
            mission=str(payload.get("mission", "go to the target")),
            history=tuple(str(item) for item in payload.get("history", [])),
            metadata=deepcopy(payload.get("metadata", {})),
        )

    def serialize(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def deserialize(cls, payload: str) -> "MiniGridPortEnvironment":
        return cls.from_dict(json.loads(payload))

    def render_ascii(self) -> str:
        token_by_pos = {pos: token for token, pos in self.token_positions.items()}
        lines: list[str] = []
        for row_index, row in enumerate(self.grid):
            chars: list[str] = []
            for col_index, cell in enumerate(row):
                pos = (row_index, col_index)
                if pos == self.start_pos:
                    chars.append("A")
                elif pos == self.goal_pos:
                    chars.append("G")
                elif pos in token_by_pos:
                    chars.append("K")
                elif cell == "wall":
                    chars.append("#")
                elif cell == "lava":
                    chars.append("~")
                else:
                    chars.append(".")
            lines.append("".join(chars))
        return "\n".join(lines)

    def to_minigrid_env(self, *, max_steps: int | None = None) -> Any:
        """Instantiate an actual MiniGrid environment when the optional package is installed."""

        try:
            from minigrid.core.grid import Grid
            from minigrid.core.mission import MissionSpace
            from minigrid.core.world_object import Goal, Key, Lava, Wall
            from minigrid.minigrid_env import MiniGridEnv
        except ImportError as exc:  # pragma: no cover - exercised only when optional dep is absent.
            raise MiniGridPortDependencyError(
                "MiniGrid is optional. Install with `pip install -e .[minigrid]` to instantiate MiniGrid envs."
            ) from exc

        port_environment = self
        step_limit = max_steps or max(32, 4 * self.width * self.height)

        class StaticMapShiftMiniGridEnv(MiniGridEnv):  # type: ignore[misc, valid-type]
            def __init__(self) -> None:
                mission_space = MissionSpace(mission_func=lambda: port_environment.mission)
                super().__init__(
                    mission_space=mission_space,
                    width=port_environment.width,
                    height=port_environment.height,
                    max_steps=step_limit,
                    see_through_walls=True,
                )
                self.mapshift_port_environment = port_environment
                self._mapshift_last_slip_applied = False

            def _gen_grid(self, width: int, height: int) -> None:
                self.grid = Grid(width, height)
                for row_index, row in enumerate(port_environment.grid):
                    for col_index, cell in enumerate(row):
                        obj = None
                        if cell == "wall":
                            obj = Wall()
                        elif cell == "lava":
                            obj = Lava()
                        self.grid.set(col_index, row_index, obj)
                for token, pos in port_environment.token_positions.items():
                    row_index, col_index = pos
                    if token == port_environment.active_token:
                        self.grid.set(col_index, row_index, Goal())
                    else:
                        self.grid.set(col_index, row_index, Key("blue"))
                start_row, start_col = port_environment.start_pos
                self.agent_pos = (start_col, start_row)
                self.agent_dir = port_environment.agent_dir
                self.mission = port_environment.mission

            def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
                applied_action = action
                self._mapshift_last_slip_applied = False
                if action == self.actions.forward and port_environment.slip_probability > 0.0:
                    if float(self.np_random.random()) < port_environment.slip_probability:
                        applied_action = self.actions.left if float(self.np_random.random()) < 0.5 else self.actions.right
                        self._mapshift_last_slip_applied = True

                obs, reward, terminated, truncated, info = super().step(applied_action)
                step_penalty = max(0.0, port_environment.movement_cost_scale - 1.0) * 0.01
                reward = float(reward) - step_penalty
                info = dict(info)
                info.update(
                    {
                        "mapshift_action_requested": int(action),
                        "mapshift_action_applied": int(applied_action),
                        "mapshift_movement_cost_scale": port_environment.movement_cost_scale,
                        "mapshift_slip_probability": port_environment.slip_probability,
                        "mapshift_slip_applied": self._mapshift_last_slip_applied,
                        "mapshift_step_penalty": step_penalty,
                    }
                )
                return obs, reward, terminated, truncated, info

        return StaticMapShiftMiniGridEnv()
