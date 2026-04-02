"""State and occupancy-grid environment containers for MapShift-2D."""

from __future__ import annotations

import json
from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from .dynamics import DynamicsParameters2D

Cell = tuple[int, int]


def _edge_key(left: str, right: str) -> str:
    return "|".join(sorted((left, right)))


def _cell_from_value(value: Cell | list[int] | tuple[int, int]) -> Cell:
    return int(value[0]), int(value[1])


def _cells_to_lists(cells: Iterable[Cell]) -> list[list[int]]:
    return [[row, col] for row, col in cells]


def _manhattan(left: Cell, right: Cell) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def _token_symbol(token: str) -> str:
    suffix = token.split("_", 1)[-1]
    for char in suffix:
        if char.isalpha():
            return char.upper()
    for char in token:
        if char.isalpha():
            return char.upper()
    return "?"


def _orthogonal_path(start: Cell, goal: Cell, prefer_horizontal: bool) -> list[Cell]:
    """Return a deterministic L-shaped path between two cells."""

    row, col = start
    goal_row, goal_col = goal
    path = [(row, col)]

    def walk_row(target_row: int) -> None:
        nonlocal row
        step = 1 if target_row >= row else -1
        for new_row in range(row + step, target_row + step, step):
            row = new_row
            path.append((row, col))

    def walk_col(target_col: int) -> None:
        nonlocal col
        step = 1 if target_col >= col else -1
        for new_col in range(col + step, target_col + step, step):
            col = new_col
            path.append((row, col))

    if prefer_horizontal:
        walk_col(goal_col)
        walk_row(goal_row)
    else:
        walk_row(goal_row)
        walk_col(goal_col)

    unique: list[Cell] = []
    for cell in path:
        if not unique or unique[-1] != cell:
            unique.append(cell)
    return unique


@dataclass(frozen=True)
class AgentPose2D:
    x: float
    y: float
    theta_deg: float


@dataclass(frozen=True)
class Map2DState:
    pose: AgentPose2D
    timestep: int = 0
    semantic_tags: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Map2DNode:
    node_id: str
    row: int
    col: int
    landmark: str = ""

    @property
    def cell(self) -> Cell:
        return self.row, self.col


@dataclass
class Map2DEnvironment:
    """Occupancy-grid benchmark environment with anchor nodes and replayable state."""

    environment_id: str
    motif_tag: str
    split_name: str
    seed: int
    width_cells: int
    height_cells: int
    occupancy_grid: list[list[int]]
    nodes: dict[str, Map2DNode]
    adjacency: dict[str, list[str]]
    room_cells: dict[str, list[Cell]]
    edge_corridors: dict[str, list[Cell]]
    start_node_id: str
    goal_node_id: str
    landmark_by_node: dict[str, str]
    goal_tokens: dict[str, str]
    dynamics: DynamicsParameters2D
    occupancy_resolution_m: float
    geometry_scale: float = 1.0
    observation_radius_m: float = 0.0
    field_of_view_deg: float = 0.0
    semantic_channels: bool = True
    agent_pose: AgentPose2D | None = None
    history: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def clone(self, environment_id: str | None = None) -> "Map2DEnvironment":
        """Return a detached environment copy."""

        return Map2DEnvironment.from_dict(
            {
                **self.to_dict(),
                "environment_id": environment_id or self.environment_id,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the environment into a deterministic dictionary."""

        return {
            "environment_id": self.environment_id,
            "motif_tag": self.motif_tag,
            "split_name": self.split_name,
            "seed": self.seed,
            "width_cells": self.width_cells,
            "height_cells": self.height_cells,
            "occupancy_grid": [list(row) for row in self.occupancy_grid],
            "nodes": {
                node_id: {
                    "node_id": node.node_id,
                    "row": node.row,
                    "col": node.col,
                    "landmark": node.landmark,
                }
                for node_id, node in sorted(self.nodes.items())
            },
            "adjacency": {node_id: sorted(neighbors) for node_id, neighbors in sorted(self.adjacency.items())},
            "room_cells": {node_id: _cells_to_lists(cells) for node_id, cells in sorted(self.room_cells.items())},
            "edge_corridors": {edge_key: _cells_to_lists(cells) for edge_key, cells in sorted(self.edge_corridors.items())},
            "start_node_id": self.start_node_id,
            "goal_node_id": self.goal_node_id,
            "landmark_by_node": dict(sorted(self.landmark_by_node.items())),
            "goal_tokens": dict(sorted(self.goal_tokens.items())),
            "dynamics": asdict(self.dynamics),
            "occupancy_resolution_m": self.occupancy_resolution_m,
            "geometry_scale": self.geometry_scale,
            "observation_radius_m": self.observation_radius_m,
            "field_of_view_deg": self.field_of_view_deg,
            "semantic_channels": self.semantic_channels,
            "agent_pose": None
            if self.agent_pose is None
            else {"x": self.agent_pose.x, "y": self.agent_pose.y, "theta_deg": self.agent_pose.theta_deg},
            "history": list(self.history),
            "metadata": deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Map2DEnvironment":
        """Deserialize an environment from a dictionary payload."""

        nodes = {
            node_id: Map2DNode(
                node_id=node_payload["node_id"],
                row=int(node_payload["row"]),
                col=int(node_payload["col"]),
                landmark=str(node_payload.get("landmark", "")),
            )
            for node_id, node_payload in payload["nodes"].items()
        }
        occupancy_grid = [[int(cell) for cell in row] for row in payload["occupancy_grid"]]
        agent_pose_payload = payload.get("agent_pose")
        agent_pose = None
        if agent_pose_payload is not None:
            agent_pose = AgentPose2D(
                x=float(agent_pose_payload["x"]),
                y=float(agent_pose_payload["y"]),
                theta_deg=float(agent_pose_payload["theta_deg"]),
            )

        return cls(
            environment_id=str(payload["environment_id"]),
            motif_tag=str(payload["motif_tag"]),
            split_name=str(payload["split_name"]),
            seed=int(payload["seed"]),
            width_cells=int(payload["width_cells"]),
            height_cells=int(payload["height_cells"]),
            occupancy_grid=occupancy_grid,
            nodes=nodes,
            adjacency={node_id: [str(neighbor) for neighbor in neighbors] for node_id, neighbors in payload["adjacency"].items()},
            room_cells={node_id: [_cell_from_value(cell) for cell in cells] for node_id, cells in payload["room_cells"].items()},
            edge_corridors={edge_key: [_cell_from_value(cell) for cell in cells] for edge_key, cells in payload["edge_corridors"].items()},
            start_node_id=str(payload["start_node_id"]),
            goal_node_id=str(payload["goal_node_id"]),
            landmark_by_node={str(node_id): str(label) for node_id, label in payload["landmark_by_node"].items()},
            goal_tokens={str(token): str(node_id) for token, node_id in payload["goal_tokens"].items()},
            dynamics=DynamicsParameters2D(**payload["dynamics"]),
            occupancy_resolution_m=float(payload["occupancy_resolution_m"]),
            geometry_scale=float(payload.get("geometry_scale", 1.0)),
            observation_radius_m=float(payload.get("observation_radius_m", 0.0)),
            field_of_view_deg=float(payload.get("field_of_view_deg", 0.0)),
            semantic_channels=bool(payload.get("semantic_channels", True)),
            agent_pose=agent_pose,
            history=[str(item) for item in payload.get("history", [])],
            metadata=deepcopy(payload.get("metadata", {})),
        )

    def serialize(self) -> str:
        """Serialize the environment to deterministic JSON."""

        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def deserialize(cls, payload: str) -> "Map2DEnvironment":
        """Deserialize an environment from deterministic JSON."""

        return cls.from_dict(json.loads(payload))

    @classmethod
    def from_manifest_metadata(cls, metadata: dict[str, Any]) -> "Map2DEnvironment":
        """Replay an environment from manifest metadata."""

        serialized = metadata.get("serialized_environment")
        if not isinstance(serialized, dict):
            raise ValueError("Manifest metadata does not contain a serialized_environment payload.")
        return cls.from_dict(serialized)

    def occupancy_signature(self) -> tuple[str, ...]:
        return tuple("".join(str(cell) for cell in row) for row in self.occupancy_grid)

    def geometry_signature(self) -> tuple[Any, ...]:
        return (
            self.width_cells,
            self.height_cells,
            self.occupancy_signature(),
            tuple(sorted((node_id, node.row, node.col) for node_id, node in self.nodes.items())),
            tuple(self.edge_list()),
            round(self.geometry_scale, 6),
            round(self.occupancy_resolution_m, 6),
        )

    def semantic_signature(self) -> tuple[Any, ...]:
        return (
            tuple(sorted(self.landmark_by_node.items())),
            tuple(sorted(self.goal_tokens.items())),
        )

    def dynamics_signature(self) -> tuple[float, ...]:
        return (
            self.dynamics.forward_gain,
            self.dynamics.turn_gain,
            self.dynamics.friction,
            self.dynamics.inertial_lag,
            self.dynamics.action_asymmetry,
            self.dynamics.odometry_bias_deg,
        )

    @property
    def start_cell(self) -> Cell:
        return self.resolve_location(self.start_node_id)

    @property
    def goal_cell(self) -> Cell:
        return self.resolve_location(self.goal_node_id)

    def node_count(self) -> int:
        return len(self.nodes)

    def edge_list(self) -> list[tuple[str, str]]:
        edges: set[tuple[str, str]] = set()
        for left, neighbors in self.adjacency.items():
            for right in neighbors:
                edges.add(tuple(sorted((left, right))))
        return sorted(edges)

    def edge_count(self) -> int:
        return len(self.edge_list())

    def free_cell_count(self) -> int:
        return sum(1 for row in self.occupancy_grid for cell in row if cell == 0)

    def corridor_for_edge(self, left: str, right: str) -> tuple[Cell, ...]:
        return tuple(self.edge_corridors.get(_edge_key(left, right), ()))

    def corridor_is_traversable(self, corridor: Iterable[Cell]) -> bool:
        cells = tuple(corridor)
        return bool(cells) and all(self.is_free(cell) for cell in cells)

    def in_bounds(self, cell: Cell) -> bool:
        row, col = cell
        return 0 <= row < self.height_cells and 0 <= col < self.width_cells

    def is_free(self, cell: Cell) -> bool:
        if not self.in_bounds(cell):
            return False
        row, col = cell
        return self.occupancy_grid[row][col] == 0

    def resolve_location(self, location: str | Cell | AgentPose2D) -> Cell:
        if isinstance(location, str):
            if location not in self.nodes:
                raise KeyError(f"Unknown node id: {location}")
            return self.nodes[location].cell
        if isinstance(location, AgentPose2D):
            return int(round(location.y)), int(round(location.x))
        return int(location[0]), int(location[1])

    def neighbors(self, location: str | Cell) -> tuple[Cell, ...]:
        row, col = self.resolve_location(location)
        candidates = ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))
        return tuple(cell for cell in candidates if self.is_free(cell))

    def shortest_path(self, start: str | Cell | AgentPose2D, goal: str | Cell | AgentPose2D) -> list[Cell] | None:
        start_cell = self.resolve_location(start)
        goal_cell = self.resolve_location(goal)
        if not self.is_free(start_cell) or not self.is_free(goal_cell):
            return None
        if start_cell == goal_cell:
            return [start_cell]

        frontier = deque([start_cell])
        parent: dict[Cell, Cell | None] = {start_cell: None}
        while frontier:
            current = frontier.popleft()
            if current == goal_cell:
                break
            for neighbor in self.neighbors(current):
                if neighbor not in parent:
                    parent[neighbor] = current
                    frontier.append(neighbor)

        if goal_cell not in parent:
            return None

        path: list[Cell] = [goal_cell]
        cursor: Cell | None = goal_cell
        while cursor is not None:
            cursor = parent[cursor]
            if cursor is not None:
                path.append(cursor)
        path.reverse()
        return path

    def oracle_shortest_path(self, start: str | Cell | AgentPose2D, goal: str | Cell | AgentPose2D) -> list[Cell] | None:
        """Alias for deterministic shortest-path computation."""

        return self.shortest_path(start, goal)

    def shortest_path_length(self, start: str | Cell | AgentPose2D, goal: str | Cell | AgentPose2D) -> float | None:
        path = self.shortest_path(start, goal)
        if path is None:
            return None
        return max(0, len(path) - 1) * self.occupancy_resolution_m * self.geometry_scale

    def reachable(self, start: str | Cell | AgentPose2D, goal: str | Cell | AgentPose2D) -> bool:
        return self.shortest_path(start, goal) is not None

    def reachable_set(self, start: str | Cell | AgentPose2D) -> set[Cell]:
        start_cell = self.resolve_location(start)
        if not self.is_free(start_cell):
            return set()

        frontier = deque([start_cell])
        seen = {start_cell}
        while frontier:
            current = frontier.popleft()
            for neighbor in self.neighbors(current):
                if neighbor not in seen:
                    seen.add(neighbor)
                    frontier.append(neighbor)
        return seen

    def connected_component(self, start: str | Cell | AgentPose2D) -> set[str] | set[Cell]:
        if isinstance(start, str):
            return {node_id for node_id in self.nodes if self.reachable(start, node_id)}
        return self.reachable_set(start)

    def connected_components(self) -> list[set[str]]:
        unseen = set(self.nodes)
        components: list[set[str]] = []
        while unseen:
            seed = next(iter(unseen))
            component = {node_id for node_id in self.nodes if self.reachable(seed, node_id)}
            components.append(component)
            unseen -= component
        return components

    def farthest_pair(self) -> tuple[str, str]:
        best_pair = (self.start_node_id, self.goal_node_id)
        best_distance = -1.0
        node_ids = sorted(self.nodes)
        for index, left in enumerate(node_ids):
            for right in node_ids[index + 1 :]:
                distance = self.shortest_path_length(left, right)
                if distance is not None and distance > best_distance:
                    best_pair = (left, right)
                    best_distance = distance
        return best_pair

    def candidate_shortcuts(self) -> list[tuple[str, str, float]]:
        candidates: list[tuple[str, str, float]] = []
        node_ids = sorted(self.nodes)
        for index, left in enumerate(node_ids):
            for right in node_ids[index + 1 :]:
                if right in self.adjacency.get(left, []):
                    continue
                path_distance = self.shortest_path_length(left, right)
                if path_distance is None:
                    continue
                direct_steps = _manhattan(self.resolve_location(left), self.resolve_location(right))
                direct_distance = direct_steps * self.occupancy_resolution_m * self.geometry_scale
                detour = path_distance - direct_distance
                if detour > max(2.0 * self.occupancy_resolution_m * self.geometry_scale, direct_distance * 0.5):
                    candidates.append((left, right, detour))
        candidates.sort(key=lambda item: item[2], reverse=True)
        return candidates

    def removable_edges(self) -> list[tuple[str, str, float]]:
        candidates: list[tuple[str, str, float]] = []
        original_distance = self.shortest_path_length(self.start_node_id, self.goal_node_id)

        for left, right in self.edge_list():
            key = _edge_key(left, right)
            corridor = list(self.edge_corridors.get(key, ()))
            self.remove_edge(left, right)
            new_distance = self.shortest_path_length(self.start_node_id, self.goal_node_id)
            if new_distance is not None:
                path_delta = 0.0 if original_distance is None else max(0.0, new_distance - original_distance)
                candidates.append((left, right, path_delta))
            self._restore_edge(left, right, corridor)

        candidates.sort(key=lambda item: item[2], reverse=True)
        return candidates

    def add_edge(self, left: str, right: str) -> None:
        if left == right:
            return
        key = _edge_key(left, right)
        if key in self.edge_corridors:
            return
        corridor = self._build_corridor_between_nodes(left, right)
        self.edge_corridors[key] = corridor
        self.adjacency.setdefault(left, [])
        self.adjacency.setdefault(right, [])
        if right not in self.adjacency[left]:
            self.adjacency[left].append(right)
        if left not in self.adjacency[right]:
            self.adjacency[right].append(left)
        self.adjacency[left].sort()
        self.adjacency[right].sort()
        self.rebuild_occupancy()

    def remove_edge(self, left: str, right: str) -> None:
        key = _edge_key(left, right)
        self.edge_corridors.pop(key, None)
        if right in self.adjacency.get(left, []):
            self.adjacency[left].remove(right)
        if left in self.adjacency.get(right, []):
            self.adjacency[right].remove(left)
        self.rebuild_occupancy()

    def rebuild_occupancy(self) -> None:
        self.occupancy_grid = [[1 for _ in range(self.width_cells)] for _ in range(self.height_cells)]
        for cells in self.room_cells.values():
            for row, col in cells:
                if self.in_bounds((row, col)):
                    self.occupancy_grid[row][col] = 0
        for cells in self.edge_corridors.values():
            for row, col in cells:
                if self.in_bounds((row, col)):
                    self.occupancy_grid[row][col] = 0

        if self.agent_pose is None:
            start_row, start_col = self.start_cell
            self.agent_pose = AgentPose2D(x=float(start_col), y=float(start_row), theta_deg=0.0)

    def semantic_label_for_cell(self, cell: Cell) -> str:
        for node_id, node in self.nodes.items():
            if node.cell != cell:
                continue
            if node_id == self.start_node_id:
                return "S"
            if node_id == self.goal_node_id:
                return "G"
            if node_id in self.landmark_by_node:
                return self.landmark_by_node[node_id][:1].upper()
            for token, goal_node in self.goal_tokens.items():
                if goal_node == node_id:
                    return _token_symbol(token)
        return ""

    def _restore_edge(self, left: str, right: str, corridor: list[Cell]) -> None:
        key = _edge_key(left, right)
        self.edge_corridors[key] = list(corridor)
        self.adjacency.setdefault(left, [])
        self.adjacency.setdefault(right, [])
        if right not in self.adjacency[left]:
            self.adjacency[left].append(right)
        if left not in self.adjacency[right]:
            self.adjacency[right].append(left)
        self.adjacency[left].sort()
        self.adjacency[right].sort()
        self.rebuild_occupancy()

    def _build_corridor_between_nodes(self, left: str, right: str) -> list[Cell]:
        start = self.resolve_location(left)
        goal = self.resolve_location(right)
        key = _edge_key(left, right)
        prefer_horizontal = (sum(ord(char) for char in key) + self.seed) % 2 == 0
        corridor = _orthogonal_path(start, goal, prefer_horizontal=prefer_horizontal)
        return corridor
