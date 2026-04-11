"""ProcTHOR wrapper types used by MapShift-3D."""

from __future__ import annotations

import importlib.util
import json
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any


def _token_symbol(token: str) -> str:
    suffix = token.split("_", 1)[-1]
    for char in suffix:
        if char.isalpha():
            return char.upper()
    for char in token:
        if char.isalpha():
            return char.upper()
    return "?"


@dataclass(frozen=True)
class ProcTHORWrapperConfig:
    scene_sampler: str
    observation_mode: str
    frame_size: tuple[int, int]
    field_of_view_deg: float
    semantic_channels: bool


@dataclass(frozen=True)
class ProcTHORPose:
    room_id: str
    x_m: float
    z_m: float
    yaw_deg: float = 0.0


@dataclass(frozen=True)
class ProcTHORObject:
    object_id: str
    object_type: str
    room_id: str
    x_m: float
    y_m: float
    z_m: float
    material: str = ""
    semantic_token: str = ""

    @property
    def semantic_label(self) -> str:
        if self.semantic_token:
            return _token_symbol(self.semantic_token)
        return self.object_type[:1].upper()


@dataclass
class ProcTHORScene:
    scene_id: str
    seed: int
    motif_tag: str
    split_name: str
    rooms: tuple[str, ...]
    connectivity: dict[str, tuple[str, ...]]
    objects: tuple[ProcTHORObject, ...]
    landmark_object_ids: tuple[str, ...]
    goal_tokens: dict[str, str]
    start_pose: ProcTHORPose
    goal_pose: ProcTHORPose
    control: dict[str, float]
    dynamics: dict[str, float]
    observation: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def environment_id(self) -> str:
        return self.scene_id

    @property
    def start_node_id(self) -> str:
        return self.start_pose.room_id

    @property
    def goal_node_id(self) -> str:
        return self.goal_pose.room_id

    def clone(self, scene_id: str | None = None) -> "ProcTHORScene":
        payload = self.to_dict()
        payload["scene_id"] = scene_id or self.scene_id
        return ProcTHORScene.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "seed": self.seed,
            "motif_tag": self.motif_tag,
            "split_name": self.split_name,
            "rooms": list(self.rooms),
            "connectivity": {room_id: list(neighbors) for room_id, neighbors in sorted(self.connectivity.items())},
            "objects": [asdict(obj) for obj in self.objects],
            "landmark_object_ids": list(self.landmark_object_ids),
            "goal_tokens": dict(sorted(self.goal_tokens.items())),
            "start_pose": asdict(self.start_pose),
            "goal_pose": asdict(self.goal_pose),
            "control": dict(self.control),
            "dynamics": dict(self.dynamics),
            "observation": dict(self.observation),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProcTHORScene":
        return cls(
            scene_id=str(payload["scene_id"]),
            seed=int(payload["seed"]),
            motif_tag=str(payload["motif_tag"]),
            split_name=str(payload["split_name"]),
            rooms=tuple(str(room_id) for room_id in payload["rooms"]),
            connectivity={
                str(room_id): tuple(str(neighbor) for neighbor in neighbors)
                for room_id, neighbors in payload["connectivity"].items()
            },
            objects=tuple(
                ProcTHORObject(
                    object_id=str(item["object_id"]),
                    object_type=str(item["object_type"]),
                    room_id=str(item["room_id"]),
                    x_m=float(item["x_m"]),
                    y_m=float(item["y_m"]),
                    z_m=float(item["z_m"]),
                    material=str(item.get("material", "")),
                    semantic_token=str(item.get("semantic_token", "")),
                )
                for item in payload["objects"]
            ),
            landmark_object_ids=tuple(str(object_id) for object_id in payload["landmark_object_ids"]),
            goal_tokens={str(token): str(object_id) for token, object_id in payload["goal_tokens"].items()},
            start_pose=ProcTHORPose(**payload["start_pose"]),
            goal_pose=ProcTHORPose(**payload["goal_pose"]),
            control={str(key): float(value) for key, value in payload["control"].items()},
            dynamics={str(key): float(value) for key, value in payload.get("dynamics", {}).items()},
            observation=dict(payload["observation"]),
            metadata=dict(payload.get("metadata", {})),
        )

    def serialize(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def deserialize(cls, payload: str) -> "ProcTHORScene":
        return cls.from_dict(json.loads(payload))

    def neighbors(self, room_id: str) -> tuple[str, ...]:
        return self.connectivity.get(room_id, ())

    def connected_components(self) -> list[set[str]]:
        unseen = set(self.rooms)
        components: list[set[str]] = []
        while unseen:
            seed_room = next(iter(unseen))
            frontier = deque([seed_room])
            component = {seed_room}
            while frontier:
                room_id = frontier.popleft()
                for neighbor in self.neighbors(room_id):
                    if neighbor not in component:
                        component.add(neighbor)
                        frontier.append(neighbor)
            components.append(component)
            unseen -= component
        return components

    def reachable(self, start_room_id: str, goal_room_id: str) -> bool:
        return self.room_path(start_room_id, goal_room_id) is not None

    def room_path(self, start_room_id: str, goal_room_id: str) -> list[str] | None:
        if start_room_id not in self.connectivity or goal_room_id not in self.connectivity:
            return None
        frontier = deque([start_room_id])
        parent: dict[str, str | None] = {start_room_id: None}
        while frontier:
            room_id = frontier.popleft()
            if room_id == goal_room_id:
                break
            for neighbor in self.neighbors(room_id):
                if neighbor not in parent:
                    parent[neighbor] = room_id
                    frontier.append(neighbor)
        if goal_room_id not in parent:
            return None
        path = [goal_room_id]
        cursor = goal_room_id
        while parent[cursor] is not None:
            cursor = str(parent[cursor])
            path.append(cursor)
        path.reverse()
        return path

    def shortest_path(self, start_room_id: str, goal_room_id: str) -> list[str] | None:
        return self.room_path(start_room_id, goal_room_id)

    def shortest_path_length(self, start_room_id: str, goal_room_id: str) -> float | None:
        path = self.room_path(start_room_id, goal_room_id)
        if path is None:
            return None
        return max(0, len(path) - 1) * float(self.control.get("move_step_m", 1.0))

    def room_count(self) -> int:
        return len(self.rooms)

    def object_count(self) -> int:
        return len(self.objects)

    def edge_list(self) -> list[tuple[str, str]]:
        edges: set[tuple[str, str]] = set()
        for room_id, neighbors in self.connectivity.items():
            for neighbor in neighbors:
                edges.add(tuple(sorted((room_id, neighbor))))
        return sorted(edges)

    def edge_count(self) -> int:
        return len(self.edge_list())

    def objects_in_room(self, room_id: str) -> tuple[ProcTHORObject, ...]:
        return tuple(obj for obj in self.objects if obj.room_id == room_id)

    def object_by_id(self, object_id: str) -> ProcTHORObject:
        for obj in self.objects:
            if obj.object_id == object_id:
                return obj
        raise KeyError(f"Unknown ProcTHOR object id: {object_id}")

    def semantic_signature(self) -> tuple[Any, ...]:
        return (
            tuple(sorted((token, object_id) for token, object_id in self.goal_tokens.items())),
            tuple(
                sorted(
                    (
                        obj.object_id,
                        obj.object_type,
                        obj.room_id,
                        obj.material,
                        obj.semantic_token,
                    )
                    for obj in self.objects
                )
            ),
        )

    def structural_signature(self) -> tuple[Any, ...]:
        return (
            tuple(self.rooms),
            tuple(sorted((room_id, tuple(sorted(neighbors))) for room_id, neighbors in self.connectivity.items())),
            tuple(sorted(self.landmark_object_ids)),
        )

    def metric_signature(self) -> tuple[float, ...]:
        return (
            float(self.control.get("move_step_m", 0.0)),
            float(self.observation.get("field_of_view_deg", 0.0)),
        )

    def dynamics_signature(self) -> tuple[float, ...]:
        return (
            float(self.dynamics.get("friction", 1.0)),
            float(self.dynamics.get("inertial_lag", 0.0)),
            float(self.dynamics.get("action_asymmetry", 0.0)),
        )

    def candidate_shortcuts(self) -> list[tuple[str, str, float]]:
        candidates: list[tuple[str, str, float]] = []
        base_step = float(self.control.get("move_step_m", 1.0))
        for left in self.rooms:
            for right in self.rooms:
                if left >= right:
                    continue
                if right in self.connectivity.get(left, ()):
                    continue
                distance = self.shortest_path_length(left, right)
                if distance is None:
                    continue
                detour = distance - base_step
                if detour > base_step:
                    candidates.append((left, right, detour))
        candidates.sort(key=lambda item: item[2], reverse=True)
        return candidates

    def removable_edges(self) -> list[tuple[str, str, float]]:
        candidates: list[tuple[str, str, float]] = []
        original_distance = self.shortest_path_length(self.start_node_id, self.goal_node_id)
        for left, right in self.edge_list():
            self.remove_edge(left, right)
            candidate_distance = self.shortest_path_length(self.start_node_id, self.goal_node_id)
            if candidate_distance is not None:
                delta = 0.0 if original_distance is None else max(0.0, candidate_distance - original_distance)
                candidates.append((left, right, delta))
            self.add_edge(left, right)
        candidates.sort(key=lambda item: item[2], reverse=True)
        return candidates

    def add_edge(self, left: str, right: str) -> None:
        if left == right:
            return
        self.connectivity.setdefault(left, tuple())
        self.connectivity.setdefault(right, tuple())
        if right not in self.connectivity[left]:
            self.connectivity[left] = tuple(sorted(self.connectivity[left] + (right,)))
        if left not in self.connectivity[right]:
            self.connectivity[right] = tuple(sorted(self.connectivity[right] + (left,)))

    def remove_edge(self, left: str, right: str) -> None:
        if right in self.connectivity.get(left, ()):
            self.connectivity[left] = tuple(neighbor for neighbor in self.connectivity[left] if neighbor != right)
        if left in self.connectivity.get(right, ()):
            self.connectivity[right] = tuple(neighbor for neighbor in self.connectivity[right] if neighbor != left)


def optional_backend_status() -> dict[str, Any]:
    """Return import-level status for optional ProcTHOR dependencies."""

    modules = {
        "prior": importlib.util.find_spec("prior") is not None,
        "ai2thor": importlib.util.find_spec("ai2thor") is not None,
        "procthor": importlib.util.find_spec("procthor") is not None,
    }
    available = bool(modules["prior"] and modules["ai2thor"])
    return {
        "available": available,
        "modules": modules,
        "mode": "native_procTHOR" if available else "deterministic_synthetic_fallback",
    }
