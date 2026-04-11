"""Deterministic ProcTHOR-compatible scene sampling for MapShift-3D."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from typing import Any

from mapshift.core.manifests import EnvironmentManifest
from mapshift.core.schemas import Env3DConfig

from .validation import validate_procthor_scene
from .wrappers import ProcTHORObject, ProcTHORPose, ProcTHORScene, ProcTHORWrapperConfig, optional_backend_status


@dataclass(frozen=True)
class ProcTHORGenerationResult:
    manifest: EnvironmentManifest
    scene: ProcTHORScene | None = None


class ProcTHORGenerator:
    """Deterministic 3D generator aligned to the MapShift-3D release contract."""

    _TOKENS = ("target_alpha", "target_beta", "target_gamma")
    _OBJECT_TYPES = ("Chair", "Table", "Sofa", "Lamp", "Painting", "Plant", "Cabinet", "Bed")
    _MATERIALS = ("oak", "walnut", "linen", "steel", "ceramic", "marble")
    _TEMPLATES: dict[str, dict[str, Any]] = {
        "loop_loft": {
            "motif_family": "loop",
            "rooms": ("kitchen", "hall", "office", "bedroom"),
            "edges": (("kitchen", "hall"), ("hall", "office"), ("office", "bedroom"), ("bedroom", "kitchen")),
        },
        "bottleneck_suite": {
            "motif_family": "bottleneck",
            "rooms": ("entry", "gallery", "bottleneck", "study", "bedroom"),
            "edges": (("entry", "gallery"), ("gallery", "bottleneck"), ("bottleneck", "study"), ("bottleneck", "bedroom")),
        },
        "connector_duplex": {
            "motif_family": "connector",
            "rooms": ("living", "connector", "kitchen", "guest", "bath"),
            "edges": (("living", "connector"), ("connector", "kitchen"), ("connector", "guest"), ("guest", "bath")),
        },
        "shortcut_penthouse": {
            "motif_family": "shortcut",
            "rooms": ("entry", "living", "kitchen", "gallery", "roof"),
            "edges": (("entry", "living"), ("living", "kitchen"), ("kitchen", "gallery"), ("gallery", "roof"), ("living", "roof")),
        },
        "room_chain_flat": {
            "motif_family": "room-chain",
            "rooms": ("entry", "living", "kitchen", "study", "bedroom", "storage"),
            "edges": (("entry", "living"), ("living", "kitchen"), ("kitchen", "study"), ("study", "bedroom"), ("bedroom", "storage")),
        },
    }
    _SPLITS = {
        "loop_loft": "train",
        "connector_duplex": "train",
        "room_chain_flat": "val",
        "bottleneck_suite": "test",
        "shortcut_penthouse": "test",
    }

    def __init__(self, config: Env3DConfig | Any) -> None:
        self.config = config
        self.wrapper_config = ProcTHORWrapperConfig(
            scene_sampler=str(config.scene_sampler),
            observation_mode=str(config.observation.mode),
            frame_size=tuple(config.observation.frame_size),
            field_of_view_deg=float(config.observation.field_of_view_deg),
            semantic_channels=bool(config.observation.semantic_channels),
        )
        self.backend_status = optional_backend_status()

    @property
    def motif_tags(self) -> tuple[str, ...]:
        return tuple(self._TEMPLATES)

    def sample(self, seed: int, motif_tag: str | None = None) -> ProcTHORGenerationResult:
        rng = random.Random(seed)
        motif = motif_tag or self.motif_tags[seed % len(self.motif_tags)]
        if motif not in self._TEMPLATES:
            raise KeyError(f"Unsupported ProcTHOR motif tag: {motif}")
        split_name = self._SPLITS.get(motif, "test")
        scene = self._build_scene(seed=seed, motif_tag=motif, split_name=split_name, rng=rng)
        validation_issues = validate_procthor_scene(scene)
        manifest = EnvironmentManifest(
            artifact_id=f"env-artifact-{scene.scene_id}",
            artifact_type="environment",
            benchmark_version="0.1-draft",
            code_version="procthor-deterministic-v1",
            config_hash=self._config_hash(),
            environment_id=scene.scene_id,
            tier="mapshift_3d",
            motif_tags=[motif, str(scene.metadata.get("motif_family", ""))],
            split_name=split_name,
            seed_values=[seed],
            metadata={
                "platform": "ProcTHOR",
                "scene_sampler": self.config.scene_sampler,
                "motif_tag": motif,
                "motif_family": scene.metadata.get("motif_family", ""),
                "room_count": scene.room_count(),
                "object_count": scene.object_count(),
                "goal_tokens": dict(scene.goal_tokens),
                "landmark_object_ids": list(scene.landmark_object_ids),
                "structural_signature": list(scene.structural_signature()),
                "semantic_signature": list(scene.semantic_signature()),
                "backend_status": dict(self.backend_status),
                "validation_issues": list(validation_issues),
                "serialized_scene": scene.to_dict(),
            },
        )
        return ProcTHORGenerationResult(manifest=manifest, scene=scene)

    def replay_from_manifest(self, manifest: EnvironmentManifest) -> ProcTHORScene:
        serialized = manifest.metadata.get("serialized_scene")
        if not isinstance(serialized, dict):
            raise ValueError("ProcTHOR manifest metadata does not contain serialized_scene")
        return ProcTHORScene.from_dict(serialized)

    def _config_hash(self) -> str:
        payload = json.dumps(asdict(self.config), sort_keys=True).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()[:12]

    def _build_scene(self, seed: int, motif_tag: str, split_name: str, rng: random.Random) -> ProcTHORScene:
        template = self._TEMPLATES[motif_tag]
        rooms = tuple(template["rooms"])
        connectivity = {room_id: [] for room_id in rooms}
        for left, right in template["edges"]:
            connectivity[left].append(right)
            connectivity[right].append(left)
        connectivity = {room_id: tuple(sorted(neighbors)) for room_id, neighbors in connectivity.items()}

        objects: list[ProcTHORObject] = []
        landmark_object_ids: list[str] = []
        room_positions = self._room_positions(rooms)
        for room_index, room_id in enumerate(rooms):
            base_x, base_z = room_positions[room_id]
            object_count = 2 + (room_index + seed) % 2
            for index in range(object_count):
                object_id = f"{room_id}_obj_{index}"
                object_type = self._OBJECT_TYPES[(room_index + index) % len(self._OBJECT_TYPES)]
                material = self._MATERIALS[(seed + room_index + index) % len(self._MATERIALS)]
                objects.append(
                    ProcTHORObject(
                        object_id=object_id,
                        object_type=object_type,
                        room_id=room_id,
                        x_m=base_x + 0.4 * (index + 1),
                        y_m=0.0,
                        z_m=base_z + 0.35 * ((room_index + index) % 3),
                        material=material,
                    )
                )
            landmark_object_ids.append(f"{room_id}_obj_0")

        start_room, goal_room = self._farthest_rooms(rooms, connectivity)
        goal_tokens = {
            token: self._token_object_id(objects, rooms[(index + seed) % len(rooms)], fallback_room=goal_room)
            for index, token in enumerate(self._TOKENS)
        }
        goal_object_id = goal_tokens[self._TOKENS[0]]
        goal_room = next(obj.room_id for obj in objects if obj.object_id == goal_object_id)
        objects = [
            ProcTHORObject(
                object_id=obj.object_id,
                object_type=obj.object_type,
                room_id=obj.room_id,
                x_m=obj.x_m,
                y_m=obj.y_m,
                z_m=obj.z_m,
                material=obj.material,
                semantic_token=next((token for token, object_id in goal_tokens.items() if object_id == obj.object_id), ""),
            )
            for obj in objects
        ]

        return ProcTHORScene(
            scene_id=f"{motif_tag}-seed{seed}",
            seed=seed,
            motif_tag=motif_tag,
            split_name=split_name,
            rooms=rooms,
            connectivity=connectivity,
            objects=tuple(objects),
            landmark_object_ids=tuple(landmark_object_ids),
            goal_tokens=goal_tokens,
            start_pose=ProcTHORPose(room_id=start_room, x_m=room_positions[start_room][0] + 0.5, z_m=room_positions[start_room][1] + 0.5),
            goal_pose=ProcTHORPose(room_id=goal_room, x_m=room_positions[goal_room][0] + 1.0, z_m=room_positions[goal_room][1] + 0.8),
            control={
                "move_step_m": float(self.config.control.move_step_m),
                "rotate_step_deg": float(self.config.control.rotate_step_deg),
            },
            dynamics={
                "friction": 1.0,
                "inertial_lag": 0.0,
                "action_asymmetry": 0.0,
            },
            observation={
                "mode": self.config.observation.mode,
                "frame_size": list(self.config.observation.frame_size),
                "field_of_view_deg": self.config.observation.field_of_view_deg,
                "semantic_channels": self.config.observation.semantic_channels,
            },
            metadata={
                "motif_family": template["motif_family"],
                "supported_intervention_families": list(self.config.supported_intervention_families),
                "documented_deviations": list(self.config.documented_deviations),
                "backend_mode": self.backend_status["mode"],
            },
        )

    def _room_positions(self, rooms: tuple[str, ...]) -> dict[str, tuple[float, float]]:
        positions: dict[str, tuple[float, float]] = {}
        for index, room_id in enumerate(rooms):
            positions[room_id] = (float((index % 3) * 3), float((index // 3) * 3))
        return positions

    def _token_object_id(self, objects: list[ProcTHORObject], preferred_room: str, fallback_room: str) -> str:
        for obj in objects:
            if obj.room_id == preferred_room:
                return obj.object_id
        for obj in objects:
            if obj.room_id == fallback_room:
                return obj.object_id
        return objects[0].object_id

    def _farthest_rooms(self, rooms: tuple[str, ...], connectivity: dict[str, tuple[str, ...]]) -> tuple[str, str]:
        def room_distance(start_room: str, goal_room: str) -> int:
            frontier = [(start_room, 0)]
            seen = {start_room}
            for room_id, distance in frontier:
                if room_id == goal_room:
                    return distance
                for neighbor in connectivity.get(room_id, ()):
                    if neighbor not in seen:
                        seen.add(neighbor)
                        frontier.append((neighbor, distance + 1))
            return 0

        best_pair = (rooms[0], rooms[-1])
        best_distance = -1
        for left in rooms:
            for right in rooms:
                if left == right:
                    continue
                distance = room_distance(left, right)
                if distance > best_distance:
                    best_pair = (left, right)
                    best_distance = distance
        return best_pair
