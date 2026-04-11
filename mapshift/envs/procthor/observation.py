"""Observation helpers for MapShift-3D ProcTHOR scenes."""

from __future__ import annotations

from dataclasses import dataclass, field

from .wrappers import ProcTHORPose, ProcTHORScene


@dataclass(frozen=True)
class ObservationFrame3D:
    frame_size: tuple[int, int]
    visible_objects: tuple[str, ...] = field(default_factory=tuple)
    visible_categories: tuple[str, ...] = field(default_factory=tuple)
    visible_semantic_labels: tuple[str, ...] = field(default_factory=tuple)
    pose_token: str = ""
    room_id: str = ""


def observe_scene(scene: ProcTHORScene, pose: ProcTHORPose | None = None) -> ObservationFrame3D:
    """Return a deterministic egocentric observation summary for one ProcTHOR scene."""

    active_pose = pose or scene.start_pose
    visible = list(scene.objects_in_room(active_pose.room_id))
    for neighbor in scene.neighbors(active_pose.room_id):
        visible.extend(scene.objects_in_room(neighbor)[:1])
    visible = sorted(visible, key=lambda obj: (obj.room_id, obj.object_id))

    return ObservationFrame3D(
        frame_size=tuple(int(value) for value in scene.observation.get("frame_size", (224, 224))),
        visible_objects=tuple(obj.object_id for obj in visible),
        visible_categories=tuple(sorted({obj.object_type for obj in visible})),
        visible_semantic_labels=tuple(sorted({obj.semantic_label for obj in visible if obj.semantic_label})),
        pose_token=f"{active_pose.room_id}@{int(round(active_pose.yaw_deg))}",
        room_id=active_pose.room_id,
    )
