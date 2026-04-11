"""Validation hooks for MapShift-3D scenes."""

from __future__ import annotations

from typing import Any

from .wrappers import ProcTHORScene


def validate_procthor_scene(scene: Any) -> list[str]:
    """Return deterministic validation issues for one ProcTHOR scene."""

    issues: list[str] = []
    if not isinstance(scene, ProcTHORScene):
        return [f"expected ProcTHORScene, got {type(scene).__name__}"]

    if not scene.scene_id:
        issues.append("scene_id is empty")
    if len(scene.rooms) < 2:
        issues.append("scene must contain at least two rooms")
    if len(set(scene.rooms)) != len(scene.rooms):
        issues.append("room ids are not unique")
    if len(set(scene.goal_tokens)) != len(scene.goal_tokens):
        issues.append("goal tokens are not unique")

    object_ids = [obj.object_id for obj in scene.objects]
    if len(set(object_ids)) != len(object_ids):
        issues.append("object ids are not unique")

    if scene.start_pose.room_id not in scene.rooms:
        issues.append("start_pose.room_id is not a valid room")
    if scene.goal_pose.room_id not in scene.rooms:
        issues.append("goal_pose.room_id is not a valid room")

    if len(scene.connected_components()) != 1:
        issues.append("scene connectivity is disconnected")
    if not scene.reachable(scene.start_node_id, scene.goal_node_id):
        issues.append("goal room is not reachable from start room")

    for token, object_id in sorted(scene.goal_tokens.items()):
        if object_id not in object_ids:
            issues.append(f"goal token {token} points to unknown object {object_id}")
    for object_id in scene.landmark_object_ids:
        if object_id not in object_ids:
            issues.append(f"landmark object {object_id} does not exist")

    supported = scene.metadata.get("supported_intervention_families", ())
    if not isinstance(supported, (list, tuple)) or not supported:
        issues.append("supported_intervention_families metadata is missing")

    return issues
