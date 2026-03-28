"""Structural, semantic, and task-template tagging for MapShift splits."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from mapshift.envs.map2d.state import Map2DEnvironment

TRAIN_MOTIFS = ("simple_loop", "two_room_connector", "branching_chain")
VAL_MOTIFS = ("asymmetric_multi_room_chain", "offset_bottleneck")
TEST_MOTIFS = ("nested_bottleneck", "deceptive_shortcut", "disconnected_subregion")

CANONICAL_MOTIF_TAGS = (
    "loop",
    "bottleneck",
    "connector",
    "shortcut",
    "room-chain",
    "disconnected",
    "nearly-disconnected",
)


def stable_template_hash(payload: Any) -> str:
    """Return a deterministic short hash for a JSON-serializable payload."""

    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


def _distance_in_steps(environment: Map2DEnvironment, left: str, right: str) -> int:
    distance = environment.shortest_path_length(left, right)
    if distance is None:
        return -1
    scale = max(environment.occupancy_resolution_m * environment.geometry_scale, 1e-9)
    return int(round(distance / scale))


def _path_length_profile(environment: Map2DEnvironment) -> tuple[int, int, float]:
    node_ids = sorted(environment.nodes)
    distances: list[int] = []
    for index, left in enumerate(node_ids):
        for right in node_ids[index + 1 :]:
            steps = _distance_in_steps(environment, left, right)
            if steps >= 0:
                distances.append(steps)
    if not distances:
        return (0, 0, 0.0)
    return (min(distances), max(distances), round(sum(distances) / len(distances), 4))


def _degree_sequence(environment: Map2DEnvironment) -> tuple[int, ...]:
    return tuple(sorted(len(environment.adjacency.get(node_id, [])) for node_id in environment.nodes))


def _component_sizes(environment: Map2DEnvironment) -> tuple[int, ...]:
    return tuple(sorted(len(component) for component in environment.connected_components()))


def _cycle_rank(environment: Map2DEnvironment) -> int:
    return environment.edge_count() - environment.node_count() + len(environment.connected_components())


def _looks_like_room_chain(environment: Map2DEnvironment) -> bool:
    degrees = _degree_sequence(environment)
    high_degree_nodes = sum(1 for degree in degrees if degree >= 3)
    leaf_count = sum(1 for degree in degrees if degree == 1)
    return leaf_count >= 2 and high_degree_nodes <= max(1, environment.node_count() // 3)


def motif_tags_for_environment(environment: Map2DEnvironment) -> tuple[str, ...]:
    """Return stable structural motif tags for one environment."""

    cached = environment.metadata.get("motif_tags")
    if isinstance(cached, (list, tuple)) and cached:
        return tuple(str(tag) for tag in cached)

    tags: set[str] = set()
    motif_family = str(environment.metadata.get("motif_family", "")).strip()
    if motif_family:
        tags.add(motif_family)

    components = environment.connected_components()
    if len(components) > 1:
        tags.add("disconnected")

    positive_bottlenecks = [delta for _left, _right, delta in environment.removable_edges() if delta > 0.0]
    if positive_bottlenecks:
        tags.add("bottleneck")
    if len(components) > 1 or positive_bottlenecks:
        tags.add("connector")
    if len(components) > 1 or any(delta >= 4.0 for delta in positive_bottlenecks):
        tags.add("nearly-disconnected")

    if _cycle_rank(environment) > 0:
        tags.add("loop")
    if environment.candidate_shortcuts():
        tags.add("shortcut")
    if motif_family == "room-chain" or _looks_like_room_chain(environment):
        tags.add("room-chain")

    ordered = [tag for tag in CANONICAL_MOTIF_TAGS if tag in tags]
    extras = sorted(tag for tag in tags if tag not in CANONICAL_MOTIF_TAGS)
    resolved = tuple(ordered + extras)
    environment.metadata["motif_tags"] = list(resolved)
    return resolved


@dataclass(frozen=True)
class StructuralSignature:
    """Stable structural fingerprint used for split leakage checks."""

    primary_motif_family: str
    motif_tags: tuple[str, ...]
    component_sizes: tuple[int, ...]
    degree_sequence: tuple[int, ...]
    cycle_rank: int
    bottleneck_edge_count: int
    bottleneck_max_delta: float
    path_length_profile: tuple[int, int, float]
    connectivity_hash: str
    geometry_hash: str
    normalized_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def structural_signature_for_environment(environment: Map2DEnvironment) -> StructuralSignature:
    """Build a canonical structural signature for one environment."""

    cached = environment.metadata.get("structural_signature")
    if isinstance(cached, dict) and "normalized_fingerprint" in cached:
        return StructuralSignature(
            primary_motif_family=str(cached.get("primary_motif_family", "")),
            motif_tags=tuple(cached.get("motif_tags", ())),
            component_sizes=tuple(int(value) for value in cached.get("component_sizes", ())),
            degree_sequence=tuple(int(value) for value in cached.get("degree_sequence", ())),
            cycle_rank=int(cached.get("cycle_rank", 0)),
            bottleneck_edge_count=int(cached.get("bottleneck_edge_count", 0)),
            bottleneck_max_delta=float(cached.get("bottleneck_max_delta", 0.0)),
            path_length_profile=tuple(cached.get("path_length_profile", (0, 0, 0.0))),  # type: ignore[arg-type]
            connectivity_hash=str(cached.get("connectivity_hash", "")),
            geometry_hash=str(cached.get("geometry_hash", "")),
            normalized_fingerprint=str(cached.get("normalized_fingerprint", "")),
        )

    motif_tags = motif_tags_for_environment(environment)
    positive_bottlenecks = [delta for _left, _right, delta in environment.removable_edges() if delta > 0.0]
    connectivity_payload = {
        "component_sizes": _component_sizes(environment),
        "degree_sequence": _degree_sequence(environment),
        "cycle_rank": _cycle_rank(environment),
        "path_length_profile": _path_length_profile(environment),
        "motif_tags": motif_tags,
    }
    geometry_payload = {
        "occupancy": environment.occupancy_signature(),
        "node_cells": tuple(sorted((node_id, node.row, node.col) for node_id, node in environment.nodes.items())),
        "edges": tuple(environment.edge_list()),
    }
    normalized_payload = {
        "motif_tags": motif_tags,
        "component_sizes": _component_sizes(environment),
        "degree_sequence": _degree_sequence(environment),
        "cycle_rank": _cycle_rank(environment),
        "bottleneck_edge_count": len(positive_bottlenecks),
        "bottleneck_max_delta": round(max(positive_bottlenecks), 4) if positive_bottlenecks else 0.0,
        "path_length_profile": _path_length_profile(environment),
        "free_space_ratio": round(environment.free_cell_count() / max(1, environment.width_cells * environment.height_cells), 4),
    }
    signature = StructuralSignature(
        primary_motif_family=str(environment.metadata.get("motif_family", "")),
        motif_tags=motif_tags,
        component_sizes=_component_sizes(environment),
        degree_sequence=_degree_sequence(environment),
        cycle_rank=_cycle_rank(environment),
        bottleneck_edge_count=len(positive_bottlenecks),
        bottleneck_max_delta=round(max(positive_bottlenecks), 4) if positive_bottlenecks else 0.0,
        path_length_profile=_path_length_profile(environment),
        connectivity_hash=stable_template_hash(connectivity_payload),
        geometry_hash=stable_template_hash(geometry_payload),
        normalized_fingerprint=stable_template_hash(normalized_payload),
    )
    environment.metadata["structural_signature"] = signature.to_dict()
    return signature


def node_role_template_id(environment: Map2DEnvironment, node_id: str) -> str:
    """Return a structural role identifier for one node."""

    cache = environment.metadata.setdefault("node_role_template_ids", {})
    if isinstance(cache, dict) and node_id in cache:
        return str(cache[node_id])

    distances = tuple(
        _distance_in_steps(environment, node_id, other)
        for other in sorted(environment.nodes)
        if other != node_id
    )
    payload = {
        "degree": len(environment.adjacency.get(node_id, [])),
        "distances": distances,
        "component_size": len(environment.connected_component(node_id)),
    }
    role_id = stable_template_hash(payload)
    if isinstance(cache, dict):
        cache[node_id] = role_id
    return role_id


def semantic_template_metadata(environment: Map2DEnvironment) -> dict[str, Any]:
    """Return canonical semantic-template identifiers for one environment."""

    if all(key in environment.metadata for key in ("landmark_layout_template_id", "goal_token_template_id", "semantic_template_id")):
        return {
            "landmark_layout_template_id": environment.metadata["landmark_layout_template_id"],
            "goal_token_template_id": environment.metadata["goal_token_template_id"],
            "semantic_template_id": environment.metadata["semantic_template_id"],
            "landmark_layout_template": list(environment.metadata.get("landmark_layout_template", [])),
            "goal_token_template": list(environment.metadata.get("goal_token_template", [])),
        }

    landmark_layout = tuple(
        sorted((node_role_template_id(environment, node_id), label) for node_id, label in environment.landmark_by_node.items())
    )
    goal_token_layout = tuple(
        sorted((token, node_role_template_id(environment, node_id)) for token, node_id in environment.goal_tokens.items())
    )
    landmark_layout_template_id = stable_template_hash({"landmarks": landmark_layout})
    goal_token_template_id = stable_template_hash({"goal_tokens": goal_token_layout})
    semantic_template_id = stable_template_hash(
        {
            "landmark_layout_template_id": landmark_layout_template_id,
            "goal_token_template_id": goal_token_template_id,
        }
    )
    metadata = {
        "landmark_layout_template_id": landmark_layout_template_id,
        "goal_token_template_id": goal_token_template_id,
        "semantic_template_id": semantic_template_id,
        "landmark_layout_template": list(landmark_layout),
        "goal_token_template": list(goal_token_layout),
    }
    environment.metadata.update(metadata)
    return metadata


def start_goal_template_metadata(
    environment: Map2DEnvironment,
    start_node_id: str,
    goal_node_id: str | None,
) -> dict[str, Any]:
    """Return a start/goal template identifier for task leakage checks."""

    if goal_node_id is None:
        return {
            "start_goal_template_id": stable_template_hash({"start": node_role_template_id(environment, start_node_id), "goal": None}),
            "start_role_template_id": node_role_template_id(environment, start_node_id),
            "goal_role_template_id": "",
            "distance_steps": -1,
        }

    distance_steps = _distance_in_steps(environment, start_node_id, goal_node_id)
    start_role = node_role_template_id(environment, start_node_id)
    goal_role = node_role_template_id(environment, goal_node_id)
    payload = {
        "start_role": start_role,
        "goal_role": goal_role,
        "distance_steps": distance_steps,
    }
    return {
        "start_goal_template_id": stable_template_hash(payload),
        "start_role_template_id": start_role,
        "goal_role_template_id": goal_role,
        "distance_steps": distance_steps,
    }


def task_template_metadata(
    environment: Map2DEnvironment,
    task_class: str,
    task_type: str,
    family: str,
    start_node_id: str | None,
    goal_node_id: str | None,
    goal_token: str | None = None,
    horizon_steps: int | None = None,
    expected_output_type: str | None = None,
    adaptation_budget_steps: int | None = None,
) -> dict[str, Any]:
    """Return canonical task-template identifiers."""

    start_goal = None
    if start_node_id is not None:
        start_goal = start_goal_template_metadata(environment, start_node_id, goal_node_id)
    query_template_id = stable_template_hash(
        {
            "task_class": task_class,
            "task_type": task_type,
            "family": family,
            "expected_output_type": expected_output_type,
        }
    )
    budget_template_id = stable_template_hash(
        {
            "task_class": task_class,
            "task_type": task_type,
            "horizon_steps": horizon_steps,
            "adaptation_budget_steps": adaptation_budget_steps,
        }
    )
    task_template_id = stable_template_hash(
        {
            "task_class": task_class,
            "task_type": task_type,
            "family": family,
            "goal_token": goal_token,
            "query_template_id": query_template_id,
            "budget_template_id": budget_template_id,
            "start_goal_template_id": None if start_goal is None else start_goal["start_goal_template_id"],
            "structural_fingerprint": structural_signature_for_environment(environment).normalized_fingerprint,
        }
    )
    return {
        "task_template_id": task_template_id,
        "query_template_id": query_template_id,
        "budget_template_id": budget_template_id,
        "start_goal_template_id": None if start_goal is None else start_goal["start_goal_template_id"],
        "start_role_template_id": None if start_goal is None else start_goal["start_role_template_id"],
        "goal_role_template_id": None if start_goal is None else start_goal["goal_role_template_id"],
        "distance_steps": None if start_goal is None else start_goal["distance_steps"],
    }


def semantic_remap_template_id(base_environment: Map2DEnvironment, transformed_environment: Map2DEnvironment) -> str:
    """Return a template identifier for a semantic remapping."""

    payload = {
        "goal_tokens": tuple(
            sorted(
                (
                    token,
                    node_role_template_id(base_environment, node_id),
                    node_role_template_id(transformed_environment, transformed_environment.goal_tokens[token]),
                )
                for token, node_id in base_environment.goal_tokens.items()
            )
        ),
        "landmarks": tuple(
            sorted(
                (
                    node_role_template_id(base_environment, node_id),
                    label,
                    transformed_environment.landmark_by_node.get(node_id, ""),
                )
                for node_id, label in base_environment.landmark_by_node.items()
            )
        ),
    }
    return stable_template_hash(payload)
