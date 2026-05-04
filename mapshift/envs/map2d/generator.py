"""Config-driven occupancy-grid generation for MapShift-2D."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from typing import Any

from mapshift.core.manifests import EnvironmentManifest
from mapshift.core.schemas import Env2DConfig
from mapshift.splits.motifs import semantic_template_metadata, structural_signature_for_environment

from .dynamics import DynamicsParameters2D
from .state import AgentPose2D, Cell, Map2DEnvironment, Map2DNode, _edge_key, _orthogonal_path


def _rect_cells(center: Cell, radius: int, width: int, height: int) -> list[Cell]:
    row, col = center
    cells: list[Cell] = []
    for current_row in range(max(0, row - radius), min(height, row + radius + 1)):
        for current_col in range(max(0, col - radius), min(width, col + radius + 1)):
            cells.append((current_row, current_col))
    return cells


@dataclass(frozen=True)
class Map2DGenerationResult:
    """Container returned by the 2D generator."""

    manifest: EnvironmentManifest
    environment: Map2DEnvironment | None = None


class Map2DGenerator:
    """Occupancy-grid generator for the MapShift-2D benchmark tier."""

    _LANDMARKS = ("red_tower", "blue_gate", "green_beacon", "amber_arch", "white_column")

    _TEMPLATES: dict[str, dict[str, Any]] = {
        "simple_loop": {
            "motif_family": "loop",
            "positions": {
                "n0": (0, 0),
                "n1": (2, 0),
                "n2": (4, 1),
                "n3": (4, 3),
                "n4": (2, 4),
                "n5": (0, 3),
            },
            "edges": [("n0", "n1"), ("n1", "n2"), ("n2", "n3"), ("n3", "n4"), ("n4", "n5"), ("n5", "n0")],
        },
        "two_room_connector": {
            "motif_family": "connector",
            "positions": {
                "n0": (0, 0),
                "n1": (0, 2),
                "n2": (2, 1),
                "n3": (4, 1),
                "n4": (6, 0),
                "n5": (6, 2),
                "n6": (8, 1),
            },
            "edges": [("n0", "n1"), ("n1", "n2"), ("n2", "n0"), ("n2", "n3"), ("n3", "n4"), ("n4", "n5"), ("n5", "n6"), ("n6", "n4")],
        },
        "branching_chain": {
            "motif_family": "room-chain",
            "positions": {
                "n0": (0, 1),
                "n1": (2, 1),
                "n2": (4, 1),
                "n3": (6, 1),
                "n4": (2, 3),
                "n5": (4, 3),
                "n6": (6, 3),
            },
            "edges": [("n0", "n1"), ("n1", "n2"), ("n2", "n3"), ("n1", "n4"), ("n2", "n5"), ("n3", "n6")],
        },
        "asymmetric_multi_room_chain": {
            "motif_family": "room-chain",
            "positions": {
                "n0": (0, 1),
                "n1": (2, 1),
                "n2": (4, 1),
                "n3": (6, 1),
                "n4": (8, 1),
                "n5": (4, 3),
                "n6": (6, 3),
                "n7": (8, 3),
            },
            "edges": [("n0", "n1"), ("n1", "n2"), ("n2", "n3"), ("n3", "n4"), ("n2", "n5"), ("n5", "n6"), ("n6", "n7"), ("n4", "n7")],
        },
        "offset_bottleneck": {
            "motif_family": "bottleneck",
            "positions": {
                "n0": (0, 0),
                "n1": (0, 2),
                "n2": (2, 1),
                "n3": (4, 1),
                "n4": (6, 0),
                "n5": (6, 2),
                "n6": (8, 1),
                "n7": (10, 1),
            },
            "edges": [("n0", "n1"), ("n1", "n2"), ("n2", "n0"), ("n2", "n3"), ("n3", "n4"), ("n3", "n5"), ("n4", "n6"), ("n5", "n6"), ("n6", "n7")],
        },
        "nested_bottleneck": {
            "motif_family": "bottleneck",
            "positions": {
                "n0": (0, 1),
                "n1": (2, 1),
                "n2": (4, 1),
                "n3": (6, 1),
                "n4": (8, 1),
                "n5": (10, 1),
                "n6": (4, 3),
                "n7": (8, 3),
            },
            "edges": [("n0", "n1"), ("n1", "n2"), ("n2", "n3"), ("n3", "n4"), ("n4", "n5"), ("n2", "n6"), ("n6", "n3"), ("n4", "n7"), ("n7", "n5")],
        },
        "deceptive_shortcut": {
            "motif_family": "shortcut",
            "positions": {
                "n0": (0, 1),
                "n1": (2, 1),
                "n2": (4, 1),
                "n3": (6, 1),
                "n4": (8, 1),
                "n5": (4, 3),
                "n6": (6, 3),
                "n7": (8, 3),
                "n8": (10, 2),
            },
            "edges": [("n0", "n1"), ("n1", "n2"), ("n2", "n3"), ("n3", "n4"), ("n2", "n5"), ("n5", "n6"), ("n6", "n7"), ("n7", "n8"), ("n4", "n8")],
        },
        "disconnected_subregion": {
            "motif_family": "connector",
            "positions": {
                "n0": (0, 1),
                "n1": (2, 1),
                "n2": (4, 1),
                "n3": (6, 1),
                "n4": (8, 1),
                "n5": (12, 0),
                "n6": (12, 2),
            },
            "edges": [("n0", "n1"), ("n1", "n2"), ("n2", "n3"), ("n3", "n4"), ("n5", "n6")],
        },
        "spiral_loop": {
            "motif_family": "loop",
            "positions": {
                "n0": (0, 0),
                "n1": (2, 0),
                "n2": (4, 0),
                "n3": (4, 2),
                "n4": (2, 2),
                "n5": (2, 4),
                "n6": (6, 4),
                "n7": (6, 1),
                "n8": (8, 1),
            },
            "edges": [("n0", "n1"), ("n1", "n2"), ("n2", "n3"), ("n3", "n4"), ("n4", "n1"), ("n4", "n5"), ("n5", "n6"), ("n6", "n7"), ("n7", "n8")],
        },
        "ladder_loop": {
            "motif_family": "loop",
            "positions": {
                "n0": (0, 0),
                "n1": (2, 0),
                "n2": (4, 0),
                "n3": (6, 0),
                "n4": (0, 3),
                "n5": (2, 3),
                "n6": (4, 3),
                "n7": (6, 3),
                "n8": (8, 1),
                "n9": (8, 5),
            },
            "edges": [("n0", "n1"), ("n1", "n2"), ("n2", "n3"), ("n4", "n5"), ("n5", "n6"), ("n6", "n7"), ("n0", "n4"), ("n1", "n5"), ("n2", "n6"), ("n3", "n7"), ("n7", "n8"), ("n8", "n9")],
        },
        "double_connector_rooms": {
            "motif_family": "connector",
            "positions": {
                "n0": (0, 0),
                "n1": (0, 2),
                "n2": (2, 1),
                "n3": (4, 1),
                "n4": (6, 0),
                "n5": (6, 2),
                "n6": (8, 1),
                "n7": (4, 4),
                "n8": (8, 4),
            },
            "edges": [("n0", "n1"), ("n1", "n2"), ("n2", "n0"), ("n2", "n3"), ("n3", "n4"), ("n3", "n5"), ("n4", "n6"), ("n5", "n6"), ("n3", "n7"), ("n7", "n8"), ("n6", "n8")],
        },
        "zigzag_chain": {
            "motif_family": "room-chain",
            "positions": {
                "n0": (0, 0),
                "n1": (2, 1),
                "n2": (4, 0),
                "n3": (6, 1),
                "n4": (8, 0),
                "n5": (10, 1),
                "n6": (12, 0),
                "n7": (6, 3),
                "n8": (10, 3),
            },
            "edges": [("n0", "n1"), ("n1", "n2"), ("n2", "n3"), ("n3", "n4"), ("n4", "n5"), ("n5", "n6"), ("n3", "n7"), ("n5", "n8")],
        },
        "forked_bottleneck": {
            "motif_family": "bottleneck",
            "positions": {
                "n0": (0, 1),
                "n1": (2, 1),
                "n2": (4, 1),
                "n3": (6, 1),
                "n4": (8, 0),
                "n5": (8, 2),
                "n6": (10, 0),
                "n7": (10, 2),
                "n8": (12, 1),
            },
            "edges": [("n0", "n1"), ("n1", "n2"), ("n2", "n3"), ("n3", "n4"), ("n3", "n5"), ("n4", "n6"), ("n5", "n7"), ("n6", "n8"), ("n7", "n8")],
        },
        "hub_spoke_deadends": {
            "motif_family": "room-chain",
            "positions": {
                "n0": (0, 2),
                "n1": (2, 2),
                "n2": (4, 2),
                "n3": (6, 2),
                "n4": (4, 0),
                "n5": (4, 4),
                "n6": (6, 0),
                "n7": (6, 4),
                "n8": (8, 2),
                "n9": (10, 2),
            },
            "edges": [("n0", "n1"), ("n1", "n2"), ("n2", "n3"), ("n3", "n8"), ("n8", "n9"), ("n2", "n4"), ("n2", "n5"), ("n3", "n6"), ("n3", "n7")],
        },
        "parallel_corridor": {
            "motif_family": "connector",
            "positions": {
                "n0": (0, 0),
                "n1": (2, 0),
                "n2": (4, 0),
                "n3": (6, 0),
                "n4": (8, 0),
                "n5": (0, 3),
                "n6": (2, 3),
                "n7": (4, 3),
                "n8": (6, 3),
                "n9": (8, 3),
                "n10": (10, 1),
            },
            "edges": [("n0", "n1"), ("n1", "n2"), ("n2", "n3"), ("n3", "n4"), ("n5", "n6"), ("n6", "n7"), ("n7", "n8"), ("n8", "n9"), ("n0", "n5"), ("n2", "n7"), ("n4", "n9"), ("n9", "n10")],
        },
        "ring_with_tail": {
            "motif_family": "loop",
            "positions": {
                "n0": (0, 1),
                "n1": (2, 0),
                "n2": (4, 0),
                "n3": (6, 1),
                "n4": (6, 3),
                "n5": (4, 4),
                "n6": (2, 4),
                "n7": (0, 3),
                "n8": (8, 2),
                "n9": (10, 2),
            },
            "edges": [("n0", "n1"), ("n1", "n2"), ("n2", "n3"), ("n3", "n4"), ("n4", "n5"), ("n5", "n6"), ("n6", "n7"), ("n7", "n0"), ("n4", "n8"), ("n8", "n9")],
        },
        "offset_loop_bridge": {
            "motif_family": "loop",
            "positions": {
                "n0": (0, 0),
                "n1": (2, 0),
                "n2": (4, 1),
                "n3": (2, 2),
                "n4": (0, 2),
                "n5": (6, 1),
                "n6": (8, 0),
                "n7": (10, 1),
                "n8": (8, 2),
                "n9": (6, 3),
            },
            "edges": [("n0", "n1"), ("n1", "n2"), ("n2", "n3"), ("n3", "n4"), ("n4", "n0"), ("n2", "n5"), ("n5", "n6"), ("n6", "n7"), ("n7", "n8"), ("n8", "n9"), ("n9", "n5")],
        },
        "split_hallway": {
            "motif_family": "room-chain",
            "positions": {
                "n0": (0, 2),
                "n1": (2, 2),
                "n2": (4, 2),
                "n3": (6, 2),
                "n4": (8, 2),
                "n5": (4, 0),
                "n6": (6, 0),
                "n7": (8, 0),
                "n8": (4, 4),
                "n9": (6, 4),
                "n10": (8, 4),
            },
            "edges": [("n0", "n1"), ("n1", "n2"), ("n2", "n3"), ("n3", "n4"), ("n2", "n5"), ("n5", "n6"), ("n6", "n7"), ("n2", "n8"), ("n8", "n9"), ("n9", "n10")],
        },
        "braided_shortcut": {
            "motif_family": "shortcut",
            "positions": {
                "n0": (0, 0),
                "n1": (2, 1),
                "n2": (4, 0),
                "n3": (6, 1),
                "n4": (8, 0),
                "n5": (10, 1),
                "n6": (2, 4),
                "n7": (4, 3),
                "n8": (6, 4),
                "n9": (8, 3),
                "n10": (10, 4),
            },
            "edges": [("n0", "n1"), ("n1", "n2"), ("n2", "n3"), ("n3", "n4"), ("n4", "n5"), ("n1", "n6"), ("n6", "n7"), ("n7", "n8"), ("n8", "n9"), ("n9", "n10"), ("n3", "n8"), ("n5", "n10")],
        },
        "narrow_gate_cluster": {
            "motif_family": "bottleneck",
            "positions": {
                "n0": (0, 1),
                "n1": (2, 0),
                "n2": (2, 2),
                "n3": (4, 1),
                "n4": (6, 1),
                "n5": (8, 0),
                "n6": (8, 2),
                "n7": (10, 1),
                "n8": (12, 1),
                "n9": (6, 4),
            },
            "edges": [("n0", "n1"), ("n0", "n2"), ("n1", "n3"), ("n2", "n3"), ("n3", "n4"), ("n4", "n5"), ("n4", "n6"), ("n5", "n7"), ("n6", "n7"), ("n7", "n8"), ("n4", "n9")],
        },
        "cul_de_sac_shortcut": {
            "motif_family": "shortcut",
            "positions": {
                "n0": (0, 1),
                "n1": (2, 1),
                "n2": (4, 1),
                "n3": (6, 1),
                "n4": (8, 1),
                "n5": (10, 1),
                "n6": (4, 3),
                "n7": (6, 3),
                "n8": (8, 3),
                "n9": (6, 5),
                "n10": (12, 2),
            },
            "edges": [("n0", "n1"), ("n1", "n2"), ("n2", "n3"), ("n3", "n4"), ("n4", "n5"), ("n2", "n6"), ("n6", "n7"), ("n7", "n8"), ("n8", "n4"), ("n7", "n9"), ("n5", "n10")],
        },
        "island_bridge": {
            "motif_family": "connector",
            "positions": {
                "n0": (0, 0),
                "n1": (2, 0),
                "n2": (4, 0),
                "n3": (6, 0),
                "n4": (8, 0),
                "n5": (10, 1),
                "n6": (4, 3),
                "n7": (6, 3),
                "n8": (8, 3),
                "n9": (12, 3),
            },
            "edges": [("n0", "n1"), ("n1", "n2"), ("n2", "n3"), ("n3", "n4"), ("n4", "n5"), ("n2", "n6"), ("n6", "n7"), ("n7", "n8"), ("n8", "n4"), ("n5", "n9")],
        },
        "broken_bridge_islands": {
            "motif_family": "connector",
            "positions": {
                "n0": (0, 0),
                "n1": (2, 0),
                "n2": (4, 1),
                "n3": (6, 1),
                "n4": (8, 0),
                "n5": (10, 0),
                "n6": (0, 4),
                "n7": (2, 4),
                "n8": (4, 5),
                "n9": (6, 5),
                "n10": (8, 4),
                "n11": (10, 4),
            },
            "edges": [("n0", "n1"), ("n1", "n2"), ("n2", "n3"), ("n3", "n4"), ("n4", "n5"), ("n6", "n7"), ("n7", "n8"), ("n8", "n9"), ("n9", "n10"), ("n10", "n11")],
        },
        "asymmetric_loop_chain": {
            "motif_family": "room-chain",
            "positions": {
                "n0": (0, 2),
                "n1": (2, 2),
                "n2": (4, 2),
                "n3": (6, 2),
                "n4": (8, 2),
                "n5": (4, 0),
                "n6": (6, 0),
                "n7": (6, 4),
                "n8": (8, 4),
                "n9": (10, 3),
            },
            "edges": [("n0", "n1"), ("n1", "n2"), ("n2", "n3"), ("n3", "n4"), ("n2", "n5"), ("n5", "n6"), ("n6", "n3"), ("n3", "n7"), ("n7", "n8"), ("n8", "n9")],
        },
    }

    def __init__(self, config: Env2DConfig | Any) -> None:
        self.config = config

    def generate(self, seed: int, motif_tag: str | None = None) -> Map2DGenerationResult:
        rng = random.Random(seed)
        motif = motif_tag or self.config.motif_families[seed % len(self.config.motif_families)]
        split_name = self._split_for_motif(motif)
        environment = self._build_environment(motif=motif, split_name=split_name, seed=seed, rng=rng)
        structural_signature = structural_signature_for_environment(environment)
        semantic_metadata = semantic_template_metadata(environment)
        environment.metadata.update(
            {
                "motif_tags": list(structural_signature.motif_tags),
                "structural_signature": structural_signature.to_dict(),
                **semantic_metadata,
            }
        )
        manifest = EnvironmentManifest(
            artifact_id=f"env-artifact-{environment.environment_id}",
            artifact_type="environment",
            benchmark_version="0.1.0",
            code_version="occupancy-grid-v1",
            config_hash=self._config_hash(),
            environment_id=environment.environment_id,
            tier="mapshift_2d",
            motif_tags=list(structural_signature.motif_tags),
            split_name=split_name,
            metadata={
                "node_count": environment.node_count(),
                "edge_count": environment.edge_count(),
                "free_cell_count": environment.free_cell_count(),
                "geometry_scale": environment.geometry_scale,
                "motif_tag": motif,
                "motif_family": environment.metadata.get("motif_family", ""),
                "motif_tags": list(structural_signature.motif_tags),
                "structural_signature": structural_signature.to_dict(),
                **semantic_metadata,
                "serialized_environment": environment.to_dict(),
            },
            seed_values=[seed],
        )
        return Map2DGenerationResult(manifest=manifest, environment=environment)

    def replay_from_manifest(self, manifest: EnvironmentManifest) -> Map2DEnvironment:
        """Replay an environment exactly from a manifest payload."""

        return Map2DEnvironment.from_manifest_metadata(manifest.metadata)

    def _config_hash(self) -> str:
        config_blob = json.dumps(asdict(self.config), sort_keys=True).encode("utf-8")
        return hashlib.sha1(config_blob).hexdigest()[:12]

    def _split_for_motif(self, motif: str) -> str:
        if motif in self.config.splits.train_motifs:
            return "train"
        if motif in self.config.splits.val_motifs:
            return "val"
        if motif in self.config.splits.test_motifs:
            return "test"
        return "unknown"

    def _build_environment(self, motif: str, split_name: str, seed: int, rng: random.Random) -> Map2DEnvironment:
        template = self._TEMPLATES[motif]
        width = self.config.map_size_cells[0]
        height = self.config.map_size_cells[1]

        node_centers = self._node_centers(template["positions"], width, height)
        room_radius = max(2, min(width, height) // 18)

        nodes = {
            node_id: Map2DNode(node_id=node_id, row=center[0], col=center[1], landmark="")
            for node_id, center in node_centers.items()
        }
        room_cells = {node_id: _rect_cells(center, room_radius, width, height) for node_id, center in node_centers.items()}
        adjacency = {node_id: [] for node_id in nodes}
        edge_corridors: dict[str, list[Cell]] = {}
        for left, right in template["edges"]:
            adjacency[left].append(right)
            adjacency[right].append(left)
            corridor = self._build_corridor(node_centers[left], node_centers[right], left, right, seed)
            edge_corridors[_edge_key(left, right)] = corridor

        landmark_by_node = self._assign_landmarks(nodes, rng)
        goal_tokens = self._assign_goal_tokens(nodes)
        environment_id = f"{motif}-seed{seed}"

        environment = Map2DEnvironment(
            environment_id=environment_id,
            motif_tag=motif,
            split_name=split_name,
            seed=seed,
            width_cells=width,
            height_cells=height,
            occupancy_grid=[[1 for _ in range(width)] for _ in range(height)],
            nodes=nodes,
            adjacency={node_id: sorted(neighbors) for node_id, neighbors in adjacency.items()},
            room_cells=room_cells,
            edge_corridors=edge_corridors,
            start_node_id="n0",
            goal_node_id=max(nodes.keys()),
            landmark_by_node=landmark_by_node,
            goal_tokens=goal_tokens,
            dynamics=DynamicsParameters2D(),
            occupancy_resolution_m=self.config.occupancy_resolution_m,
            geometry_scale=1.0,
            observation_radius_m=self.config.observation.radius_m,
            field_of_view_deg=self.config.observation.field_of_view_deg,
            semantic_channels=self.config.observation.semantic_channels,
            history=["base_environment"],
            metadata={
                "generator_name": self.config.generator_name,
                "room_radius_cells": room_radius,
                "motif_family": template["motif_family"],
            },
        )
        environment.rebuild_occupancy()
        start, goal = environment.farthest_pair()
        environment.start_node_id = start
        environment.goal_node_id = goal
        environment.agent_pose = AgentPose2D(x=float(environment.start_cell[1]), y=float(environment.start_cell[0]), theta_deg=0.0)
        canonical_token = next(iter(environment.goal_tokens))
        environment.goal_tokens[canonical_token] = goal
        for node_id, label in environment.landmark_by_node.items():
            node = environment.nodes[node_id]
            environment.nodes[node_id] = Map2DNode(node_id=node.node_id, row=node.row, col=node.col, landmark=label)
        return environment

    def _node_centers(self, positions: dict[str, tuple[int, int]], width: int, height: int) -> dict[str, Cell]:
        max_x = max(coords[0] for coords in positions.values())
        max_y = max(coords[1] for coords in positions.values())
        x_scale = max(4, min(8, width // max(max_x + 6, 1)))
        y_scale = max(4, min(8, height // max(max_y + 6, 1)))
        col_offset = max(4, (width - (max_x * x_scale)) // 2)
        row_offset = max(4, (height - (max_y * y_scale)) // 2)

        return {
            node_id: (row_offset + (coords[1] * y_scale), col_offset + (coords[0] * x_scale))
            for node_id, coords in positions.items()
        }

    def _build_corridor(self, start: Cell, goal: Cell, left: str, right: str, seed: int) -> list[Cell]:
        edge_key = _edge_key(left, right)
        prefer_horizontal = (sum(ord(char) for char in edge_key) + seed) % 2 == 0
        return _orthogonal_path(start, goal, prefer_horizontal=prefer_horizontal)

    def _assign_landmarks(self, nodes: dict[str, Map2DNode], rng: random.Random) -> dict[str, str]:
        node_ids = list(nodes)
        rng.shuffle(node_ids)
        landmark_by_node: dict[str, str] = {}
        for index, node_id in enumerate(node_ids[: min(len(node_ids), len(self._LANDMARKS))]):
            landmark_by_node[node_id] = self._LANDMARKS[index]
        return landmark_by_node

    def _assign_goal_tokens(self, nodes: dict[str, Map2DNode]) -> dict[str, str]:
        node_ids = sorted(nodes)
        tokens = ("target_alpha", "target_beta", "target_gamma")
        return {token: node_ids[index % len(node_ids)] for index, token in enumerate(tokens)}
