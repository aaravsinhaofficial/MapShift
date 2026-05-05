"""Deterministic MiniGrid adapter generation for MapShift CPE wrappers."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any

from mapshift.core.manifests import EnvironmentManifest

from .state import MiniGridPortEnvironment


@dataclass(frozen=True)
class MiniGridPortGenerationResult:
    manifest: EnvironmentManifest
    environment: MiniGridPortEnvironment


class MiniGridPortGenerator:
    """Generate small MiniGrid-compatible maps with MapShift matched-pair metadata."""

    _TOKENS = ("target_alpha", "target_beta", "target_gamma")
    _MOTIFS = (
        "two_room_door",
        "four_room_cross",
        "corridor_bend",
        "loop_key_room",
        "lava_gap",
        "zigzag_rooms",
    )
    _SPLITS = {
        "two_room_door": "train",
        "four_room_cross": "train",
        "corridor_bend": "val",
        "loop_key_room": "train",
        "lava_gap": "test",
        "zigzag_rooms": "test",
    }

    def __init__(self, width: int = 11, height: int = 11, observation_radius: int = 7) -> None:
        if width < 7 or height < 7:
            raise ValueError("MiniGrid port maps must be at least 7x7")
        self.width = int(width)
        self.height = int(height)
        self.observation_radius = int(observation_radius)

    @property
    def motif_tags(self) -> tuple[str, ...]:
        return self._MOTIFS

    def generate(self, seed: int, motif_tag: str | None = None) -> MiniGridPortGenerationResult:
        rng = random.Random(seed)
        motif = motif_tag or self._MOTIFS[seed % len(self._MOTIFS)]
        if motif not in self._MOTIFS:
            raise KeyError(f"Unsupported MiniGrid port motif: {motif}")
        environment = self._build_environment(seed=seed, motif_tag=motif, rng=rng)
        manifest = EnvironmentManifest(
            artifact_id=f"env-artifact-{environment.environment_id}",
            artifact_type="environment",
            benchmark_version="0.1.0",
            code_version="minigrid-port-v1",
            config_hash=self._config_hash(),
            environment_id=environment.environment_id,
            tier="minigrid_port",
            motif_tags=[motif],
            split_name=environment.split_name,
            seed_values=[seed],
            metadata={
                "platform": "MiniGrid",
                "adapter": "mapshift.envs.minigrid_port",
                "motif_tag": motif,
                "active_token": environment.active_token,
                "topology_signature": list(environment.topology_signature()),
                "semantic_signature": repr(environment.semantic_signature()),
                "serialized_environment": environment.to_dict(),
                "validation_scope": [
                    "matched_base_intervened_pairs",
                    "declared_family_and_severity",
                    "invariant_checks",
                    "optional_minigrid_instantiation",
                ],
            },
        )
        return MiniGridPortGenerationResult(manifest=manifest, environment=environment)

    def replay_from_manifest(self, manifest: EnvironmentManifest) -> MiniGridPortEnvironment:
        serialized = manifest.metadata.get("serialized_environment")
        if not isinstance(serialized, dict):
            raise ValueError("MiniGrid port manifest metadata does not contain serialized_environment")
        return MiniGridPortEnvironment.from_dict(serialized)

    def _config_hash(self) -> str:
        payload = json.dumps({"width": self.width, "height": self.height, "observation_radius": self.observation_radius}, sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

    def _build_environment(self, *, seed: int, motif_tag: str, rng: random.Random) -> MiniGridPortEnvironment:
        grid = [["wall" for _ in range(self.width)] for _ in range(self.height)]
        for row in range(1, self.height - 1):
            for col in range(1, self.width - 1):
                grid[row][col] = "floor"

        if motif_tag == "two_room_door":
            mid_col = self.width // 2
            for row in range(1, self.height - 1):
                grid[row][mid_col] = "wall"
            grid[self.height // 2][mid_col] = "floor"
            start_pos = (self.height - 2, 1)
            token_positions = {
                "target_alpha": (1, self.width - 2),
                "target_beta": (self.height - 2, self.width - 2),
                "target_gamma": (1, 1),
            }
        elif motif_tag == "four_room_cross":
            mid_col = self.width // 2
            mid_row = self.height // 2
            for row in range(1, self.height - 1):
                grid[row][mid_col] = "wall"
            for col in range(1, self.width - 1):
                grid[mid_row][col] = "wall"
            for pos in ((mid_row, 2), (mid_row, self.width - 3), (2, mid_col), (self.height - 3, mid_col), (mid_row, mid_col)):
                grid[pos[0]][pos[1]] = "floor"
            start_pos = (self.height - 2, 1)
            token_positions = {
                "target_alpha": (1, self.width - 2),
                "target_beta": (self.height - 2, self.width - 2),
                "target_gamma": (1, 1),
            }
        elif motif_tag == "corridor_bend":
            for row in range(2, self.height - 2):
                for col in range(2, self.width - 2):
                    grid[row][col] = "wall"
            bend_row = self.height // 2
            for col in range(1, self.width - 1):
                grid[bend_row][col] = "floor"
            for row in range(1, bend_row + 1):
                grid[row][1] = "floor"
            for row in range(bend_row, self.height - 1):
                grid[row][self.width - 2] = "floor"
            start_pos = (1, 1)
            token_positions = {
                "target_alpha": (self.height - 2, self.width - 2),
                "target_beta": (bend_row, self.width - 2),
                "target_gamma": (bend_row, 1),
            }
        elif motif_tag == "loop_key_room":
            for row in range(3, self.height - 3):
                for col in range(3, self.width - 3):
                    grid[row][col] = "wall"
            for pos in ((self.height // 2, 3), (self.height // 2, self.width - 4), (3, self.width // 2), (self.height - 4, self.width // 2)):
                grid[pos[0]][pos[1]] = "floor"
            start_pos = (self.height - 2, 1)
            token_positions = {
                "target_alpha": (1, self.width - 2),
                "target_beta": (self.height - 2, self.width - 2),
                "target_gamma": (1, 1),
            }
        elif motif_tag == "lava_gap":
            lava_row = self.height // 2
            for col in range(1, self.width - 1):
                grid[lava_row][col] = "lava"
            for col in (2, self.width // 2, self.width - 3):
                grid[lava_row][col] = "floor"
            for row in range(2, self.height - 2):
                grid[row][self.width // 2] = "wall"
            for row in (2, lava_row, self.height - 3):
                grid[row][self.width // 2] = "floor"
            start_pos = (self.height - 2, 1)
            token_positions = {
                "target_alpha": (1, self.width - 2),
                "target_beta": (self.height - 2, self.width - 2),
                "target_gamma": (1, 1),
            }
        else:
            openings = {
                3: 2,
                5: self.height - 3,
                7: 4,
            }
            for col, opening in openings.items():
                if col >= self.width - 1:
                    continue
                for row in range(1, self.height - 1):
                    grid[row][col] = "wall"
                grid[opening][col] = "floor"
            start_pos = (self.height - 2, 1)
            token_positions = {
                "target_alpha": (1, self.width - 2),
                "target_beta": (self.height - 2, self.width - 2),
                "target_gamma": (self.height // 2, self.width - 2),
            }

        active_token = self._TOKENS[seed % len(self._TOKENS)]
        environment = MiniGridPortEnvironment(
            environment_id=f"minigrid-{motif_tag}-seed{seed}",
            motif_tag=motif_tag,
            split_name=self._SPLITS[motif_tag],
            seed=seed,
            width=self.width,
            height=self.height,
            grid=tuple(tuple(row) for row in grid),
            start_pos=start_pos,
            agent_dir=rng.randrange(4),
            token_positions=token_positions,
            active_token=active_token,
            observation_radius=self.observation_radius,
            mission=f"go to {active_token}",
            metadata={
                "port": "minigrid",
                "generator": "MiniGridPortGenerator",
                "matched_pair_contract": True,
            },
        )
        if not environment.shortest_path():
            raise RuntimeError(f"Generated unsolvable MiniGrid port environment: {environment.environment_id}")
        return environment
