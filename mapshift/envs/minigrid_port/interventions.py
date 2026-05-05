"""Controlled intervention families for the MiniGrid port."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any

from mapshift.core.manifests import InterventionManifest

from .state import GridPos, MiniGridPortEnvironment


_METRIC_SCALE = {0: 1.0, 1: 1.1, 2: 1.25, 3: 1.5}
_SLIP_PROBABILITY = {0: 0.0, 1: 0.05, 2: 0.10, 3: 0.20}


@dataclass(frozen=True)
class MiniGridPortInterventionResult:
    manifest: InterventionManifest
    environment: MiniGridPortEnvironment


class MiniGridPortIntervention:
    def __init__(self, family: str) -> None:
        if family not in {"metric", "topology", "dynamics", "semantic"}:
            raise KeyError(f"Unsupported MiniGrid port intervention family: {family}")
        self.family = family

    def apply(self, environment: MiniGridPortEnvironment, severity: int, seed: int) -> MiniGridPortInterventionResult:
        if severity not in {0, 1, 2, 3}:
            raise ValueError("MiniGrid port severity must be one of 0, 1, 2, 3")
        rng = random.Random(seed)
        if severity == 0:
            transformed = environment.clone(environment_id=f"{environment.environment_id}-{self.family}-s0")
            transformed.history = (*environment.history, f"{self.family}:severity0")
        elif self.family == "metric":
            transformed = self._apply_metric(environment, severity)
        elif self.family == "topology":
            transformed = self._apply_topology(environment, severity, rng)
        elif self.family == "dynamics":
            transformed = self._apply_dynamics(environment, severity)
        else:
            transformed = self._apply_semantic(environment, severity)

        manifest = InterventionManifest(
            artifact_id=f"intervention-{transformed.environment_id}",
            artifact_type="intervention",
            benchmark_version="0.1.0",
            code_version="minigrid-port-v1",
            config_hash=self._config_hash(),
            parent_ids=[environment.environment_id],
            seed_values=[seed],
            metadata={
                "platform": "MiniGrid",
                "adapter": "mapshift.envs.minigrid_port",
                "serialized_environment": transformed.to_dict(),
                "base_environment": environment.to_dict(),
                "invariant_contract": self._invariant_contract(),
            },
            base_environment_id=environment.environment_id,
            intervention_family=self.family,
            severity_level=severity,
            severity_parameter=self._severity_parameter(),
            severity_value=self._severity_value(severity),
        )
        return MiniGridPortInterventionResult(manifest=manifest, environment=transformed)

    def _apply_metric(self, environment: MiniGridPortEnvironment, severity: int) -> MiniGridPortEnvironment:
        payload = environment.to_dict()
        payload["environment_id"] = f"{environment.environment_id}-metric-s{severity}"
        payload["movement_cost_scale"] = _METRIC_SCALE[severity]
        payload["history"] = [*payload["history"], f"metric:movement_cost_scale={_METRIC_SCALE[severity]}"]
        return MiniGridPortEnvironment.from_dict(payload)

    def _apply_dynamics(self, environment: MiniGridPortEnvironment, severity: int) -> MiniGridPortEnvironment:
        payload = environment.to_dict()
        payload["environment_id"] = f"{environment.environment_id}-dynamics-s{severity}"
        payload["slip_probability"] = _SLIP_PROBABILITY[severity]
        payload["history"] = [*payload["history"], f"dynamics:slip_probability={_SLIP_PROBABILITY[severity]}"]
        return MiniGridPortEnvironment.from_dict(payload)

    def _apply_semantic(self, environment: MiniGridPortEnvironment, severity: int) -> MiniGridPortEnvironment:
        tokens = sorted(environment.token_positions)
        current_index = tokens.index(environment.active_token)
        new_token = tokens[(current_index + severity) % len(tokens)]
        payload = environment.to_dict()
        payload["environment_id"] = f"{environment.environment_id}-semantic-s{severity}"
        payload["active_token"] = new_token
        payload["mission"] = f"go to {new_token}"
        payload["history"] = [*payload["history"], f"semantic:active_token={new_token}"]
        return MiniGridPortEnvironment.from_dict(payload)

    def _apply_topology(self, environment: MiniGridPortEnvironment, severity: int, rng: random.Random) -> MiniGridPortEnvironment:
        candidates = list(self._topology_blocker_candidates(environment))
        rng.shuffle(candidates)
        blocker_count = min(len(candidates), severity)
        transformed = environment.clone(environment_id=f"{environment.environment_id}-topology-s{severity}")
        for blocker in candidates[:blocker_count]:
            candidate = transformed.with_cell(blocker, "wall")
            if candidate.shortest_path():
                transformed = candidate
        payload = transformed.to_dict()
        payload["history"] = [*payload["history"], f"topology:blockers={blocker_count}"]
        return MiniGridPortEnvironment.from_dict(payload)

    def _topology_blocker_candidates(self, environment: MiniGridPortEnvironment) -> tuple[GridPos, ...]:
        path = environment.shortest_path()
        protected = {environment.start_pos, environment.goal_pos, *environment.token_positions.values()}
        return tuple(pos for pos in path[1:-1] if pos not in protected and environment.cell_type(pos) == "floor")

    def _severity_parameter(self) -> str:
        return {
            "metric": "movement_cost_scale",
            "topology": "blocker_count",
            "dynamics": "slip_probability",
            "semantic": "active_goal_token_offset",
        }[self.family]

    def _severity_value(self, severity: int) -> float:
        if self.family == "metric":
            return _METRIC_SCALE[severity]
        if self.family == "dynamics":
            return _SLIP_PROBABILITY[severity]
        return float(severity)

    def _invariant_contract(self) -> dict[str, Any]:
        return {
            "metric": ["topology", "dynamics", "semantics"],
            "topology": ["metric", "dynamics", "semantics"],
            "dynamics": ["topology", "metric", "semantics"],
            "semantic": ["topology", "metric", "dynamics"],
        }[self.family]

    def _config_hash(self) -> str:
        return hashlib.sha1(json.dumps({"family": self.family}, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def build_minigrid_intervention(family: str) -> MiniGridPortIntervention:
    return MiniGridPortIntervention(family)
