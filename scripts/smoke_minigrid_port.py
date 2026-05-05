#!/usr/bin/env python3
"""Smoke-test the MiniGrid port and its MapShift intervention validators."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mapshift.envs.minigrid_port import (
    MiniGridPortDependencyError,
    MiniGridPortGenerator,
    build_minigrid_intervention,
    validate_minigrid_intervention_pair,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--motif", default="two_room_door")
    parser.add_argument("--family", choices=("metric", "topology", "dynamics", "semantic"), default="topology")
    parser.add_argument("--severity", type=int, choices=(0, 1, 2, 3), default=2)
    parser.add_argument("--instantiate-minigrid", action="store_true", help="Instantiate the optional MiniGrid backend and call reset().")
    args = parser.parse_args()

    generator = MiniGridPortGenerator()
    base = generator.generate(seed=args.seed, motif_tag=args.motif).environment
    intervention = build_minigrid_intervention(args.family)
    transformed = intervention.apply(base, severity=args.severity, seed=args.seed + 17).environment
    validation = validate_minigrid_intervention_pair(base, transformed, family=args.family, severity=args.severity)

    backend = {"requested": bool(args.instantiate_minigrid), "ok": None, "error": ""}
    if args.instantiate_minigrid:
        try:
            env = transformed.to_minigrid_env()
            env.reset(seed=args.seed)
            backend["ok"] = True
        except MiniGridPortDependencyError as exc:
            backend["ok"] = False
            backend["error"] = str(exc)

    payload = {
        "base_environment_id": base.environment_id,
        "transformed_environment_id": transformed.environment_id,
        "family": args.family,
        "severity": args.severity,
        "validation_ok": validation.ok,
        "validation_issues": list(validation.issues),
        "validation_metrics": validation.metrics,
        "base_ascii": base.render_ascii(),
        "transformed_ascii": transformed.render_ascii(),
        "optional_minigrid_backend": backend,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if validation.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
