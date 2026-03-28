#!/usr/bin/env python3
"""Generate and render a MapShift-2D base map plus one intervention from each family."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mapshift.core.schemas import load_release_bundle
from mapshift.envs.map2d import Map2DGenerator, render_debug_view
from mapshift.interventions import build_intervention


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="configs/benchmark/release_v0_1.json")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--motif", default="simple_loop")
    parser.add_argument("--severity", type=int, default=2)
    args = parser.parse_args()

    bundle = load_release_bundle(args.config)
    generator = Map2DGenerator(bundle.env2d)
    base = generator.generate(seed=args.seed, motif_tag=args.motif)

    print("=== Base Environment ===")
    print(render_debug_view(base.environment))
    print()

    for family in bundle.interventions.canonical_family_order:
        intervention = build_intervention(family, bundle.interventions.families[family])
        result = intervention.apply(base.environment, severity=args.severity, seed=args.seed + args.severity)
        print(f"=== {family.title()} Intervention (severity={args.severity}) ===")
        print(render_debug_view(result.environment))
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
