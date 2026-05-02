#!/usr/bin/env python3
"""Generate a family-wise calibration report for the first MapShift baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mapshift.core.schemas import load_release_bundle
from mapshift.runners.evaluate import run_calibration_suite


def _default_run_configs_for_tier(tier: str) -> list[str]:
    if tier == "mapshift_3d":
        return [
            "configs/calibration/oracle_post_intervention_planner_v0_1.json",
            "configs/calibration/weak_heuristic_baseline_v0_1.json",
        ]
    return [
        "configs/calibration/oracle_post_intervention_planner_v0_1.json",
        "configs/calibration/same_environment_upper_baseline_v0_1.json",
        "configs/calibration/weak_heuristic_baseline_v0_1.json",
        "configs/calibration/monolithic_recurrent_world_model_v0_1.json",
        "configs/calibration/persistent_memory_world_model_v0_1.json",
        "configs/calibration/relational_graph_world_model_v0_1.json",
        "configs/calibration/structured_dynamics_world_model_v0_1.json",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="configs/benchmark/release_v0_1.json")
    parser.add_argument(
        "--run-config",
        action="append",
        dest="run_configs",
        default=[],
        help="Path to a baseline run config. May be provided multiple times.",
    )
    parser.add_argument("--tier", choices=("mapshift_2d", "mapshift_3d"), default=None)
    parser.add_argument("--samples-per-motif", type=int, default=1)
    parser.add_argument("--task-samples-per-class", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    release_bundle = load_release_bundle(args.config)
    active_tier = args.tier or release_bundle.root.primary_tier
    run_configs = args.run_configs or _default_run_configs_for_tier(active_tier)
    report = run_calibration_suite(
        release_bundle=release_bundle,
        baseline_run_configs=run_configs,
        sample_count_per_motif=args.samples_per_motif,
        task_samples_per_class=args.task_samples_per_class,
        tier=active_tier,
    )
    payload = report.to_dict()
    if args.output is not None:
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
