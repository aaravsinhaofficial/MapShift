#!/usr/bin/env python3
"""Run the canonical MapShift protocol-comparison analyses."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mapshift.core.schemas import load_release_bundle
from mapshift.runners.compare_protocols import run_protocol_comparison_suite


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
    parser.add_argument("--samples-per-motif", type=int, default=1)
    parser.add_argument("--task-samples-per-class", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    bundle = load_release_bundle(args.config)
    run_configs = args.run_configs or [
        "configs/calibration/oracle_post_intervention_planner_v0_1.json",
        "configs/calibration/weak_heuristic_baseline_v0_1.json",
        "configs/calibration/monolithic_recurrent_world_model_v0_1.json",
        "configs/calibration/persistent_memory_world_model_v0_1.json",
        "configs/calibration/relational_graph_world_model_v0_1.json",
    ]
    report = run_protocol_comparison_suite(
        release_bundle=bundle,
        baseline_run_configs=run_configs,
        sample_count_per_motif=args.samples_per_motif,
        task_samples_per_class=args.task_samples_per_class,
    )
    payload = report.to_dict()
    if args.output is not None:
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
