#!/usr/bin/env python3
"""Generate a family-wise calibration report for the first MapShift baselines."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mapshift.core.schemas import load_release_bundle
from mapshift.runners.evaluate import run_calibration_suite
from mapshift.baselines.api import load_baseline_run_config


def _configure_logging(log_file: Path | None) -> None:
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    root.addHandler(stream)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


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
    parser.add_argument(
        "--model-seed",
        action="append",
        type=int,
        default=[],
        help="Override/expand the provided run config over one or more model seeds.",
    )
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args()

    _configure_logging(args.log_file)
    release_bundle = load_release_bundle(args.config)
    active_tier = args.tier or release_bundle.root.primary_tier
    run_config_paths = args.run_configs or _default_run_configs_for_tier(active_tier)
    if args.model_seed:
        run_configs = []
        for path in run_config_paths:
            config = load_baseline_run_config(path)
            for seed in args.model_seed:
                suffix = f"seed{seed}"
                run_configs.append(replace(config, seed=seed, run_name=f"{config.run_name}_{suffix}"))
    else:
        run_configs = run_config_paths
    report = run_calibration_suite(
        release_bundle=release_bundle,
        baseline_run_configs=run_configs,
        sample_count_per_motif=args.samples_per_motif,
        task_samples_per_class=args.task_samples_per_class,
        tier=active_tier,
    )
    payload = report.to_dict()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    if args.print_summary:
        familywise_scores = [
            {
                "baseline_name": row["baseline_name"],
                "family": row["family"],
                "episode_count": row["episode_count"],
                "family_primary_score": row["family_primary_score"],
            }
            for row in report.familywise_summary["rows"]
        ]
        summary = {
            "release_name": report.release_name,
            "protocol_name": report.protocol_name,
            "records": len(report.records),
            "baseline_names": sorted(report.baseline_metadata),
            "familywise_scores": familywise_scores,
            "output": str(args.output) if args.output is not None else "",
            "log_file": str(args.log_file) if args.log_file is not None else "",
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
