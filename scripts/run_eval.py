#!/usr/bin/env python3
"""Entry point for small MapShift baseline evaluation runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mapshift.core.schemas import load_release_bundle
from mapshift.runners.evaluate import run_calibration_suite


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="configs/benchmark/release_v0_1.json")
    parser.add_argument("--run-config", default="configs/calibration/monolithic_recurrent_world_model_v0_1.json")
    parser.add_argument("--samples-per-motif", type=int, default=1)
    parser.add_argument("--task-samples-per-class", type=int, default=1)
    args = parser.parse_args()

    bundle = load_release_bundle(args.config)
    report = run_calibration_suite(
        release_bundle=bundle,
        baseline_run_configs=[args.run_config],
        sample_count_per_motif=args.samples_per_motif,
        task_samples_per_class=args.task_samples_per_class,
    )
    print(f"Loaded release {bundle.root.release_name}.")
    print(f"Evaluated {Path(args.run_config).name} over {len(report.records)} tasks.")
    print(report.familywise_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
