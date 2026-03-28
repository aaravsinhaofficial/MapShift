#!/usr/bin/env python3
"""Run the first full MapShift-2D benchmark study bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mapshift.analysis.study import (
    load_mapshift_2d_study_config,
    run_mapshift_2d_study,
    write_mapshift_2d_study_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "study_config",
        nargs="?",
        default="configs/analysis/mapshift_2d_full_study_v0_1.json",
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args()

    study_config = load_mapshift_2d_study_config(args.study_config)
    bundle = run_mapshift_2d_study(study_config)
    output_dir = args.output_dir or study_config.output_subdir or f"outputs/studies/{study_config.study_name}"
    artifact_paths = write_mapshift_2d_study_bundle(bundle, output_dir)

    if args.print_summary:
        summary = {
            "study_name": bundle.study_name,
            "release_name": bundle.release_name,
            "p1_status": bundle.proposition_support["P1"]["status"],
            "p2_status": bundle.proposition_support["P2"]["status"],
            "p3_status": bundle.proposition_support["P3"]["status"],
            "artifact_paths": artifact_paths,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
