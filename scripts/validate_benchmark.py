#!/usr/bin/env python3
"""Validate and summarize a MapShift release bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mapshift.core.schemas import ConfigValidationError, load_release_bundle
from mapshift.envs.procthor.generator import ProcTHORGenerator
from mapshift.envs.procthor.validation import validate_procthor_scene
from mapshift.splits.builders import build_canonical_release_split_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/benchmark/release_v0_1.json",
        help="Path to the root benchmark release config.",
    )
    parser.add_argument(
        "--tier",
        choices=("mapshift_2d", "mapshift_3d", "all"),
        default="all",
        help="Validation tier. Use mapshift_2d for the primary artifact path.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path = Path(args.config)

    try:
        bundle = load_release_bundle(config_path)
    except ConfigValidationError as exc:
        print(f"Config validation failed: {exc}", file=sys.stderr)
        return 1

    active_tier = args.tier
    split_bundle = build_canonical_release_split_bundle(bundle, sample_count_per_motif=1, task_samples_per_class=1)
    procthor_issues: list[str] = []
    procthor_backend_mode = "skipped_for_mapshift_2d"
    if active_tier in {"all", "mapshift_3d"}:
        procthor_generator = ProcTHORGenerator(bundle.env3d)
        procthor_result = procthor_generator.sample(seed=13)
        procthor_scene = procthor_result.scene
        procthor_issues = validate_procthor_scene(procthor_scene) if procthor_scene is not None else ["generator_returned_no_scene"]
        procthor_backend_mode = procthor_generator.backend_status.get("mode", "unknown")
    summary = bundle.summary() | {
        "validation_tier": active_tier,
        "split_validation_ok": split_bundle.ok,
        "split_validation_issues": list(split_bundle.validation_issues),
        "split_warning_count": len(split_bundle.leakage_report.warnings),
        "split_benign_count": len(split_bundle.leakage_report.benign),
        "split_error_count": len(split_bundle.leakage_report.errors),
        "split_environment_counts": {
            split_name: len(manifest.environment_ids) for split_name, manifest in sorted(split_bundle.manifests.items())
        },
        "env3d_smoke_ok": active_tier == "mapshift_2d" or not procthor_issues,
        "env3d_smoke_issues": procthor_issues,
        "env3d_backend_mode": procthor_backend_mode,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    tier_ok = split_bundle.ok and (active_tier == "mapshift_2d" or not procthor_issues)
    return 0 if tier_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
