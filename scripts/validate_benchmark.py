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
from mapshift.splits.builders import build_canonical_release_split_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/benchmark/release_v0_1.json",
        help="Path to the root benchmark release config.",
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

    split_bundle = build_canonical_release_split_bundle(bundle, sample_count_per_motif=1, task_samples_per_class=1)
    summary = bundle.summary() | {
        "split_validation_ok": split_bundle.ok,
        "split_validation_issues": list(split_bundle.validation_issues),
        "split_warning_count": len(split_bundle.leakage_report.warnings),
        "split_error_count": len(split_bundle.leakage_report.errors),
        "split_environment_counts": {
            split_name: len(manifest.environment_ids) for split_name, manifest in sorted(split_bundle.manifests.items())
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if split_bundle.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
