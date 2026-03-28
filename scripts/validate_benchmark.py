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

    print(json.dumps(bundle.summary(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
