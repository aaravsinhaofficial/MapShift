#!/usr/bin/env python3
"""Scaffold entry point for MapShift evaluation runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mapshift.core.schemas import load_release_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="configs/benchmark/release_v0_1.json")
    parser.add_argument("--baseline", default="monolithic_recurrent_world_model")
    args = parser.parse_args()

    bundle = load_release_bundle(args.config)
    print(f"Loaded release {bundle.root.release_name}.")
    print(f"Evaluation runner scaffold invoked for baseline {args.baseline}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
