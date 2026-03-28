#!/usr/bin/env python3
"""Scaffold entry point for building MapShift benchmark artifacts."""

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
    args = parser.parse_args()

    bundle = load_release_bundle(args.config)
    print(f"Loaded benchmark release {bundle.root.release_name} for build planning.")
    print("Benchmark build execution is scaffolded but not implemented yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
