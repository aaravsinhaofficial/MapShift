#!/usr/bin/env python3
"""Generate deterministic canonical release split artifacts for MapShift-2D."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mapshift.core.schemas import load_release_bundle
from mapshift.splits.builders import build_canonical_release_split_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="configs/benchmark/release_v0_1.json")
    parser.add_argument("--samples-per-motif", type=int, default=1)
    parser.add_argument("--task-samples-per-class", type=int, default=1)
    parser.add_argument("--output", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    bundle = load_release_bundle(args.config)
    split_bundle = build_canonical_release_split_bundle(
        release_bundle=bundle,
        sample_count_per_motif=args.samples_per_motif,
        task_samples_per_class=args.task_samples_per_class,
    )
    payload = json.dumps(split_bundle.to_dict(), indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0 if split_bundle.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
