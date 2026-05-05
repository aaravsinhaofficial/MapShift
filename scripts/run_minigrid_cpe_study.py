#!/usr/bin/env python3
"""Run the compact MiniGrid CPE sanity-transfer study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mapshift.analysis.minigrid_cpe import DEFAULT_SEVERITIES, PROTOCOLS, run_minigrid_cpe_study
from mapshift.envs.minigrid_port import MiniGridPortGenerator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="outputs/studies/minigrid_cpe_sanity_transfer",
        help="Directory for study bundle, tables, and raw records.",
    )
    parser.add_argument("--seeds-per-motif", type=int, default=4, help="Number of deterministic seeds per MiniGrid motif.")
    parser.add_argument(
        "--motif",
        action="append",
        choices=MiniGridPortGenerator().motif_tags,
        help="Motif to include. Repeat to select multiple; default uses all MiniGrid port motifs.",
    )
    parser.add_argument(
        "--severity",
        action="append",
        type=int,
        choices=DEFAULT_SEVERITIES,
        help="Severity level to include. Repeat to select multiple; default uses 0, 1, 2, and 3.",
    )
    parser.add_argument(
        "--protocol",
        action="append",
        choices=PROTOCOLS,
        help="Protocol to include. Repeat to select multiple; default uses cep, same_environment, and no_exploration.",
    )
    parser.add_argument("--print-summary", action="store_true", help="Print artifact paths and headline diagnostics.")
    args = parser.parse_args(argv)

    if args.seeds_per_motif < 1:
        parser.error("--seeds-per-motif must be positive")

    result = run_minigrid_cpe_study(
        output_dir=Path(args.output_dir),
        seeds_per_motif=args.seeds_per_motif,
        motifs=tuple(args.motif) if args.motif else None,
        severity_levels=tuple(args.severity) if args.severity else DEFAULT_SEVERITIES,
        protocols=tuple(args.protocol) if args.protocol else PROTOCOLS,
    )

    if args.print_summary:
        payload = {
            "artifact_paths": result.artifact_paths,
            "health_summary": result.summary["health_summary"],
            "protocol_sensitivity": result.summary["protocol_sensitivity"],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
