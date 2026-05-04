#!/usr/bin/env python3
"""Analyze held-out stale-map versus belief-update effects from a study bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mapshift.analysis.mechanism_diagnostic import (
    DEFAULT_BELIEF_UPDATE_BASELINE,
    DEFAULT_CEP_PROTOCOL,
    DEFAULT_FAMILIES,
    DEFAULT_SAME_ENV_PROTOCOL,
    DEFAULT_STALE_MAP_BASELINE,
    MechanismDiagnosticConfig,
    analyze_mechanism_diagnostic_bundle,
    render_heldout_consistency_markdown,
    render_heldout_summary_markdown,
    render_paired_bootstrap_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_bundle", help="Path to study_bundle.json, protocol_comparison_report.json, or cep_report.json.")
    parser.add_argument("--output-dir", default="", help="Output directory. Defaults to <study_bundle_dir>/mechanism_diagnostic_analysis.")
    parser.add_argument("--split", default="test", help="Split to analyze, or 'all'. Defaults to held-out test split.")
    parser.add_argument("--family", action="append", dest="families", help="Family to analyze. May be repeated.")
    parser.add_argument("--include-identity", action="store_true", help="Include severity 0 identity interventions.")
    parser.add_argument("--belief-update-baseline", default=DEFAULT_BELIEF_UPDATE_BASELINE)
    parser.add_argument("--stale-map-baseline", default=DEFAULT_STALE_MAP_BASELINE)
    parser.add_argument("--cep-protocol", default=DEFAULT_CEP_PROTOCOL)
    parser.add_argument("--same-environment-protocol", default=DEFAULT_SAME_ENV_PROTOCOL)
    parser.add_argument("--substantial-delta-threshold", type=float, default=0.05)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-unit-field", default="base_environment_id")
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args()

    study_bundle_path = Path(args.study_bundle).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else study_bundle_path.parent / "mechanism_diagnostic_analysis"
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(study_bundle_path.read_text(encoding="utf-8"))
    config = MechanismDiagnosticConfig(
        belief_update_baseline=args.belief_update_baseline,
        stale_map_baseline=args.stale_map_baseline,
        cep_protocol=args.cep_protocol,
        same_environment_protocol=args.same_environment_protocol,
        split_name=args.split,
        families=tuple(args.families or DEFAULT_FAMILIES),
        exclude_identity=not args.include_identity,
        substantial_delta_threshold=args.substantial_delta_threshold,
        bootstrap_resamples=args.resamples,
        confidence_level=args.confidence_level,
        bootstrap_unit_field=args.bootstrap_unit_field,
        bootstrap_seed=args.bootstrap_seed,
    )
    analysis = analyze_mechanism_diagnostic_bundle(payload, config=config)

    paths = {
        "summary": output_dir / "p2_p3_summary.json",
        "heldout_motif_consistency": tables_dir / "heldout_motif_consistency.json",
        "heldout_motif_consistency_markdown": tables_dir / "heldout_motif_consistency.md",
        "heldout_motif_summary": tables_dir / "heldout_motif_summary.json",
        "heldout_motif_summary_markdown": tables_dir / "heldout_motif_summary.md",
        "paired_delta_bootstrap": tables_dir / "paired_delta_bootstrap.json",
        "paired_delta_bootstrap_markdown": tables_dir / "paired_delta_bootstrap.md",
    }

    heldout_rows = analysis["heldout_motif_consistency"]["rows"]
    heldout_summary = analysis["heldout_motif_consistency"]["summary_by_family"]
    bootstrap_rows = analysis["paired_delta_bootstrap"]["rows"]

    paths["summary"].write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["heldout_motif_consistency"].write_text(json.dumps(heldout_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["heldout_motif_consistency_markdown"].write_text(render_heldout_consistency_markdown(heldout_rows), encoding="utf-8")
    paths["heldout_motif_summary"].write_text(json.dumps(heldout_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["heldout_motif_summary_markdown"].write_text(render_heldout_summary_markdown(heldout_summary), encoding="utf-8")
    paths["paired_delta_bootstrap"].write_text(json.dumps(bootstrap_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["paired_delta_bootstrap_markdown"].write_text(render_paired_bootstrap_markdown(bootstrap_rows), encoding="utf-8")

    summary = {
        "analysis": "mechanism_diagnostic_p2_p3",
        "input": str(study_bundle_path),
        "output_dir": str(output_dir),
        "record_count": analysis["record_count"],
        "families": list(config.families),
        "split": config.split_name,
        "exclude_identity": config.exclude_identity,
        "paths": {key: str(path) for key, path in paths.items()},
    }
    if args.print_summary:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
