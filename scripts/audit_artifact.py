#!/usr/bin/env python3
"""Run reviewer-facing checks for the MapShift executable artifact."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "audit" / "mapshift_quick"
EXPECTED_CONTRACT = REPO_ROOT / "tests" / "fixtures" / "expected_artifact_contract.json"
SMOKE_STUDY_CONFIG = REPO_ROOT / "configs" / "analysis" / "mapshift_2d_full_study_smoke_v0_1.json"
RELEASE_CONFIG = REPO_ROOT / "configs" / "benchmark" / "release_v0_1.json"


def _run(command: list[str], *, cwd: Path = REPO_ROOT) -> None:
    printable = " ".join(command)
    print(f"\n[audit] $ {printable}")
    result = subprocess.run(command, cwd=cwd, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {printable}")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_path(path: Path) -> None:
    if not path.exists():
        raise AssertionError(f"Missing required artifact path: {path}")


def _check_release_artifact(output_dir: Path, contract: dict[str, Any]) -> None:
    for relative in contract["required_release_paths"]:
        _require_path(output_dir / relative)

    paper_outputs_dir = output_dir / "paper_outputs"
    for relative in contract["required_paper_outputs"]:
        _require_path(paper_outputs_dir / relative)

    manifest = _load_json(output_dir / "manifests" / "release_manifest.json")
    metadata = manifest.get("metadata", {})
    expected_release_name = contract["release_name"]
    expected_benchmark_version = contract["benchmark_version"]
    if manifest.get("benchmark_version") != expected_benchmark_version:
        raise AssertionError(
            f"Unexpected benchmark_version: {manifest.get('benchmark_version')} != {expected_benchmark_version}"
        )
    if metadata.get("release_name") != expected_release_name:
        raise AssertionError(f"Unexpected release_name: {metadata.get('release_name')} != {expected_release_name}")
    if not metadata.get("split_validation_ok"):
        raise AssertionError("Split validation failed in release manifest.")
    if metadata.get("fatal_leakage_count") != 0:
        raise AssertionError(f"Fatal leakage count is nonzero: {metadata.get('fatal_leakage_count')}")
    if metadata.get("health_validator_failures") != 0:
        raise AssertionError(f"Health validator failures are nonzero: {metadata.get('health_validator_failures')}")

    health = _load_json(output_dir / "health" / "benchmark_health.json")
    validator_summary = health.get("validator_summary", {})
    if validator_summary.get("failed_intervention_count") != 0:
        raise AssertionError("Benchmark health report has intervention validator failures.")
    if validator_summary.get("severity_monotonicity_failures"):
        raise AssertionError("Benchmark health report has severity monotonicity failures.")
    undercovered_cells = health.get("task_coverage", {}).get("undercovered_cells", [])
    if undercovered_cells:
        raise AssertionError(f"Benchmark health report has undercovered cells: {undercovered_cells[:3]}")

    cep_report = _load_json(output_dir / "study" / "raw" / "cep_report.json")
    observed_families = sorted({row["family"] for row in cep_report["records"]})
    observed_task_classes = sorted({row["task_class"] for row in cep_report["records"]})
    expected_families = sorted(contract.get("smoke_families", contract["families"]))
    if observed_families != expected_families:
        raise AssertionError(f"Unexpected families: {observed_families}")
    if observed_task_classes != sorted(contract["task_classes"]):
        raise AssertionError(f"Unexpected task classes: {observed_task_classes}")

    paper_manifest = _load_json(output_dir / "paper_outputs" / "paper_outputs_manifest.json")
    if len(paper_manifest) < 9:
        raise AssertionError("Rendered paper output manifest is unexpectedly small.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run the standard reviewer smoke artifact audit.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for generated audit artifacts.")
    parser.add_argument("--skip-tests", action="store_true", help="Skip the unittest step when CI already ran it.")
    parser.add_argument("--skip-build", action="store_true", help="Audit an existing output directory without rebuilding it.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    contract = _load_json(EXPECTED_CONTRACT)

    try:
        _run([sys.executable, "-c", "import mapshift; print(mapshift.__version__)"])
        _run([sys.executable, "scripts/validate_benchmark.py", "--tier", "mapshift_2d", str(RELEASE_CONFIG)])
        if not args.skip_tests:
            _run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"])
        if not args.skip_build:
            _run(
                [
                    sys.executable,
                    "scripts/build_benchmark.py",
                    "--tier",
                    "mapshift_2d",
                    "--study-config",
                    str(SMOKE_STUDY_CONFIG),
                    "--output-dir",
                    str(output_dir),
                    "--print-summary",
                ]
            )
        _check_release_artifact(output_dir, contract)
    except Exception as exc:
        print(f"\nArtifact audit failed: {exc}", file=sys.stderr)
        return 1

    print("\nArtifact audit passed")
    print(f"Audited output directory: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
