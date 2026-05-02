#!/usr/bin/env python3
"""Build a frozen MapShift-2D release artifact directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mapshift.analysis.construct_validity import generate_mapshift_2d_benchmark_health_report
from mapshift.analysis.study import (
    load_mapshift_2d_study_config,
    run_mapshift_2d_study,
    write_mapshift_2d_study_bundle,
)
from mapshift.core.manifests import ArtifactManifest
from mapshift.core.schemas import load_release_bundle
from mapshift.splits.builders import build_canonical_release_split_bundle
from render_paper_outputs import render_outputs


DEFAULT_SMOKE_STUDY = "configs/analysis/mapshift_2d_full_study_smoke_v0_1.json"


def _json_hash(payload: Any) -> str:
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _copy_release_configs(bundle: Any, study_config_path: Path, output_dir: Path) -> dict[str, str]:
    config_dir = output_dir / "configs"
    copied: dict[str, str] = {}
    root_path = Path(bundle.root_path)
    study_payload = json.loads(study_config_path.read_text(encoding="utf-8"))
    destinations = {
        "benchmark": config_dir / "benchmark" / root_path.name,
        "env2d": config_dir / "env2d" / Path(bundle.root.config_refs.env2d).name,
        "env3d_prototype": config_dir / "env3d" / Path(bundle.root.config_refs.env3d).name,
        "interventions": config_dir / "interventions" / Path(bundle.root.config_refs.interventions).name,
        "tasks": config_dir / "tasks" / Path(bundle.root.config_refs.tasks).name,
        "baselines": config_dir / "baselines" / Path(bundle.root.config_refs.baselines).name,
        "analysis": config_dir / "analysis" / Path(bundle.root.config_refs.analysis).name,
        "study": config_dir / "analysis" / study_config_path.name,
    }
    sources = {
        "benchmark": root_path,
        "env2d": (root_path.parent / bundle.root.config_refs.env2d).resolve(),
        "env3d_prototype": (root_path.parent / bundle.root.config_refs.env3d).resolve(),
        "interventions": (root_path.parent / bundle.root.config_refs.interventions).resolve(),
        "tasks": (root_path.parent / bundle.root.config_refs.tasks).resolve(),
        "baselines": (root_path.parent / bundle.root.config_refs.baselines).resolve(),
        "analysis": (root_path.parent / bundle.root.config_refs.analysis).resolve(),
        "study": study_config_path.resolve(),
    }
    for name, source in sources.items():
        destinations[name].parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destinations[name])
        copied[name] = str(destinations[name])
    for index, item in enumerate(study_payload.get("baseline_run_configs", [])):
        source = (study_config_path.parent / str(item)).resolve()
        destination = config_dir / "calibration" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied[f"calibration_{index}_{source.stem}"] = str(destination)
    return copied


def _write_split_artifacts(split_bundle: Any, output_dir: Path) -> dict[str, str]:
    split_dir = output_dir / "splits"
    paths = {"split_bundle": str(split_dir / "split_bundle.json")}
    _write_json(split_dir / "split_bundle.json", split_bundle.to_dict())
    for split_name, manifest in sorted(split_bundle.manifests.items()):
        path = split_dir / f"{split_name}_manifest.json"
        _write_json(path, manifest.to_dict())
        paths[f"{split_name}_manifest"] = str(path)
    return paths


def _write_recipe_artifacts(split_bundle: Any, output_dir: Path) -> dict[str, str]:
    recipe_dir = output_dir / "recipes"
    intervention_entries = []
    task_entries = []
    rejection_entries = []
    for split_name, manifest in sorted(split_bundle.manifests.items()):
        for entry in manifest.metadata.get("intervention_entries", []):
            intervention_entries.append({"split_name": split_name, **entry})
        for entry in manifest.metadata.get("task_entries", []):
            task_entries.append({"split_name": split_name, **entry})
        for entry in manifest.metadata.get("rejection_entries", []):
            rejection_entries.append({"split_name": split_name, **entry})
    paths = {
        "intervention_recipes": str(recipe_dir / "intervention_recipes.json"),
        "task_recipes": str(recipe_dir / "task_recipes.json"),
        "task_rejections": str(recipe_dir / "task_rejections.json"),
    }
    _write_json(Path(paths["intervention_recipes"]), intervention_entries)
    _write_json(Path(paths["task_recipes"]), task_entries)
    _write_json(Path(paths["task_rejections"]), rejection_entries)
    return paths


def _copy_study_tables_and_figures(study_paths: dict[str, str], output_dir: Path) -> dict[str, str]:
    copied: dict[str, str] = {}
    for name, source_text in sorted(study_paths.items()):
        source = Path(source_text)
        if source.parent.name not in {"tables", "figures", "raw", "manifests"}:
            continue
        destination = output_dir / source.parent.name / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied[name] = str(destination)
    return copied


def _write_artifact_readme(output_dir: Path, release_name: str, study_config_path: Path) -> str:
    path = output_dir / "README.md"
    path.write_text(
        "\n".join(
            [
                f"# {release_name} Artifact",
                "",
                "This directory is generated by `scripts/build_benchmark.py --tier mapshift_2d`.",
                "",
                "Contents:",
                "- `configs/`: frozen release and study configs used for this build.",
                "- `splits/`: canonical motif split manifests and leakage report.",
                "- `recipes/`: intervention and task recipe manifests generated from code.",
                "- `health/`: benchmark health report generated before model results.",
                "- `study/`: raw CEP/protocol reports plus study tables and figure data.",
                "- `tables/` and `figures/`: paper-facing JSON outputs copied from the study bundle.",
                "- `manifests/release_manifest.json`: top-level provenance for this artifact.",
                "",
                "Reviewer smoke command:",
                "`python3 scripts/build_benchmark.py --tier mapshift_2d --study-config configs/analysis/mapshift_2d_full_study_smoke_v0_1.json --output-dir outputs/releases/mapshift_2d_v0_1_smoke`",
                "",
                "Full reproduction command:",
                "`python3 scripts/build_benchmark.py --tier mapshift_2d --study-config configs/analysis/mapshift_2d_full_study_v0_1.json --output-dir outputs/releases/mapshift_2d_v0_1_full`",
                "",
                f"Study config used for this build: `{study_config_path}`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return str(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="configs/benchmark/release_v0_1.json")
    parser.add_argument("--tier", choices=("mapshift_2d",), default="mapshift_2d")
    parser.add_argument("--study-config", default=DEFAULT_SMOKE_STUDY)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--samples-per-motif", type=int, default=None)
    parser.add_argument("--task-samples-per-class", type=int, default=None)
    parser.add_argument("--min-cell-coverage", type=int, default=1)
    parser.add_argument("--print-summary", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    bundle = load_release_bundle(args.config)
    if args.tier != "mapshift_2d":
        raise ValueError("The frozen v0.1 artifact builder only supports --tier mapshift_2d.")

    study_config_path = Path(args.study_config).resolve()
    study_config = load_mapshift_2d_study_config(study_config_path)
    samples_per_motif = args.samples_per_motif or study_config.sample_count_per_motif
    task_samples_per_class = args.task_samples_per_class or study_config.task_samples_per_class
    output_dir = Path(args.output_dir or f"outputs/releases/{bundle.root.release_name}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    copied_configs = _copy_release_configs(bundle, study_config_path, output_dir)
    split_bundle = build_canonical_release_split_bundle(
        release_bundle=bundle,
        sample_count_per_motif=samples_per_motif,
        task_samples_per_class=task_samples_per_class,
    )
    split_paths = _write_split_artifacts(split_bundle, output_dir)
    recipe_paths = _write_recipe_artifacts(split_bundle, output_dir)

    health_report = generate_mapshift_2d_benchmark_health_report(
        release_bundle=bundle,
        sample_count_per_motif=samples_per_motif,
        task_samples_per_class=task_samples_per_class,
        min_cell_coverage=args.min_cell_coverage,
    )
    health_path = output_dir / "health" / "benchmark_health.json"
    _write_json(health_path, health_report.to_dict())

    study_bundle = run_mapshift_2d_study(study_config, release_bundle=bundle)
    study_paths = write_mapshift_2d_study_bundle(study_bundle, output_dir / "study")
    copied_study_paths = _copy_study_tables_and_figures(study_paths, output_dir)
    rendered_paper_paths = render_outputs(Path(study_paths["study_bundle"]), output_dir / "paper_outputs")
    readme_path = _write_artifact_readme(output_dir, bundle.root.release_name, study_config_path)

    artifact_paths = {
        "artifact_readme": readme_path,
        "configs": copied_configs,
        "splits": split_paths,
        "recipes": recipe_paths,
        "benchmark_health": str(health_path),
        "study": study_paths,
        "paper_json_outputs": copied_study_paths,
        "rendered_paper_outputs": rendered_paper_paths,
    }
    manifest = ArtifactManifest(
        artifact_id=f"release-{bundle.root.release_name}",
        artifact_type="release_artifact",
        benchmark_version=bundle.root.benchmark_version,
        code_version="mapshift-2d-release-builder-v1",
        config_hash=_json_hash(bundle.summary() | study_config.to_dict()),
        metadata={
            "release_name": bundle.root.release_name,
            "tier": args.tier,
            "status": bundle.root.status,
            "samples_per_motif": samples_per_motif,
            "task_samples_per_class": task_samples_per_class,
            "split_validation_ok": split_bundle.ok,
            "fatal_leakage_count": len(split_bundle.leakage_report.errors),
            "diagnostic_leakage_warning_count": len(split_bundle.leakage_report.warnings),
            "benign_leakage_count": len(split_bundle.leakage_report.benign),
            "health_validator_failures": health_report.validator_summary["failed_intervention_count"],
            "artifact_paths": artifact_paths,
        },
    )
    manifest_path = output_dir / "manifests" / "release_manifest.json"
    _write_json(manifest_path, manifest.to_dict())

    summary = {
        "release_name": bundle.root.release_name,
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "split_validation_ok": split_bundle.ok,
        "fatal_leakage_count": len(split_bundle.leakage_report.errors),
        "health_validator_failures": health_report.validator_summary["failed_intervention_count"],
        "study_bundle": study_paths["study_bundle"],
    }
    if args.print_summary:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if split_bundle.ok and health_report.validator_summary["failed_intervention_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
