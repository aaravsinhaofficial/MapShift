#!/usr/bin/env python3
"""Build a MapShift release artifact directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
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


def _configure_build_logging(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    return logging.getLogger("mapshift.build_benchmark")


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
                "- `configs/`: release and study configs used for this build.",
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
    parser.add_argument("--log-file", default="", help="Build log path. Defaults to <output-dir>/logs/build_benchmark.log.")
    parser.add_argument("--samples-per-motif", type=int, default=None)
    parser.add_argument("--task-samples-per-class", type=int, default=None)
    parser.add_argument("--min-cell-coverage", type=int, default=1)
    parser.add_argument("--print-summary", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    bundle = load_release_bundle(args.config)
    if args.tier != "mapshift_2d":
        raise ValueError("The primary artifact builder only supports --tier mapshift_2d.")

    study_config_path = Path(args.study_config).resolve()
    study_config = load_mapshift_2d_study_config(study_config_path)
    samples_per_motif = args.samples_per_motif or study_config.sample_count_per_motif
    task_samples_per_class = args.task_samples_per_class or study_config.task_samples_per_class
    output_dir = Path(args.output_dir or f"outputs/releases/{bundle.root.release_name}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_file).resolve() if args.log_file else output_dir / "logs" / "build_benchmark.log"
    logger = _configure_build_logging(log_path)
    logger.info("Starting MapShift artifact build")
    logger.info("Release=%s benchmark_version=%s tier=%s", bundle.root.release_name, bundle.root.benchmark_version, args.tier)
    logger.info("Benchmark config: %s", Path(args.config).resolve())
    logger.info("Study config: %s", study_config_path)
    logger.info("Output directory: %s", output_dir)
    logger.info("Build log: %s", log_path)
    logger.info("MAPSHIFT_TORCH_DEVICE=%s", os.environ.get("MAPSHIFT_TORCH_DEVICE", "<unset>"))
    logger.info("MAPSHIFT_CHECKPOINT_DIR=%s", os.environ.get("MAPSHIFT_CHECKPOINT_DIR", "<unset>"))
    logger.info(
        "Study grid: samples_per_motif=%d task_samples_per_class=%d severity_levels=%s protocols=%s",
        samples_per_motif,
        task_samples_per_class,
        list(study_config.severity_levels),
        list(study_config.protocol_names),
    )

    logger.info("Copying benchmark and study configs")
    copied_configs = _copy_release_configs(bundle, study_config_path, output_dir)
    logger.info("Copied %d config artifacts", len(copied_configs))

    logger.info("Building canonical split manifests and leakage report")
    split_bundle = build_canonical_release_split_bundle(
        release_bundle=bundle,
        sample_count_per_motif=samples_per_motif,
        task_samples_per_class=task_samples_per_class,
    )
    split_paths = _write_split_artifacts(split_bundle, output_dir)
    logger.info(
        "Split manifests written: ok=%s fatal_leakage=%d diagnostic_warnings=%d benign=%d",
        split_bundle.ok,
        len(split_bundle.leakage_report.errors),
        len(split_bundle.leakage_report.warnings),
        len(split_bundle.leakage_report.benign),
    )

    logger.info("Writing intervention and task recipe artifacts")
    recipe_paths = _write_recipe_artifacts(split_bundle, output_dir)
    logger.info("Recipe artifacts written: %s", ", ".join(sorted(recipe_paths)))

    logger.info("Generating benchmark health report before model results")
    health_report = generate_mapshift_2d_benchmark_health_report(
        release_bundle=bundle,
        sample_count_per_motif=samples_per_motif,
        task_samples_per_class=task_samples_per_class,
        min_cell_coverage=args.min_cell_coverage,
    )
    health_path = output_dir / "health" / "benchmark_health.json"
    _write_json(health_path, health_report.to_dict())
    logger.info(
        "Benchmark health written: %s validator_failures=%d undercovered_cells=%d",
        health_path,
        health_report.validator_summary["failed_intervention_count"],
        len(health_report.task_coverage["undercovered_cells"]),
    )

    logger.info("Running MapShift-2D study; this is the long stage")
    study_bundle = run_mapshift_2d_study(study_config, release_bundle=bundle)
    cep_records = len(study_bundle.raw_reports["cep_report"]["records"])
    protocol_reports = study_bundle.raw_reports["protocol_comparison_report"]["protocol_reports"]
    protocol_record_counts = {
        name: len(report["records"])
        for name, report in sorted(protocol_reports.items())
    }
    logger.info("Study complete: cep_records=%d protocol_record_counts=%s", cep_records, protocol_record_counts)

    logger.info("Writing study bundle artifacts")
    study_paths = write_mapshift_2d_study_bundle(study_bundle, output_dir / "study")
    logger.info("Study bundle written: %s", study_paths["study_bundle"])

    logger.info("Copying paper-facing tables and figures")
    copied_study_paths = _copy_study_tables_and_figures(study_paths, output_dir)
    logger.info("Copied %d paper-facing JSON artifacts", len(copied_study_paths))

    logger.info("Rendering paper-ready markdown/SVG outputs")
    rendered_paper_paths = render_outputs(Path(study_paths["study_bundle"]), output_dir / "paper_outputs")
    logger.info("Rendered %d paper outputs", len(rendered_paper_paths))

    readme_path = _write_artifact_readme(output_dir, bundle.root.release_name, study_config_path)

    artifact_paths = {
        "artifact_readme": readme_path,
        "build_log": str(log_path),
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
    logger.info("Release manifest written: %s", manifest_path)

    summary = {
        "release_name": bundle.root.release_name,
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "build_log": str(log_path),
        "split_validation_ok": split_bundle.ok,
        "fatal_leakage_count": len(split_bundle.leakage_report.errors),
        "health_validator_failures": health_report.validator_summary["failed_intervention_count"],
        "study_bundle": study_paths["study_bundle"],
    }
    logger.info(
        "Build finished: split_validation_ok=%s fatal_leakage=%d health_validator_failures=%d",
        summary["split_validation_ok"],
        summary["fatal_leakage_count"],
        summary["health_validator_failures"],
    )
    if args.print_summary:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if split_bundle.ok and health_report.validator_summary["failed_intervention_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
