"""Canonical split builders and validation for MapShift."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from mapshift.analysis.severity import find_undercovered_cells, summarize_family_severity_counts
from mapshift.core.manifests import SplitManifest
from mapshift.core.schemas import ReleaseBundle
from mapshift.envs.map2d.generator import Map2DGenerator
from mapshift.interventions import build_intervention
from mapshift.metrics.statistics import summarize_numeric
from mapshift.splits.leakage_checks import LeakageReport, generate_leakage_report
from mapshift.splits.motifs import structural_signature_for_environment
from mapshift.tasks.samplers import TaskSampler, TaskSamplingRejected


@dataclass(frozen=True)
class CanonicalSplitBundle:
    """Deterministic split artifacts for one release bundle."""

    manifests: dict[str, SplitManifest]
    coverage_summary: dict[str, Any]
    leakage_report: LeakageReport
    validation_issues: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.validation_issues and self.leakage_report.ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "validation_issues": list(self.validation_issues),
            "coverage_summary": self.coverage_summary,
            "leakage_report": self.leakage_report.to_dict(),
            "manifests": {split_name: manifest.to_dict() for split_name, manifest in self.manifests.items()},
        }


def _config_hash(release_bundle: ReleaseBundle) -> str:
    blob = json.dumps(release_bundle.summary(), sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


def canonical_environment_seed(release_bundle: ReleaseBundle, motif: str, sample_index: int = 0) -> int:
    """Return the canonical deterministic environment seed for one motif/sample."""

    motif_index = release_bundle.env2d.motif_families.index(motif)
    return (motif_index + 1) * 100 + sample_index


def _split_motifs(release_bundle: ReleaseBundle) -> dict[str, tuple[str, ...]]:
    return {
        "train": release_bundle.env2d.splits.train_motifs,
        "val": release_bundle.env2d.splits.val_motifs,
        "test": release_bundle.env2d.splits.test_motifs,
    }


def _environment_entry(environment: Any, manifest: Any) -> dict[str, Any]:
    structural_signature = environment.metadata["structural_signature"]
    return {
        "artifact_id": manifest.artifact_id,
        "environment_id": environment.environment_id,
        "motif_tag": environment.motif_tag,
        "motif_family": environment.metadata.get("motif_family", ""),
        "motif_tags": list(environment.metadata.get("motif_tags", ())),
        "split_name": environment.split_name,
        "seed": environment.seed,
        "semantic_template_id": environment.metadata.get("semantic_template_id", ""),
        "goal_token_template_id": environment.metadata.get("goal_token_template_id", ""),
        "landmark_layout_template_id": environment.metadata.get("landmark_layout_template_id", ""),
        "connectivity_hash": structural_signature["connectivity_hash"],
        "geometry_hash": structural_signature["geometry_hash"],
        "normalized_structural_fingerprint": structural_signature["normalized_fingerprint"],
        "start_goal_distance": environment.shortest_path_length(environment.start_node_id, environment.goal_node_id),
    }


def _intervention_entry(split_name: str, manifest: Any) -> dict[str, Any]:
    metadata = manifest.metadata
    target_signature = metadata.get("target_structural_signature", {})
    return {
        "split_name": split_name,
        "base_environment_id": manifest.base_environment_id,
        "family": manifest.intervention_family,
        "severity": manifest.severity_level,
        "intervention_template_id": metadata.get("intervention_template_id", ""),
        "source_semantic_template_id": metadata.get("source_semantic_template_id", ""),
        "target_semantic_template_id": metadata.get("target_semantic_template_id", ""),
        "target_structural_fingerprint": target_signature.get("normalized_fingerprint", ""),
    }


def _task_entry(split_name: str, family: str, severity: int, manifest: Any) -> dict[str, Any]:
    metadata = manifest.metadata
    return {
        "split_name": split_name,
        "base_environment_id": manifest.base_environment_id,
        "intervened_environment_id": manifest.intervened_environment_id,
        "family": family,
        "severity": severity,
        "task_class": manifest.task_class,
        "task_type": manifest.task_type,
        "task_template_id": metadata.get("task_template_id", ""),
        "start_goal_template_id": metadata.get("start_goal_template_id", ""),
        "query_template_id": metadata.get("query_template_id", ""),
        "budget_template_id": metadata.get("budget_template_id", ""),
    }


def _count_by_key(entries: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        value = entry.get(key)
        if isinstance(value, list):
            for item in value:
                counts[str(item)] = counts.get(str(item), 0) + 1
        elif value not in (None, ""):
            counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items()))


def _summarize_split_coverage(
    release_bundle: ReleaseBundle,
    split_name: str,
    environment_entries: list[dict[str, Any]],
    intervention_entries: list[dict[str, Any]],
    task_entries: list[dict[str, Any]],
    rejection_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    task_records = [{**entry, "count": 1} for entry in task_entries]
    intervention_records = [{**entry, "count": 1} for entry in intervention_entries]
    expected_intervention_values = {
        "family": release_bundle.interventions.canonical_family_order,
        "severity": tuple(range(4)),
    }
    expected_task_values = {
        "family": release_bundle.interventions.canonical_family_order,
        "severity": tuple(range(4)),
        "task_class": tuple(class_name for class_name, config in release_bundle.tasks.classes.items() if config.enabled),
    }
    intervention_table = summarize_family_severity_counts(intervention_records)
    task_table = summarize_family_severity_counts(task_records)
    missing_intervention_cells = find_undercovered_cells(
        intervention_records,
        ("family", "severity"),
        min_coverage=max(1, len(environment_entries)),
        expected_values=expected_intervention_values,
    )
    missing_task_cells = find_undercovered_cells(
        task_records,
        ("family", "severity", "task_class"),
        min_coverage=1,
        expected_values=expected_task_values,
    )
    return {
        "split_name": split_name,
        "environment_count": len(environment_entries),
        "motif_family_counts": _count_by_key(environment_entries, "motif_family"),
        "motif_tag_counts": _count_by_key(environment_entries, "motif_tag"),
        "motif_structural_tag_counts": _count_by_key(environment_entries, "motif_tags"),
        "semantic_template_counts": _count_by_key(environment_entries, "semantic_template_id"),
        "goal_token_template_counts": _count_by_key(environment_entries, "goal_token_template_id"),
        "landmark_template_counts": _count_by_key(environment_entries, "landmark_layout_template_id"),
        "task_class_counts": _count_by_key(task_entries, "task_class"),
        "task_type_counts": _count_by_key(task_entries, "task_type"),
        "task_template_counts": _count_by_key(task_entries, "task_template_id"),
        "intervention_family_severity_counts": intervention_table,
        "task_family_severity_counts": task_table,
        "path_length_summary": summarize_numeric(
            [entry["start_goal_distance"] for entry in environment_entries if entry["start_goal_distance"] is not None]
        ).to_dict(),
        "task_template_distance_summary": summarize_numeric(
            [
                manifest_distance
                for manifest_distance in (
                    entry.get("distance_steps") for entry in task_entries
                )
                if manifest_distance not in (None, "")
            ]
        ).to_dict(),
        "missing_intervention_cells": missing_intervention_cells,
        "missing_task_cells": missing_task_cells,
        "task_rejections_by_reason": _count_by_key(rejection_entries, "reason"),
    }


def _validate_split_bundle(release_bundle: ReleaseBundle, manifests: dict[str, SplitManifest], leakage_report: LeakageReport) -> list[str]:
    issues: list[str] = []
    expected_motifs = _split_motifs(release_bundle)

    for split_name, manifest in manifests.items():
        actual_motifs = sorted({str(entry["motif_tag"]) for entry in manifest.metadata.get("environment_entries", [])})
        if actual_motifs != sorted(expected_motifs[split_name]):
            issues.append(f"{split_name} split motifs do not match config: expected {sorted(expected_motifs[split_name])}, got {actual_motifs}")
        if not manifest.environment_ids:
            issues.append(f"{split_name} split has no environments")
        coverage = manifest.metadata.get("coverage_summary", {})
        if coverage.get("missing_intervention_cells"):
            issues.append(f"{split_name} split is missing intervention coverage cells")

    for finding in leakage_report.errors:
        issues.append(
            f"split leakage error {finding.category} between {finding.left_split} and {finding.right_split}: {', '.join(finding.overlap_keys[:3])}"
        )

    return issues


def validate_release_split_artifacts(
    release_bundle: ReleaseBundle,
    manifests: dict[str, SplitManifest],
    leakage_report: LeakageReport,
) -> list[str]:
    """Validate split manifests and leakage report against one release bundle."""

    return _validate_split_bundle(release_bundle, manifests, leakage_report)


def validate_canonical_release_split_bundle(split_bundle: CanonicalSplitBundle) -> list[str]:
    """Return validation issues for an already-built split bundle."""

    return list(split_bundle.validation_issues)


def build_canonical_release_split_bundle(
    release_bundle: ReleaseBundle,
    sample_count_per_motif: int = 1,
    task_samples_per_class: int = 1,
) -> CanonicalSplitBundle:
    """Build deterministic train/val/test split artifacts for a release bundle."""

    generator = Map2DGenerator(release_bundle.env2d)
    split_manifests: dict[str, SplitManifest] = {}
    environment_entries_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    intervention_entries_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    task_entries_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    rejection_entries_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}

    for split_name, motifs in _split_motifs(release_bundle).items():
        sampler = TaskSampler(release_bundle.tasks)
        environment_entries: list[dict[str, Any]] = []
        intervention_entries: list[dict[str, Any]] = []
        task_entries: list[dict[str, Any]] = []
        rejection_entries: list[dict[str, Any]] = []

        for motif in motifs:
            for sample_index in range(sample_count_per_motif):
                seed = canonical_environment_seed(release_bundle, motif, sample_index)
                base_result = generator.generate(seed=seed, motif_tag=motif)
                base_environment = generator.replay_from_manifest(base_result.manifest)
                environment_entries.append(_environment_entry(base_environment, base_result.manifest))

                for family in release_bundle.interventions.canonical_family_order:
                    intervention = build_intervention(family, release_bundle.interventions.families[family])
                    for severity in range(4):
                        intervention_result = intervention.apply(base_environment, severity=severity, seed=seed + severity)
                        transformed_environment = base_environment.__class__.from_manifest_metadata(intervention_result.manifest.metadata)
                        intervention_entries.append(_intervention_entry(split_name, intervention_result.manifest))

                        for task_class, class_config in release_bundle.tasks.classes.items():
                            if not class_config.enabled:
                                continue
                            for task_index in range(task_samples_per_class):
                                task_seed = seed * 1000 + severity * 100 + task_index * 10 + len(task_class)
                                try:
                                    task_result = sampler.sample(
                                        base_environment=base_environment,
                                        intervened_environment=transformed_environment,
                                        family=family,
                                        seed=task_seed,
                                        task_class=task_class,
                                    )
                                except TaskSamplingRejected as exc:
                                    rejection_entries.append(
                                        {
                                            "split_name": split_name,
                                            "family": family,
                                            "severity": severity,
                                            "task_class": task_class,
                                            "reason": exc.rejection.reason,
                                        }
                                    )
                                    continue

                                task_entry = _task_entry(split_name, family, severity, task_result.manifest)
                                task_entry["distance_steps"] = task_result.manifest.metadata.get("distance_steps")
                                task_entries.append(task_entry)

        coverage_summary = _summarize_split_coverage(
            release_bundle=release_bundle,
            split_name=split_name,
            environment_entries=environment_entries,
            intervention_entries=intervention_entries,
            task_entries=task_entries,
            rejection_entries=rejection_entries,
        )
        split_manifest = SplitManifest(
            artifact_id=f"split-{split_name}-{release_bundle.root.release_name}",
            artifact_type="split",
            benchmark_version=release_bundle.root.benchmark_version,
            code_version="split-control-v1",
            config_hash=_config_hash(release_bundle),
            created_at=f"canonical-{release_bundle.root.release_name}-{split_name}",
            split_name=split_name,
            tier="mapshift_2d",
            environment_ids=[entry["environment_id"] for entry in environment_entries],
            release_name=release_bundle.root.release_name,
            metadata={
                "motifs": list(motifs),
                "environment_entries": environment_entries,
                "intervention_entries": intervention_entries,
                "task_entries": task_entries,
                "rejection_entries": rejection_entries,
                "coverage_summary": coverage_summary,
            },
        )
        split_manifests[split_name] = split_manifest
        environment_entries_by_split[split_name] = environment_entries
        intervention_entries_by_split[split_name] = intervention_entries
        task_entries_by_split[split_name] = task_entries
        rejection_entries_by_split[split_name] = rejection_entries

    leakage_report = generate_leakage_report(
        environment_entries_by_split=environment_entries_by_split,
        task_entries_by_split=task_entries_by_split,
        intervention_entries_by_split=intervention_entries_by_split,
    )
    coverage_summary = {
        split_name: manifest.metadata["coverage_summary"]
        for split_name, manifest in sorted(split_manifests.items())
    }
    coverage_summary["leakage_report"] = leakage_report.to_dict()
    validation_issues = tuple(_validate_split_bundle(release_bundle, split_manifests, leakage_report))
    return CanonicalSplitBundle(
        manifests=split_manifests,
        coverage_summary=coverage_summary,
        leakage_report=leakage_report,
        validation_issues=validation_issues,
    )
