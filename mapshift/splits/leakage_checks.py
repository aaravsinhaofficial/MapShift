"""Leakage checks for canonical MapShift release splits."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def disjoint_overlap(left: set[str], right: set[str]) -> set[str]:
    """Return the overlap between two sets."""

    return left & right


@dataclass(frozen=True)
class LeakageFinding:
    """Machine-readable description of one split-leakage finding."""

    category: str
    severity: str
    left_split: str
    right_split: str
    overlap_keys: tuple[str, ...]
    example_pairs: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LeakageReport:
    """Aggregated leakage report across train/val/test."""

    findings: tuple[LeakageFinding, ...]

    @property
    def errors(self) -> tuple[LeakageFinding, ...]:
        return tuple(finding for finding in self.findings if finding.severity == "error")

    @property
    def warnings(self) -> tuple[LeakageFinding, ...]:
        return tuple(finding for finding in self.findings if finding.severity == "warning")

    @property
    def benign(self) -> tuple[LeakageFinding, ...]:
        return tuple(finding for finding in self.findings if finding.severity == "benign")

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "benign_count": len(self.benign),
            "findings": [finding.to_dict() for finding in self.findings],
        }


def _entry_key_map(entries: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    mapping: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        value = str(entry.get(key, "")).strip()
        if not value:
            continue
        mapping.setdefault(value, []).append(entry)
    return mapping


def _examples_for_overlap(
    left_entries: list[dict[str, Any]],
    right_entries: list[dict[str, Any]],
    keys: list[str],
) -> tuple[dict[str, Any], ...]:
    left_by_key = _entry_key_map(left_entries, "_lookup_key")
    right_by_key = _entry_key_map(right_entries, "_lookup_key")
    examples = []
    for key in keys[:3]:
        left_example = dict(left_by_key[key][0])
        right_example = dict(right_by_key[key][0])
        left_example.pop("_lookup_key", None)
        right_example.pop("_lookup_key", None)
        examples.append({"key": key, "left": left_example, "right": right_example})
    return tuple(examples)


def _make_finding(
    category: str,
    severity: str,
    left_split: str,
    right_split: str,
    left_entries: list[dict[str, Any]],
    right_entries: list[dict[str, Any]],
    key_field: str,
    rationale: str = "",
) -> LeakageFinding | None:
    left_with_lookup = [entry | {"_lookup_key": str(entry.get(key_field, "")).strip()} for entry in left_entries]
    right_with_lookup = [entry | {"_lookup_key": str(entry.get(key_field, "")).strip()} for entry in right_entries]
    overlap_keys = sorted(
        key
        for key in disjoint_overlap(set(_entry_key_map(left_with_lookup, "_lookup_key")), set(_entry_key_map(right_with_lookup, "_lookup_key")))
        if key
    )
    if not overlap_keys:
        return None
    return LeakageFinding(
        category=category,
        severity=severity,
        left_split=left_split,
        right_split=right_split,
        overlap_keys=tuple(overlap_keys),
        example_pairs=_examples_for_overlap(left_with_lookup, right_with_lookup, overlap_keys),
        rationale=rationale,
    )


def generate_leakage_report(
    environment_entries_by_split: dict[str, list[dict[str, Any]]],
    task_entries_by_split: dict[str, list[dict[str, Any]]] | None = None,
    intervention_entries_by_split: dict[str, list[dict[str, Any]]] | None = None,
) -> LeakageReport:
    """Check split leakage across environment, task, and intervention artifacts."""

    task_entries_by_split = task_entries_by_split or {}
    intervention_entries_by_split = intervention_entries_by_split or {}
    split_names = sorted(environment_entries_by_split)
    findings: list[LeakageFinding] = []

    for index, left_split in enumerate(split_names):
        for right_split in split_names[index + 1 :]:
            left_envs = environment_entries_by_split.get(left_split, [])
            right_envs = environment_entries_by_split.get(right_split, [])
            left_tasks = task_entries_by_split.get(left_split, [])
            right_tasks = task_entries_by_split.get(right_split, [])
            left_interventions = intervention_entries_by_split.get(left_split, [])
            right_interventions = intervention_entries_by_split.get(right_split, [])

            for category, severity, key_field, left_entries, right_entries, rationale in (
                ("motif_instance_overlap", "error", "motif_tag", left_envs, right_envs, ""),
                ("structural_exact_overlap", "error", "geometry_hash", left_envs, right_envs, ""),
                ("structural_near_overlap", "warning", "normalized_structural_fingerprint", left_envs, right_envs, "Near-overlap is a diagnostic warning; exact geometry/hash overlap is treated as fatal."),
                ("semantic_template_overlap", "error", "semantic_template_id", left_envs, right_envs, ""),
                ("goal_token_template_overlap", "benign", "goal_token_template_id", left_envs, right_envs, "Goal-token template ids describe reusable query vocabulary, not shared map instances."),
                ("landmark_template_overlap", "benign", "landmark_layout_template_id", left_envs, right_envs, "Landmark layout templates are reusable semantic recipes; fatal leakage is checked by semantic_template_id."),
                ("task_template_overlap", "error", "task_template_id", left_tasks, right_tasks, ""),
                ("start_goal_template_overlap", "benign", "start_goal_template_id", left_tasks, right_tasks, "Start-goal templates are reusable query forms and are paired with disjoint motif instances."),
                ("query_template_overlap", "benign", "query_template_id", left_tasks, right_tasks, "Query templates are intentionally reusable across splits to hold language form fixed."),
                ("intervention_template_overlap", "error", "intervention_template_id", left_interventions, right_interventions, ""),
            ):
                finding = _make_finding(
                    category=category,
                    severity=severity,
                    left_split=left_split,
                    right_split=right_split,
                    left_entries=left_entries,
                    right_entries=right_entries,
                    key_field=key_field,
                    rationale=rationale,
                )
                if finding is not None:
                    findings.append(finding)

            left_families = {str(entry.get("motif_family", "")) for entry in left_envs if entry.get("motif_family")}
            right_families = {str(entry.get("motif_family", "")) for entry in right_envs if entry.get("motif_family")}
            family_overlap = sorted(disjoint_overlap(left_families, right_families))
            if family_overlap:
                findings.append(
                    LeakageFinding(
                        category="motif_family_overlap",
                        severity="benign",
                        left_split=left_split,
                        right_split=right_split,
                        overlap_keys=tuple(family_overlap),
                        rationale="Coarse motif-family labels can recur while concrete motif tags and structural hashes remain disjoint.",
                    )
                )

    return LeakageReport(findings=tuple(findings))
