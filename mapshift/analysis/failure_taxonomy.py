"""Failure taxonomy and task-rejection categories for MapShift."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

FAILURE_TYPES = (
    "local_geometry_failure",
    "connectivity_update_failure",
    "dynamics_calibration_failure",
    "semantic_remapping_failure",
    "relocalization_failure",
)

TASK_REJECTION_REASONS = (
    "impossible_path",
    "trivial_goal_already_solved",
    "topology_no_reroute_required",
    "topology_no_connectivity_change",
    "semantic_no_counterfactual_change",
    "semantic_goal_collapses_to_start",
    "dynamics_no_transition_change",
    "metric_no_geometry_change",
    "uninformative_no_material_change",
)


@dataclass(frozen=True)
class TaskRejectionSummary:
    reason: str
    count: int
    families: tuple[str, ...]
    task_classes: tuple[str, ...]


def categorize_rejection(reason: str, family: str) -> str:
    """Map a task rejection reason into a failure-taxonomy category."""

    if reason in {"topology_no_reroute_required", "topology_no_connectivity_change"}:
        return "connectivity_update_failure" if family == "topology" else "relocalization_failure"
    if reason in {"dynamics_no_transition_change"}:
        return "dynamics_calibration_failure"
    if reason in {"semantic_no_counterfactual_change", "semantic_goal_collapses_to_start"}:
        return "semantic_remapping_failure"
    if reason in {"metric_no_geometry_change"}:
        return "local_geometry_failure"
    if reason in {"impossible_path", "trivial_goal_already_solved", "uninformative_no_material_change"}:
        return "relocalization_failure"
    return "relocalization_failure"


def summarize_rejection_log(rejections: list[dict[str, Any]]) -> dict[str, TaskRejectionSummary]:
    """Aggregate rejection logs into deterministic summaries."""

    grouped: dict[str, dict[str, Any]] = {}
    for entry in rejections:
        reason = str(entry["reason"])
        bucket = grouped.setdefault(reason, {"count": 0, "families": set(), "task_classes": set()})
        bucket["count"] += 1
        bucket["families"].add(str(entry["family"]))
        bucket["task_classes"].add(str(entry["task_class"]))

    return {
        reason: TaskRejectionSummary(
            reason=reason,
            count=data["count"],
            families=tuple(sorted(data["families"])),
            task_classes=tuple(sorted(data["task_classes"])),
        )
        for reason, data in sorted(grouped.items())
    }
