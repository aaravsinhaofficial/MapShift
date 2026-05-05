"""MiniGrid CPE transfer study for the MapShift intervention contract."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Sequence

from mapshift.envs.minigrid_port import MiniGridPortGenerator, build_minigrid_intervention, validate_minigrid_intervention_pair
from mapshift.envs.minigrid_port.state import GridPos, MiniGridPortEnvironment

FAMILIES = ("metric", "topology", "dynamics", "semantic")
METHODS = ("planning_solvability_reference", "stale_map_planner", "weak_local_planner", "belief_update_planner")
PROTOCOLS = ("cep", "same_environment", "no_exploration")
DEFAULT_SEVERITIES = (0, 1, 2, 3)


@dataclass(frozen=True)
class MiniGridCPEStudyResult:
    output_dir: Path
    artifact_paths: dict[str, str]
    summary: dict[str, Any]


def run_minigrid_cpe_study(
    *,
    output_dir: Path,
    seeds_per_motif: int = 4,
    motifs: Sequence[str] | None = None,
    severity_levels: Sequence[int] = DEFAULT_SEVERITIES,
    protocols: Sequence[str] = PROTOCOLS,
) -> MiniGridCPEStudyResult:
    """Run a compact MiniGrid CPE sanity-transfer study.

    The study intentionally uses deterministic planning probes rather than RL
    training. Its purpose is to verify that the CPE contract transfers to a
    MiniGrid-compatible substrate: matched intervention pairs, validators,
    reference solvability, a non-saturating weak floor, and protocol-sensitive
    conclusions about stale reuse versus explicit belief update.
    """

    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    raw_dir = output_dir / "raw"
    manifests_dir = output_dir / "manifests"
    for directory in (tables_dir, raw_dir, manifests_dir):
        directory.mkdir(parents=True, exist_ok=True)

    generator = MiniGridPortGenerator()
    motif_tags = tuple(motifs or generator.motif_tags)
    for motif in motif_tags:
        if motif not in generator.motif_tags:
            raise KeyError(f"Unsupported MiniGrid CPE motif: {motif}")
    for protocol in protocols:
        if protocol not in PROTOCOLS:
            raise KeyError(f"Unsupported MiniGrid CPE protocol: {protocol}")

    environments = []
    for motif_index, motif in enumerate(motif_tags):
        for offset in range(seeds_per_motif):
            seed = 1000 + motif_index * 100 + offset
            environments.append(generator.generate(seed=seed, motif_tag=motif))

    records: list[dict[str, Any]] = []
    pair_records: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    manifests.extend(result.manifest.to_dict() for result in environments)

    for result in environments:
        base = result.environment
        for family in FAMILIES:
            intervention = build_minigrid_intervention(family)
            for severity in severity_levels:
                transformed_result = intervention.apply(base, severity=int(severity), seed=base.seed + 17 * int(severity))
                transformed = transformed_result.environment
                manifests.append(transformed_result.manifest.to_dict())
                validation = validate_minigrid_intervention_pair(base, transformed, family=family, severity=int(severity))
                reference_solvable = bool(transformed.shortest_path())
                pair_records.append(
                    {
                        "environment_id": base.environment_id,
                        "motif_tag": base.motif_tag,
                        "split_name": base.split_name,
                        "seed": base.seed,
                        "family": family,
                        "severity": int(severity),
                        "transformed_environment_id": transformed.environment_id,
                        "validation_ok": validation.ok,
                        "validation_issues": list(validation.issues),
                        "reference_solvable": reference_solvable,
                        "base_path_length": len(base.shortest_path()),
                        "transformed_path_length": len(transformed.shortest_path()),
                        "metrics": validation.metrics,
                    }
                )
                for protocol in protocols:
                    for method in METHODS:
                        record = _evaluate_method(
                            base_environment=base,
                            transformed_environment=transformed,
                            family=family,
                            severity=int(severity),
                            protocol=protocol,
                            method=method,
                            validation_ok=validation.ok,
                            reference_solvable=reference_solvable,
                        )
                        records.append(record)

    health_summary = _health_summary(environments=environments, pair_records=pair_records, records=records)
    familywise_scores = _familywise_scores(records)
    protocol_sensitivity = _protocol_sensitivity(records)
    split_sensitivity = _split_sensitivity(records)
    bundle = {
        "study_name": "minigrid_cpe_sanity_transfer",
        "description": (
            "Compact MiniGrid-compatible CPE transfer study with matched intervention pairs, "
            "deterministic mechanism probes, validators, and protocol sensitivity summaries."
        ),
        "motifs": list(motif_tags),
        "seeds_per_motif": seeds_per_motif,
        "severity_levels": list(severity_levels),
        "families": list(FAMILIES),
        "protocols": list(protocols),
        "methods": list(METHODS),
        "health_summary": health_summary,
        "familywise_scores": familywise_scores,
        "protocol_sensitivity": protocol_sensitivity,
        "split_sensitivity": split_sensitivity,
    }

    artifact_paths = {
        "study_bundle": str(output_dir / "study_bundle.json"),
        "health_summary_json": str(tables_dir / "health_summary.json"),
        "health_summary_md": str(tables_dir / "health_summary.md"),
        "familywise_scores_json": str(tables_dir / "familywise_scores.json"),
        "familywise_scores_md": str(tables_dir / "familywise_scores.md"),
        "protocol_sensitivity_json": str(tables_dir / "protocol_sensitivity.json"),
        "protocol_sensitivity_md": str(tables_dir / "protocol_sensitivity.md"),
        "split_sensitivity_json": str(tables_dir / "split_sensitivity.json"),
        "split_sensitivity_md": str(tables_dir / "split_sensitivity.md"),
        "raw_records_jsonl": str(raw_dir / "records.jsonl"),
        "intervention_pairs_jsonl": str(raw_dir / "intervention_pairs.jsonl"),
        "manifest": str(manifests_dir / "minigrid_cpe_manifest.json"),
    }

    _write_json(Path(artifact_paths["study_bundle"]), bundle)
    _write_json(Path(artifact_paths["health_summary_json"]), health_summary)
    _write_json(Path(artifact_paths["familywise_scores_json"]), familywise_scores)
    _write_json(Path(artifact_paths["protocol_sensitivity_json"]), protocol_sensitivity)
    _write_json(Path(artifact_paths["split_sensitivity_json"]), split_sensitivity)
    _write_json(Path(artifact_paths["manifest"]), {"artifact_paths": artifact_paths, "manifests": manifests})
    Path(artifact_paths["health_summary_md"]).write_text(_health_markdown(health_summary), encoding="utf-8")
    Path(artifact_paths["familywise_scores_md"]).write_text(_familywise_markdown(familywise_scores), encoding="utf-8")
    Path(artifact_paths["protocol_sensitivity_md"]).write_text(_protocol_markdown(protocol_sensitivity), encoding="utf-8")
    Path(artifact_paths["split_sensitivity_md"]).write_text(_split_markdown(split_sensitivity), encoding="utf-8")
    Path(artifact_paths["raw_records_jsonl"]).write_text("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n", encoding="utf-8")
    Path(artifact_paths["intervention_pairs_jsonl"]).write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in pair_records) + "\n",
        encoding="utf-8",
    )

    return MiniGridCPEStudyResult(output_dir=output_dir, artifact_paths=artifact_paths, summary=bundle)


def _evaluate_method(
    *,
    base_environment: MiniGridPortEnvironment,
    transformed_environment: MiniGridPortEnvironment,
    family: str,
    severity: int,
    protocol: str,
    method: str,
    validation_ok: bool,
    reference_solvable: bool,
) -> dict[str, Any]:
    evaluation_environment = base_environment if protocol == "same_environment" else transformed_environment
    if method == "planning_solvability_reference":
        expected_goal = evaluation_environment.goal_pos
        planned_path = evaluation_environment.shortest_path(goal=expected_goal)
        access = "evaluation_environment_shortest_path"
    elif method == "stale_map_planner":
        if protocol == "no_exploration":
            expected_goal = evaluation_environment.goal_pos
            planned_path = _greedy_local_path(evaluation_environment, expected_goal)
            access = "no_exploration_local_greedy"
        else:
            expected_goal = evaluation_environment.goal_pos
            planned_path = base_environment.shortest_path(goal=base_environment.goal_pos)
            access = "base_environment_memory_only"
    elif method == "belief_update_planner":
        if protocol == "no_exploration":
            expected_goal = evaluation_environment.goal_pos
            planned_path = _greedy_local_path(evaluation_environment, expected_goal)
            access = "no_exploration_local_greedy"
        else:
            expected_goal = evaluation_environment.goal_pos
            planned_path = evaluation_environment.shortest_path(goal=expected_goal)
            access = "matched_pair_belief_update"
    elif method == "weak_local_planner":
        expected_goal = evaluation_environment.goal_pos
        planned_path = _greedy_local_path(evaluation_environment, expected_goal)
        access = "orthogonal_visible_path_heuristic"
    else:
        raise KeyError(f"Unsupported MiniGrid CPE method: {method}")

    score, valid_path, path_cost = _path_score(evaluation_environment, planned_path, expected_goal)
    if not validation_ok or not reference_solvable:
        score = 0.0
    return {
        "environment_id": base_environment.environment_id,
        "motif_tag": base_environment.motif_tag,
        "split_name": base_environment.split_name,
        "seed": base_environment.seed,
        "family": family,
        "severity": severity,
        "protocol": protocol,
        "method": method,
        "score": round(float(score), 6),
        "valid_path": valid_path,
        "planned_path_length": max(0, len(planned_path) - 1),
        "path_cost": round(float(path_cost), 6),
        "reference_solvable": reference_solvable,
        "validation_ok": validation_ok,
        "access_rule": access,
    }


def _path_score(environment: MiniGridPortEnvironment, path: Sequence[GridPos], expected_goal: GridPos) -> tuple[float, bool, float]:
    if not _path_is_valid(environment, path, expected_goal):
        return 0.0, False, 0.0
    reference_path = environment.shortest_path(goal=expected_goal)
    if not reference_path:
        return 0.0, False, 0.0
    path_steps = max(1, len(path) - 1)
    reference_steps = max(1, len(reference_path) - 1)
    efficiency = min(1.0, reference_steps / path_steps)
    metric_penalty = 1.0 / max(1.0, float(environment.movement_cost_scale))
    dynamics_penalty = max(0.0, 1.0 - float(environment.slip_probability) * min(1.0, path_steps / 12.0))
    score = efficiency * metric_penalty * dynamics_penalty
    path_cost = path_steps * float(environment.movement_cost_scale) * (1.0 + float(environment.slip_probability))
    return max(0.0, min(1.0, score)), True, path_cost


def _path_is_valid(environment: MiniGridPortEnvironment, path: Sequence[GridPos], expected_goal: GridPos) -> bool:
    if not path or path[0] != environment.start_pos or path[-1] != expected_goal:
        return False
    for pos in path:
        if not environment.passable(pos):
            return False
    for left, right in zip(path, path[1:]):
        if right not in environment.neighbors(left):
            return False
    return True


def _greedy_local_path(environment: MiniGridPortEnvironment, goal: GridPos) -> tuple[GridPos, ...]:
    candidates = (
        _orthogonal_path(environment.start_pos, goal, row_first=True),
        _orthogonal_path(environment.start_pos, goal, row_first=False),
    )
    for candidate in candidates:
        if _path_is_valid(environment, candidate, goal):
            return candidate
    return candidates[0]


def _orthogonal_path(start: GridPos, goal: GridPos, *, row_first: bool) -> tuple[GridPos, ...]:
    cursor = start
    path = [cursor]
    axes = (0, 1) if row_first else (1, 0)
    for axis in axes:
        while cursor[axis] != goal[axis]:
            delta = 1 if goal[axis] > cursor[axis] else -1
            if axis == 0:
                cursor = (cursor[0] + delta, cursor[1])
            else:
                cursor = (cursor[0], cursor[1] + delta)
            path.append(cursor)
    return tuple(path)


def _health_summary(*, environments: Sequence[Any], pair_records: Sequence[dict[str, Any]], records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    split_counts = Counter(result.environment.split_name for result in environments)
    failures = [pair for pair in pair_records if not pair["validation_ok"]]
    unsolved = [pair for pair in pair_records if not pair["reference_solvable"]]
    weak_scores = [
        float(record["score"])
        for record in records
        if record["method"] == "weak_local_planner" and record["protocol"] == "cep"
    ]
    weak_score = _mean(weak_scores)
    return {
        "environment_count": len(environments),
        "motif_count": len({result.environment.motif_tag for result in environments}),
        "split_counts": dict(sorted(split_counts.items())),
        "intervention_pair_count": len(pair_records),
        "task_rejections": len(failures) + len(unsolved),
        "validation_failures": len(failures),
        "reference_solvability": _mean(1.0 if pair["reference_solvable"] else 0.0 for pair in pair_records),
        "weak_baseline_score": weak_score,
        "weak_baseline_saturated": weak_score <= 0.05 or weak_score >= 0.95,
        "families": list(FAMILIES),
    }


def _familywise_scores(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for record in records:
        grouped[(record["protocol"], record["method"], record["family"])].append(float(record["score"]))
    rows = []
    for (protocol, method, family), scores in sorted(grouped.items()):
        rows.append(
            {
                "protocol": protocol,
                "method": method,
                "family": family,
                "records": len(scores),
                "mean_score": round(_mean(scores), 6),
            }
        )
    return rows


def _protocol_sensitivity(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for family in FAMILIES:
        cep_scores = _method_scores(records, protocol="cep", family=family)
        same_scores = _method_scores(records, protocol="same_environment", family=family)
        no_exploration_scores = _method_scores(records, protocol="no_exploration", family=family)
        cep_gap = cep_scores.get("belief_update_planner", 0.0) - cep_scores.get("stale_map_planner", 0.0)
        same_gap = same_scores.get("belief_update_planner", 0.0) - same_scores.get("stale_map_planner", 0.0)
        no_exploration_gap = no_exploration_scores.get("belief_update_planner", 0.0) - no_exploration_scores.get("stale_map_planner", 0.0)
        same_order = _rank_order(same_scores)
        cep_order = _rank_order(cep_scores)
        rows.append(
            {
                "family": family,
                "same_environment_belief_update_minus_stale": round(same_gap, 6),
                "cep_belief_update_minus_stale": round(cep_gap, 6),
                "protocol_delta": round(cep_gap - same_gap, 6),
                "no_exploration_belief_update_minus_stale": round(no_exploration_gap, 6),
                "same_environment_order": same_order,
                "cep_order": cep_order,
                "kendall_tau_same_environment_vs_cep": round(_kendall_tau(same_order, cep_order), 6),
                "rank_reversals_same_environment_vs_cep": _rank_reversal_count(same_order, cep_order),
            }
        )
    return rows


def _split_sensitivity(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for split_name in sorted({record["split_name"] for record in records}):
        for family in ("topology", "semantic"):
            split_records = [record for record in records if record["split_name"] == split_name]
            cep_scores = _method_scores(split_records, protocol="cep", family=family)
            same_scores = _method_scores(split_records, protocol="same_environment", family=family)
            rows.append(
                {
                    "split_name": split_name,
                    "family": family,
                    "cep_belief_update_minus_stale": round(
                        cep_scores.get("belief_update_planner", 0.0) - cep_scores.get("stale_map_planner", 0.0),
                        6,
                    ),
                    "same_environment_belief_update_minus_stale": round(
                        same_scores.get("belief_update_planner", 0.0) - same_scores.get("stale_map_planner", 0.0),
                        6,
                    ),
                }
            )
    return rows


def _method_scores(records: Sequence[dict[str, Any]], *, protocol: str, family: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if record["protocol"] == protocol and record["family"] == family:
            grouped[record["method"]].append(float(record["score"]))
    return {method: _mean(scores) for method, scores in grouped.items()}


def _rank_order(scores: dict[str, float]) -> list[str]:
    return [method for method, _score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]


def _kendall_tau(order_a: Sequence[str], order_b: Sequence[str]) -> float:
    shared = [method for method in order_a if method in set(order_b)]
    pairs = list(combinations(shared, 2))
    if not pairs:
        return 0.0
    pos_a = {method: index for index, method in enumerate(order_a)}
    pos_b = {method: index for index, method in enumerate(order_b)}
    concordant = 0
    discordant = 0
    for left, right in pairs:
        product = (pos_a[left] - pos_a[right]) * (pos_b[left] - pos_b[right])
        if product > 0:
            concordant += 1
        elif product < 0:
            discordant += 1
    return (concordant - discordant) / len(pairs)


def _rank_reversal_count(order_a: Sequence[str], order_b: Sequence[str]) -> int:
    shared = [method for method in order_a if method in set(order_b)]
    pos_a = {method: index for index, method in enumerate(order_a)}
    pos_b = {method: index for index, method in enumerate(order_b)}
    count = 0
    for left, right in combinations(shared, 2):
        if (pos_a[left] - pos_a[right]) * (pos_b[left] - pos_b[right]) < 0:
            count += 1
    return count


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _health_markdown(summary: dict[str, Any]) -> str:
    rows = [
        ("Environment count", summary["environment_count"]),
        ("Motif count", summary["motif_count"]),
        ("Split counts", json.dumps(summary["split_counts"], sort_keys=True)),
        ("Intervention pairs", summary["intervention_pair_count"]),
        ("Task rejections", summary["task_rejections"]),
        ("Validator failures", summary["validation_failures"]),
        ("Reference solvability", f"{summary['reference_solvability']:.3f}"),
        ("Weak-baseline score", f"{summary['weak_baseline_score']:.3f}"),
        ("Weak-baseline saturated", summary["weak_baseline_saturated"]),
    ]
    return _markdown_table(["Check", "Value"], rows)


def _familywise_markdown(rows: Sequence[dict[str, Any]]) -> str:
    return _markdown_table(
        ["Protocol", "Method", "Family", "Records", "Mean score"],
        [
            (row["protocol"], row["method"], row["family"], row["records"], f"{row['mean_score']:.3f}")
            for row in rows
        ],
    )


def _protocol_markdown(rows: Sequence[dict[str, Any]]) -> str:
    return _markdown_table(
        [
            "Family",
            "BU-STM same-env",
            "BU-STM CPE",
            "Protocol delta",
            "Tau",
            "Rank reversals",
        ],
        [
            (
                row["family"],
                f"{row['same_environment_belief_update_minus_stale']:.3f}",
                f"{row['cep_belief_update_minus_stale']:.3f}",
                f"{row['protocol_delta']:.3f}",
                f"{row['kendall_tau_same_environment_vs_cep']:.3f}",
                row["rank_reversals_same_environment_vs_cep"],
            )
            for row in rows
        ],
    )


def _split_markdown(rows: Sequence[dict[str, Any]]) -> str:
    return _markdown_table(
        ["Split", "Family", "BU-STM CPE", "BU-STM same-env"],
        [
            (
                row["split_name"],
                row["family"],
                f"{row['cep_belief_update_minus_stale']:.3f}",
                f"{row['same_environment_belief_update_minus_stale']:.3f}",
            )
            for row in rows
        ],
    )


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines) + "\n"
