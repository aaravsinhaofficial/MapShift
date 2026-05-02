#!/usr/bin/env python3
"""Render paper-facing MapShift-2D tables and SVG figures from saved JSON outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _write(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def _bootstrap_ci_lookup(cep_report: dict[str, Any], metric_name: str) -> dict[tuple[str, str, str], tuple[float, float]]:
    lookup: dict[tuple[str, str, str], tuple[float, float]] = {}
    for row in cep_report.get("bootstrap_summary", {}).get("familywise_main_results", []):
        if row.get("metric_name") != metric_name:
            continue
        group = row.get("group", {})
        summary = row.get("summary", {})
        key = (str(group.get("protocol_name")), str(group.get("baseline_name")), str(group.get("family")))
        lookup[key] = (float(summary.get("lower", 0.0)), float(summary.get("upper", 0.0)))
    return lookup


def _render_main_results_table(bundle: dict[str, Any]) -> str:
    cep = bundle["raw_reports"]["cep_report"]
    ci_lookup = _bootstrap_ci_lookup(cep, "family_primary_score")
    rows = []
    for row in cep["familywise_summary"]["rows"]:
        key = (str(row["protocol_name"]), str(row["baseline_name"]), str(row["family"]))
        lower, upper = ci_lookup.get(key, (None, None))
        rows.append(
            [
                row["baseline_name"],
                row["family"],
                row["episode_count"],
                row["family_primary_score"],
                "" if lower is None else f"[{lower:.3f}, {upper:.3f}]",
            ]
        )
    return _markdown_table(["Method", "Family", "Episodes", "Primary Score", "95% Bootstrap CI"], rows)


def _render_health_table(bundle: dict[str, Any]) -> str:
    health = bundle["benchmark_health"]
    split_leakage = health["split_health"]["canonical_split_leakage_report"]
    rows = [
        ["2D environments", health["environment_health"]["environment_count"]],
        ["Motif splits", json.dumps(health["environment_health"]["split_counts"], sort_keys=True)],
        ["Task count", health["task_difficulty"]["task_count"]],
        ["Task rejections", health["rejection_statistics"]["total_rejections"]],
        ["Oracle solvability", health["task_difficulty"]["oracle_solvability"]],
        ["Weak baseline estimate", health["task_difficulty"]["heuristic_success_estimate"]],
        ["Fatal leakage", split_leakage["error_count"]],
        ["Benign leakage warnings", split_leakage["warning_count"]],
        ["Intervention validator failures", health["validator_summary"]["failed_intervention_count"]],
        ["Severity monotonicity failures", len(health["validator_summary"]["severity_monotonicity_failures"])],
    ]
    return _markdown_table(["Check", "Value"], rows)


def _render_baseline_metadata_table(bundle: dict[str, Any]) -> str:
    metadata = bundle["raw_reports"]["cep_report"]["baseline_metadata"]
    rows = []
    for name, item in sorted(metadata.items()):
        rows.append(
            [
                name,
                item.get("category", ""),
                item.get("parameter_count_min", item.get("parameter_count", 0)),
                item.get("parameter_count_max", item.get("parameter_count", 0)),
                item.get("trainable_parameter_count_min", item.get("trainable_parameter_count", 0)),
                ", ".join(str(seed) for seed in item.get("seeds", [])),
                item.get("run_count", 1),
                item.get("implementation_kind", ""),
            ]
        )
    return _markdown_table(
        ["Method", "Category", "Params Min", "Params Max", "Trainable Min", "Seeds", "Runs", "Implementation"],
        rows,
    )


def _render_protocol_sensitivity_table(bundle: dict[str, Any]) -> str:
    protocol = bundle["protocol_sensitivity"]
    rows = []
    for name, comparison in sorted(protocol["pairwise_comparisons"].items()):
        rows.append(
            [
                name,
                comparison.get("kendall_tau", 0.0),
                comparison.get("best_method_changes", False),
                len(comparison.get("rank_reversals", [])),
                len(comparison.get("family_rank_changes", [])),
            ]
        )
    pooled = protocol.get("pooled_vs_familywise_reporting", {})
    if pooled:
        rows.append(
            [
                "pooled_vs_familywise_reporting",
                pooled.get("kendall_tau", 0.0),
                pooled.get("best_method_changes", False),
                "",
                len(pooled.get("disagreements", [])),
            ]
        )
    return _markdown_table(["Comparison", "Kendall Tau", "Best Changes", "Rank Reversals", "Family Changes"], rows)


def _svg(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<style>text{font-family:Arial,sans-serif;fill:#1f2933} .small{font-size:11px} '
        '.label{font-size:13px;font-weight:700} .axis{stroke:#475569;stroke-width:1} '
        '.box{fill:#f8fafc;stroke:#334155;stroke-width:1.2} .muted{fill:#64748b}</style>\n'
        f"{body}\n</svg>\n"
    )


def _cep_protocol_svg() -> str:
    labels = ["Generate", "Intervene", "Explore", "Evaluate", "Analyze"]
    details = ["seeded 2D map", "one family, severity s", "reward-free T=800", "post-shift tasks", "family-wise CIs"]
    body = ['<rect width="900" height="170" fill="#ffffff"/>']
    x = 30
    for index, label in enumerate(labels):
        body.append(f'<rect class="box" x="{x}" y="45" width="145" height="70" rx="6"/>')
        body.append(f'<text class="label" x="{x + 72}" y="74" text-anchor="middle">{label}</text>')
        body.append(f'<text class="small muted" x="{x + 72}" y="96" text-anchor="middle">{details[index]}</text>')
        if index < len(labels) - 1:
            body.append(f'<line class="axis" x1="{x + 150}" y1="80" x2="{x + 195}" y2="80"/>')
            body.append(f'<polygon points="{x + 195},80 {x + 185},74 {x + 185},86" fill="#475569"/>')
        x += 185
    return _svg(900, 170, "\n".join(body))


def _intervention_examples_svg() -> str:
    families = [
        ("Metric", "#0f766e", "scale / odometry"),
        ("Topology", "#b91c1c", "blocked corridor"),
        ("Dynamics", "#7c3aed", "friction / lag"),
        ("Semantic", "#0369a1", "token remap"),
    ]
    body = ['<rect width="900" height="260" fill="#ffffff"/>']
    for index, (name, color, subtitle) in enumerate(families):
        ox = 30 + index * 215
        body.append(f'<text class="label" x="{ox}" y="28">{name}</text>')
        body.append(f'<text class="small muted" x="{ox}" y="45">{subtitle}</text>')
        for row in range(5):
            for col in range(5):
                fill = "#f8fafc"
                if row in {0, 4} or col in {0, 4}:
                    fill = "#dbe4ee"
                if name == "Topology" and row == 2 and col in {2, 3}:
                    fill = color
                if name == "Semantic" and row == 1 and col == 3:
                    fill = color
                if name == "Dynamics" and row == col:
                    fill = "#ede9fe"
                if name == "Metric" and row == 3 and col in {1, 2, 3}:
                    fill = "#ccfbf1"
                body.append(
                    f'<rect x="{ox + col * 32}" y="{70 + row * 32}" width="30" height="30" fill="{fill}" stroke="#94a3b8"/>'
                )
        body.append(f'<circle cx="{ox + 48}" cy="118" r="8" fill="#111827"/>')
        body.append(f'<circle cx="{ox + 112}" cy="182" r="8" fill="{color}"/>')
    return _svg(900, 260, "\n".join(body))


def _bar_chart_svg(rows: list[dict[str, Any]], title: str) -> str:
    families = sorted({str(row["family"]) for row in rows})
    methods = sorted({str(row["baseline_name"]) for row in rows})
    width = 980
    height = 90 + len(methods) * 24 + len(families) * 145
    colors = ["#2563eb", "#059669", "#d97706", "#7c3aed", "#dc2626", "#0891b2", "#4b5563"]
    body = [f'<rect width="{width}" height="{height}" fill="#ffffff"/>', f'<text class="label" x="30" y="30">{title}</text>']
    y = 70
    for family in families:
        body.append(f'<text class="label" x="30" y="{y}">{family}</text>')
        y += 20
        family_rows = [row for row in rows if row["family"] == family]
        for index, row in enumerate(sorted(family_rows, key=lambda item: str(item["baseline_name"]))):
            score = max(0.0, min(1.0, float(row["family_primary_score"])))
            bar_width = int(score * 360)
            color = colors[methods.index(str(row["baseline_name"])) % len(colors)]
            body.append(f'<text class="small" x="40" y="{y + 13}">{row["baseline_name"]}</text>')
            body.append(f'<rect x="310" y="{y}" width="360" height="15" fill="#e2e8f0"/>')
            body.append(f'<rect x="310" y="{y}" width="{bar_width}" height="15" fill="{color}"/>')
            body.append(f'<text class="small" x="682" y="{y + 13}">{score:.3f}</text>')
            y += 22
        y += 20
    return _svg(width, height, "\n".join(body))


def _severity_curves_svg(rows: list[dict[str, Any]]) -> str:
    width, height = 960, 430
    body = [f'<rect width="{width}" height="{height}" fill="#ffffff"/>', '<text class="label" x="30" y="30">Severity Response Curves</text>']
    left, top, chart_w, chart_h = 70, 70, 820, 290
    body.append(f'<line class="axis" x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}"/>')
    body.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}"/>')
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["baseline_name"]), str(row["family"])), []).append(row)
    colors = ["#2563eb", "#059669", "#d97706", "#7c3aed", "#dc2626", "#0891b2", "#4b5563"]
    for index, ((baseline, family), items) in enumerate(sorted(grouped.items())):
        ordered = sorted(items, key=lambda item: int(item["severity"]))
        points = []
        for item in ordered:
            severity = int(item["severity"])
            score = max(0.0, min(1.0, float(item["family_primary_score"])))
            x = left + (severity / 3.0) * chart_w
            y = top + (1.0 - score) * chart_h
            points.append((x, y))
        color = colors[index % len(colors)]
        body.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in points) + f'" fill="none" stroke="{color}" stroke-width="1.4"/>')
        for x, y in points:
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.8" fill="{color}"/>')
        if points:
            body.append(f'<text class="small" x="{points[-1][0] + 5:.1f}" y="{points[-1][1] + 4:.1f}">{baseline}/{family}</text>')
    for severity in range(4):
        x = left + (severity / 3.0) * chart_w
        body.append(f'<text class="small" x="{x}" y="{top + chart_h + 24}" text-anchor="middle">{severity}</text>')
    return _svg(width, height, "\n".join(body))


def _rank_comparison_svg(protocol: dict[str, Any]) -> str:
    comparison = protocol["pairwise_comparisons"].get("same_environment_vs_cep", {})
    left_order = comparison.get("left_order", [])
    right_order = comparison.get("right_order", [])
    methods = list(dict.fromkeys(list(left_order) + list(right_order)))
    height = 120 + max(1, len(methods)) * 34
    body = [f'<rect width="820" height="{height}" fill="#ffffff"/>', '<text class="label" x="30" y="30">Same-Environment vs CEP Ranking</text>']
    body.append('<text class="small muted" x="30" y="52">Lines connect each method rank under the two protocols.</text>')
    x_left, x_right = 230, 590
    body.append(f'<text class="label" x="{x_left}" y="82" text-anchor="middle">Same Environment</text>')
    body.append(f'<text class="label" x="{x_right}" y="82" text-anchor="middle">CEP</text>')
    colors = ["#2563eb", "#059669", "#d97706", "#7c3aed", "#dc2626", "#0891b2", "#4b5563"]
    for index, method in enumerate(methods):
        left_rank = left_order.index(method) if method in left_order else len(methods)
        right_rank = right_order.index(method) if method in right_order else len(methods)
        y_left = 110 + left_rank * 32
        y_right = 110 + right_rank * 32
        color = colors[index % len(colors)]
        body.append(f'<line x1="{x_left}" y1="{y_left}" x2="{x_right}" y2="{y_right}" stroke="{color}" stroke-width="1.4"/>')
        body.append(f'<circle cx="{x_left}" cy="{y_left}" r="4" fill="{color}"/>')
        body.append(f'<circle cx="{x_right}" cy="{y_right}" r="4" fill="{color}"/>')
        body.append(f'<text class="small" x="30" y="{110 + index * 18}">{method}</text>')
    return _svg(820, height, "\n".join(body))


def render_outputs(study_bundle_path: Path, output_dir: Path) -> dict[str, str]:
    bundle = json.loads(study_bundle_path.read_text(encoding="utf-8"))
    cep_report = bundle["raw_reports"]["cep_report"]
    paths = {
        "main_familywise_results_table": _write(output_dir / "tables" / "main_familywise_results.md", _render_main_results_table(bundle)),
        "benchmark_health_table": _write(output_dir / "tables" / "benchmark_health_summary.md", _render_health_table(bundle)),
        "baseline_metadata_table": _write(output_dir / "tables" / "baseline_metadata.md", _render_baseline_metadata_table(bundle)),
        "protocol_sensitivity_table": _write(output_dir / "tables" / "protocol_sensitivity_summary.md", _render_protocol_sensitivity_table(bundle)),
        "cep_protocol_diagram": _write(output_dir / "figures" / "cep_protocol_diagram.svg", _cep_protocol_svg()),
        "intervention_examples": _write(output_dir / "figures" / "intervention_examples_2d.svg", _intervention_examples_svg()),
        "familywise_main_results": _write(
            output_dir / "figures" / "familywise_main_results.svg",
            _bar_chart_svg(cep_report["familywise_summary"]["rows"], "Family-Wise Main Results"),
        ),
        "severity_response_curves": _write(
            output_dir / "figures" / "severity_response_curves.svg",
            _severity_curves_svg(cep_report["familywise_summary"]["by_severity"]),
        ),
        "protocol_rank_reversal_comparison": _write(
            output_dir / "figures" / "protocol_rank_reversal_comparison.svg",
            _rank_comparison_svg(bundle["protocol_sensitivity"]),
        ),
    }
    _write(output_dir / "paper_outputs_manifest.json", json.dumps(paths, indent=2, sort_keys=True) + "\n")
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_bundle", help="Path to a saved study_bundle.json file.")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args()

    study_bundle_path = Path(args.study_bundle).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else study_bundle_path.parent / "paper_outputs"
    paths = render_outputs(study_bundle_path, output_dir)
    if args.print_summary:
        print(json.dumps(paths, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
