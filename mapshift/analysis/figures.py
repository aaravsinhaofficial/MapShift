"""Figure-ready data exports for MapShift study bundles."""

from __future__ import annotations

from typing import Any

CORE_FIGURES = (
    "cep_pipeline",
    "intervention_family_examples",
    "familywise_degradation_curves",
    "protocol_ranking_comparison",
    "failure_taxonomy_summary",
    "external_validity_summary",
    "mapshift_changes_conclusion",
)


def familywise_degradation_curves_figure_data(cep_report: dict[str, Any]) -> dict[str, Any]:
    """Return figure-ready severity-response data from one CEP report payload."""

    rows = cep_report["familywise_summary"]["by_severity"]
    return {
        "figure_name": "familywise_degradation_curves",
        "x_axis": "severity",
        "y_axis": "family_primary_score",
        "series": rows,
    }


def protocol_ranking_comparison_figure_data(protocol_report: dict[str, Any]) -> dict[str, Any]:
    """Return figure-ready protocol ranking comparison data."""

    return {
        "figure_name": "protocol_ranking_comparison",
        "pairwise_comparisons": protocol_report["pairwise_comparisons"],
        "pooled_vs_familywise": protocol_report["pooled_vs_familywise"],
    }


def mapshift_changes_conclusion_figure_data(protocol_report: dict[str, Any], study_summary: dict[str, Any]) -> dict[str, Any]:
    """Return one concise figure export focused on whether MapShift changes conclusions."""

    comparison = protocol_report["pairwise_comparisons"].get("same_environment_vs_cep", {})
    return {
        "figure_name": "mapshift_changes_conclusion",
        "same_environment_vs_cep": comparison,
        "pooled_vs_familywise": protocol_report["pooled_vs_familywise"],
        "p3_support": study_summary["proposition_support"]["P3"],
    }
