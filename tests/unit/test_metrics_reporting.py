from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from mapshift.metrics.adaptation_metrics import adaptation_sample_efficiency, summarize_adaptation_curve
from mapshift.metrics.planning_metrics import mean_normalized_path_efficiency, success_rate
from mapshift.metrics.ranking import kendall_tau, rank_by_metric, rank_reversals, ranking_spread
from mapshift.metrics.statistics import bootstrap_statistic, mean_or_zero
from mapshift.core.schemas import load_release_bundle
from mapshift.runners.compare_protocols import run_protocol_comparison_suite
from mapshift.runners.evaluate import default_post_intervention_protocol, run_calibration_suite


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_CONFIG = REPO_ROOT / "configs" / "benchmark" / "release_v0_1.json"
ORACLE_CONFIG = REPO_ROOT / "configs" / "calibration" / "oracle_post_intervention_planner_v0_1.json"
HEURISTIC_CONFIG = REPO_ROOT / "configs" / "calibration" / "weak_heuristic_baseline_v0_1.json"
RECURRENT_CONFIG = REPO_ROOT / "configs" / "calibration" / "monolithic_recurrent_world_model_v0_1.json"
MEMORY_CONFIG = REPO_ROOT / "configs" / "calibration" / "persistent_memory_world_model_v0_1.json"
RELATIONAL_CONFIG = REPO_ROOT / "configs" / "calibration" / "relational_graph_world_model_v0_1.json"
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class MetricsReportingTests(unittest.TestCase):
    def test_planning_metric_helpers_on_known_values(self) -> None:
        self.assertAlmostEqual(success_rate([True, False, True]), 2.0 / 3.0)
        self.assertAlmostEqual(
            mean_normalized_path_efficiency([4.0, 4.0], [4.0, 8.0]),
            0.75,
        )

    def test_adaptation_metric_helpers_on_synthetic_curve(self) -> None:
        curve = [0.0, 0.5, 1.0]
        self.assertAlmostEqual(adaptation_sample_efficiency(curve), 0.5)
        summary = summarize_adaptation_curve(curve)
        self.assertEqual(summary.start, 0.0)
        self.assertEqual(summary.end, 1.0)
        self.assertEqual(summary.improvement, 1.0)
        self.assertAlmostEqual(summary.area_under_curve, 0.5)

    def test_bootstrap_summary_is_reproducible(self) -> None:
        first = bootstrap_statistic([1.0, 2.0, 3.0], mean_or_zero, resamples=200, confidence_level=0.95, seed=7)
        second = bootstrap_statistic([1.0, 2.0, 3.0], mean_or_zero, resamples=200, confidence_level=0.95, seed=7)

        self.assertEqual(first.to_dict(), second.to_dict())

    def test_ranking_spread_and_reversals_on_mock_values(self) -> None:
        metric_values = {"oracle": 0.9, "recurrent": 0.6, "heuristic": 0.3}
        order_a = rank_by_metric(metric_values)
        order_b = ["oracle", "heuristic", "recurrent"]

        self.assertEqual(order_a, ["oracle", "recurrent", "heuristic"])
        self.assertGreater(ranking_spread(metric_values), 0.0)
        self.assertLess(kendall_tau(order_a, order_b), 1.0)
        self.assertTrue(rank_reversals(order_a, order_b))

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is required for learned baseline metrics tests")
    def test_calibration_report_contains_grouped_metrics_and_bootstrap_rows(self) -> None:
        bundle = load_release_bundle(ROOT_CONFIG)
        report = run_calibration_suite(
            release_bundle=bundle,
            baseline_run_configs=[ORACLE_CONFIG, HEURISTIC_CONFIG, RECURRENT_CONFIG, MEMORY_CONFIG, RELATIONAL_CONFIG],
            sample_count_per_motif=1,
            task_samples_per_class=1,
            severity_levels=(0,),
            protocol=default_post_intervention_protocol(),
            motif_tags=["simple_loop"],
            family_names=["metric", "semantic"],
        )

        self.assertTrue(report.familywise_summary["rows"])
        self.assertTrue(report.bootstrap_summary["familywise_main_results"])
        self.assertTrue(report.ranking_summary["pooled_supplementary_orders"])
        self.assertIn("tables", report.report_artifacts)
        self.assertIn("familywise_main_results", report.report_artifacts["tables"])

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is required for learned baseline metrics tests")
    def test_protocol_comparison_outputs_exist_on_small_fixture_run(self) -> None:
        bundle = load_release_bundle(ROOT_CONFIG)
        report = run_protocol_comparison_suite(
            release_bundle=bundle,
            baseline_run_configs=[ORACLE_CONFIG, HEURISTIC_CONFIG, RECURRENT_CONFIG, MEMORY_CONFIG, RELATIONAL_CONFIG],
            sample_count_per_motif=1,
            task_samples_per_class=1,
            severity_levels=(0,),
            motif_tags=["simple_loop"],
            family_names=["metric"],
        )

        self.assertIn("same_environment_vs_cep", report.pairwise_comparisons)
        self.assertIn("no_exploration_vs_reward_free_exploration", report.pairwise_comparisons)
        self.assertIn("short_horizon_vs_long_horizon", report.pairwise_comparisons)
        self.assertIn("kendall_tau", report.pairwise_comparisons["same_environment_vs_cep"])
        self.assertIn("pooled_order", report.pooled_vs_familywise)


if __name__ == "__main__":
    unittest.main()
