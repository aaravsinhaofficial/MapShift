from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from mapshift.analysis.construct_validity import generate_mapshift_2d_benchmark_health_report
from mapshift.analysis.study import (
    build_mapshift_2d_study_bundle,
    load_mapshift_2d_study_config,
    run_mapshift_2d_study,
    write_mapshift_2d_study_bundle,
)
from mapshift.core.schemas import load_release_bundle


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_CONFIG = REPO_ROOT / "configs" / "benchmark" / "release_v0_1.json"
SMOKE_STUDY_CONFIG = REPO_ROOT / "configs" / "analysis" / "mapshift_2d_full_study_smoke_v0_1.json"
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for the learned-baseline 2D study integration test")
class MapShift2DStudyIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.release_bundle = load_release_bundle(ROOT_CONFIG)
        cls.study_config = load_mapshift_2d_study_config(SMOKE_STUDY_CONFIG)
        cls.study_bundle = run_mapshift_2d_study(cls.study_config, release_bundle=cls.release_bundle)
        cls.output_dir_context = tempfile.TemporaryDirectory(prefix="mapshift-study-")
        cls.output_dir = Path(cls.output_dir_context.name)
        cls.artifact_paths = write_mapshift_2d_study_bundle(cls.study_bundle, cls.output_dir)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.output_dir_context.cleanup()

    def test_smoke_study_writes_expected_artifacts(self) -> None:
        expected_keys = {
            "study_bundle",
            "raw_cep_report",
            "raw_protocol_comparison_report",
            "raw_benchmark_health_report",
            "study_manifest",
            "familywise_main_results",
            "severity_response",
            "protocol_sensitivity_and_rank_correlation",
            "benchmark_health_summary",
            "construct_validity_summary",
            "discriminative_power_summary",
            "familywise_degradation_curves",
            "protocol_ranking_comparison",
            "mapshift_changes_conclusion",
        }
        self.assertEqual(set(self.artifact_paths), expected_keys)
        for path in self.artifact_paths.values():
            self.assertTrue(Path(path).is_file(), path)

        familywise_rows = json.loads(Path(self.artifact_paths["familywise_main_results"]).read_text(encoding="utf-8"))
        self.assertEqual(familywise_rows, self.study_bundle.raw_reports["cep_report"]["familywise_summary"]["rows"])

        protocol_table = json.loads(
            Path(self.artifact_paths["protocol_sensitivity_and_rank_correlation"]).read_text(encoding="utf-8")
        )
        self.assertIn("pairwise_comparisons", protocol_table)
        self.assertIn("same_environment_vs_cep", protocol_table["pairwise_comparisons"])
        familywise_rows = self.study_bundle.report_artifacts["tables"]["familywise_main_results"]
        self.assertIn(
            "relational_graph_world_model",
            {str(row["baseline_name"]) for row in familywise_rows},
        )

    def test_study_bundle_can_be_regenerated_from_raw_reports(self) -> None:
        rebuilt = build_mapshift_2d_study_bundle(
            release_bundle=self.release_bundle,
            study_config=self.study_config,
            cep_report_payload=self.study_bundle.raw_reports["cep_report"],
            protocol_report_payload=self.study_bundle.raw_reports["protocol_comparison_report"],
            benchmark_health_payload=self.study_bundle.raw_reports["benchmark_health_report"],
        )

        self.assertEqual(
            rebuilt.report_artifacts["tables"]["familywise_main_results"],
            self.study_bundle.report_artifacts["tables"]["familywise_main_results"],
        )
        self.assertEqual(
            rebuilt.protocol_sensitivity["pairwise_comparisons"]["same_environment_vs_cep"],
            self.study_bundle.protocol_sensitivity["pairwise_comparisons"]["same_environment_vs_cep"],
        )
        self.assertEqual(rebuilt.proposition_support, self.study_bundle.proposition_support)

    def test_missing_cells_are_preserved_in_benchmark_health_artifacts(self) -> None:
        undercovered_config = replace(self.study_config, min_cell_coverage=2)
        health_report = generate_mapshift_2d_benchmark_health_report(
            release_bundle=self.release_bundle,
            sample_count_per_motif=undercovered_config.sample_count_per_motif,
            task_samples_per_class=undercovered_config.task_samples_per_class,
            min_cell_coverage=undercovered_config.min_cell_coverage,
            motif_tags=undercovered_config.motif_tags,
            family_names=undercovered_config.family_names,
        )
        rebuilt = build_mapshift_2d_study_bundle(
            release_bundle=self.release_bundle,
            study_config=undercovered_config,
            cep_report_payload=self.study_bundle.raw_reports["cep_report"],
            protocol_report_payload=self.study_bundle.raw_reports["protocol_comparison_report"],
            benchmark_health_payload=health_report.to_dict(),
        )

        undercovered_cells = rebuilt.benchmark_health["task_coverage"]["undercovered_cells"]
        self.assertTrue(undercovered_cells)

        with tempfile.TemporaryDirectory(prefix="mapshift-study-undercovered-") as tmpdir:
            artifact_paths = write_mapshift_2d_study_bundle(rebuilt, tmpdir)
            persisted = json.loads(Path(artifact_paths["benchmark_health_summary"]).read_text(encoding="utf-8"))
        self.assertIn("undercovered_cells", persisted["task_coverage"])
        self.assertEqual(persisted["task_coverage"]["undercovered_cells"], undercovered_cells)


if __name__ == "__main__":
    unittest.main()
