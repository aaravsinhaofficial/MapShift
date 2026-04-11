from __future__ import annotations

import unittest
from pathlib import Path

from mapshift.core.schemas import load_release_bundle
from mapshift.envs.procthor.generator import ProcTHORGenerator
from mapshift.envs.procthor.observation import observe_scene
from mapshift.envs.procthor.validation import validate_procthor_scene
from mapshift.interventions import build_intervention
from mapshift.runners.compare_protocols import run_protocol_comparison_suite
from mapshift.runners.evaluate import run_calibration_suite
from mapshift.tasks.samplers import TaskSampler


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_CONFIG = REPO_ROOT / "configs" / "benchmark" / "release_v0_1.json"
ORACLE_CONFIG = REPO_ROOT / "configs" / "calibration" / "oracle_post_intervention_planner_v0_1.json"
HEURISTIC_CONFIG = REPO_ROOT / "configs" / "calibration" / "weak_heuristic_baseline_v0_1.json"
RECURRENT_CONFIG = REPO_ROOT / "configs" / "calibration" / "monolithic_recurrent_world_model_v0_1.json"


class ProcTHORIntegrationTests(unittest.TestCase):
    def test_generator_is_deterministic_and_replayable(self) -> None:
        bundle = load_release_bundle(ROOT_CONFIG)
        generator = ProcTHORGenerator(bundle.env3d)

        first = generator.sample(seed=13, motif_tag="loop_loft")
        second = generator.sample(seed=13, motif_tag="loop_loft")
        self.assertIsNotNone(first.scene)
        self.assertEqual(first.scene.serialize(), second.scene.serialize())

        replayed = generator.replay_from_manifest(first.manifest)
        self.assertEqual(first.scene.serialize(), replayed.serialize())
        self.assertEqual(validate_procthor_scene(first.scene), [])

    def test_observation_and_semantic_task_sampling_work_on_procthor(self) -> None:
        bundle = load_release_bundle(ROOT_CONFIG)
        generator = ProcTHORGenerator(bundle.env3d)
        sampler = TaskSampler(bundle.tasks)
        base_scene = generator.sample(seed=17, motif_tag="connector_duplex").scene
        semantic = build_intervention("semantic", bundle.interventions.families["semantic"]).apply(
            base_scene,
            severity=2,
            seed=23,
        ).environment

        frame = observe_scene(base_scene)
        self.assertEqual(frame.room_id, base_scene.start_node_id)
        self.assertTrue(frame.visible_objects)

        inference = sampler.sample(
            base_environment=base_scene,
            intervened_environment=semantic,
            family="semantic",
            seed=91,
            task_class="inference",
        ).task
        self.assertEqual(inference.expected_output_type, "room_id")
        self.assertIn(str(inference.expected_answer), base_scene.rooms)

    def test_3d_calibration_suite_runs_end_to_end(self) -> None:
        bundle = load_release_bundle(ROOT_CONFIG)
        report = run_calibration_suite(
            release_bundle=bundle,
            baseline_run_configs=[ORACLE_CONFIG, HEURISTIC_CONFIG],
            sample_count_per_motif=1,
            task_samples_per_class=1,
            severity_levels=(0, 1),
            motif_tags=["loop_loft"],
            family_names=["metric", "topology", "semantic"],
            tier="mapshift_3d",
        )

        self.assertEqual(report.tier, "mapshift_3d")
        self.assertTrue(report.records)
        self.assertEqual(set(report.baseline_metadata), {"oracle_post_intervention_planner", "weak_heuristic_baseline"})
        self.assertTrue(report.familywise_summary["rows"])

    def test_3d_protocol_comparison_runs(self) -> None:
        bundle = load_release_bundle(ROOT_CONFIG)
        report = run_protocol_comparison_suite(
            release_bundle=bundle,
            baseline_run_configs=[ORACLE_CONFIG, HEURISTIC_CONFIG],
            sample_count_per_motif=1,
            task_samples_per_class=1,
            severity_levels=(0,),
            motif_tags=["room_chain_flat"],
            family_names=["metric"],
            protocol_names=("cep", "same_environment"),
            tier="mapshift_3d",
        )

        self.assertEqual(report.tier, "mapshift_3d")
        self.assertIn("same_environment_vs_cep", report.pairwise_comparisons)

    def test_3d_runner_rejects_2d_only_baselines(self) -> None:
        bundle = load_release_bundle(ROOT_CONFIG)
        with self.assertRaises(ValueError):
            run_calibration_suite(
                release_bundle=bundle,
                baseline_run_configs=[RECURRENT_CONFIG],
                sample_count_per_motif=1,
                task_samples_per_class=1,
                severity_levels=(0,),
                motif_tags=["loop_loft"],
                family_names=["metric"],
                tier="mapshift_3d",
            )


if __name__ == "__main__":
    unittest.main()
