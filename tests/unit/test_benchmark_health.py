from __future__ import annotations

import unittest
from pathlib import Path

from mapshift.analysis.construct_validity import generate_mapshift_2d_benchmark_health_report
from mapshift.core.schemas import load_release_bundle
from mapshift.envs.map2d.dynamics import DynamicsParameters2D
from mapshift.envs.map2d.state import AgentPose2D, Map2DEnvironment, Map2DNode
from mapshift.envs.map2d.validation import analyze_map2d_environment, summarize_environment_diagnostics
from mapshift.interventions import build_intervention
from mapshift.interventions.validators import validate_intervention_pair
from mapshift.tasks.samplers import TaskSampler, TaskSamplingRejected


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_CONFIG = REPO_ROOT / "configs" / "benchmark" / "release_v0_1.json"


def handcrafted_environment() -> Map2DEnvironment:
    occupancy = [
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ]
    nodes = {
        "n0": Map2DNode(node_id="n0", row=0, col=0),
        "n1": Map2DNode(node_id="n1", row=0, col=4),
    }
    return Map2DEnvironment(
        environment_id="handcrafted",
        motif_tag="handcrafted",
        split_name="test",
        seed=0,
        width_cells=5,
        height_cells=5,
        occupancy_grid=[list(row) for row in occupancy],
        nodes=nodes,
        adjacency={"n0": [], "n1": []},
        room_cells={"n0": [(0, 0)], "n1": [(0, 4)]},
        edge_corridors={},
        start_node_id="n0",
        goal_node_id="n1",
        landmark_by_node={},
        goal_tokens={"target_alpha": "n1"},
        dynamics=DynamicsParameters2D(),
        occupancy_resolution_m=1.0,
        geometry_scale=1.0,
        observation_radius_m=2.0,
        field_of_view_deg=180.0,
        semantic_channels=True,
        agent_pose=AgentPose2D(x=0.0, y=0.0, theta_deg=0.0),
        history=["handcrafted"],
        metadata={},
    )


class BenchmarkHealthTests(unittest.TestCase):
    def test_environment_diagnostics_on_handcrafted_map(self) -> None:
        environment = handcrafted_environment()

        diagnostic = analyze_map2d_environment(environment)
        summary = summarize_environment_diagnostics([diagnostic])

        self.assertEqual(diagnostic.free_cell_count, 9)
        self.assertEqual(diagnostic.map_area_cells, 25)
        self.assertAlmostEqual(diagnostic.free_space_ratio, 9 / 25)
        self.assertEqual(diagnostic.start_goal_distance, 8.0)
        self.assertEqual(summary["environment_count"], 1)
        self.assertEqual(summary["path_length_summary"]["sum"], 8.0)
        self.assertEqual(summary["free_cell_count_summary"]["sum"], 9.0)

    def test_task_sampler_rejects_impossible_tasks_with_logged_reason(self) -> None:
        bundle = load_release_bundle(ROOT_CONFIG)
        sampler = TaskSampler(bundle.tasks)
        base_environment = handcrafted_environment()
        intervened_environment = base_environment.clone(environment_id="handcrafted-impossible")
        intervened_environment.occupancy_grid[2] = [1, 1, 1, 1, 1]

        with self.assertRaises(TaskSamplingRejected) as context:
            sampler.sample(
                base_environment=base_environment,
                intervened_environment=intervened_environment,
                family="metric",
                seed=11,
                task_class="planning",
            )

        self.assertEqual(context.exception.rejection.reason, "impossible_path")
        self.assertEqual(sampler.rejection_summary(), {"impossible_path": 1})

    def test_task_sampler_rejects_trivial_tasks_with_logged_reason(self) -> None:
        bundle = load_release_bundle(ROOT_CONFIG)
        sampler = TaskSampler(bundle.tasks)
        base_environment = handcrafted_environment()
        intervened_environment = base_environment.clone(environment_id="handcrafted-trivial")
        intervened_environment.goal_node_id = intervened_environment.start_node_id
        intervened_environment.goal_tokens["target_alpha"] = intervened_environment.start_node_id

        with self.assertRaises(TaskSamplingRejected) as context:
            sampler.sample(
                base_environment=base_environment,
                intervened_environment=intervened_environment,
                family="metric",
                seed=13,
                task_class="planning",
            )

        self.assertEqual(context.exception.rejection.reason, "trivial_goal_already_solved")
        self.assertEqual(sampler.rejection_summary(), {"trivial_goal_already_solved": 1})

    def test_intervention_validator_accepts_noop_and_rejects_malformed_topology(self) -> None:
        bundle = load_release_bundle(ROOT_CONFIG)
        generator = build_generator(bundle)
        base_environment = generator.generate(seed=17, motif_tag="simple_loop").environment

        topology = build_intervention("topology", bundle.interventions.families["topology"])
        noop = topology.apply(base_environment, severity=0, seed=101).environment
        noop_validation = validate_intervention_pair(
            base_environment=base_environment,
            transformed_environment=noop,
            family="topology",
            severity=0,
            family_config=bundle.interventions.families["topology"],
        )
        self.assertTrue(noop_validation.ok)

        malformed = base_environment.clone(environment_id="simple-loop-malformed")
        malformed.goal_tokens["target_alpha"] = malformed.start_node_id
        malformed_validation = validate_intervention_pair(
            base_environment=base_environment,
            transformed_environment=malformed,
            family="topology",
            severity=2,
            family_config=bundle.interventions.families["topology"],
        )
        self.assertIn("topology shift changed semantic assignments", malformed_validation.issues)

    def test_topology_sampler_chooses_route_affected_pair(self) -> None:
        bundle = load_release_bundle(ROOT_CONFIG)
        generator = build_generator(bundle)
        sampler = TaskSampler(bundle.tasks)
        base_environment = generator.generate(seed=200, motif_tag="two_room_connector").environment

        intervention = build_intervention("topology", bundle.interventions.families["topology"])
        transformed_environment = intervention.apply(base_environment, severity=1, seed=201).environment
        sampled = sampler.sample(
            base_environment=base_environment,
            intervened_environment=transformed_environment,
            family="topology",
            seed=77,
            task_class="planning",
        )

        self.assertEqual(sampled.task.family, "topology")
        self.assertIn(sampled.task.task_type, {"shortest_path_to_target", "reroute_after_blockage"})
        self.assertTrue(sampled.task.metadata["route_changed"])
        self.assertTrue(
            sampled.task.metadata["path_changed"]
            or sampled.task.metadata["distance_delta"] is not None
        )

    def test_release_benchmark_health_report_is_stable(self) -> None:
        bundle = load_release_bundle(ROOT_CONFIG)
        report = generate_mapshift_2d_benchmark_health_report(
            release_bundle=bundle,
            sample_count_per_motif=1,
            task_samples_per_class=1,
        )

        self.assertEqual(report.environment_health["environment_count"], 24)
        self.assertEqual(report.environment_health["split_counts"], {"test": 8, "train": 10, "val": 6})
        self.assertEqual(report.environment_health["path_length_summary"]["sum"], 457.0)
        self.assertEqual(report.intervention_coverage["undercovered_cells"], [])
        self.assertEqual(report.validator_summary["failed_intervention_count"], 0)
        self.assertEqual(report.validator_summary["severity_monotonicity_failures"], [])
        self.assertEqual(report.rejection_statistics["rejections_by_reason"], {})
        self.assertEqual(report.task_coverage["undercovered_cells"], [])
        self.assertEqual(report.task_difficulty["task_count"], 1152)

        for family, severity_counts in report.intervention_coverage["family_severity_table"].items():
            self.assertEqual(severity_counts, {"0": 24, "1": 24, "2": 24, "3": 24}, family)


def build_generator(bundle: object):
    from mapshift.envs.map2d.generator import Map2DGenerator

    return Map2DGenerator(bundle.env2d)


if __name__ == "__main__":
    unittest.main()
