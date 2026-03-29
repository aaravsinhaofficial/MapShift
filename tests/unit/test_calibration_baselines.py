from __future__ import annotations

import unittest
from pathlib import Path

from mapshift.baselines import instantiate_baseline, load_baseline_run_config
from mapshift.baselines.api import BaselineContext
from mapshift.core.schemas import load_release_bundle
from mapshift.envs.map2d.dynamics import DynamicsParameters2D
from mapshift.envs.map2d.generator import Map2DGenerator
from mapshift.envs.map2d.state import AgentPose2D, Map2DEnvironment, Map2DNode
from mapshift.interventions import build_intervention
from mapshift.runners.evaluate import run_calibration_suite, run_evaluation
from mapshift.runners.explore import run_exploration
from mapshift.tasks.inference import InferenceTask
from mapshift.tasks.planning import PlanningTask


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_CONFIG = REPO_ROOT / "configs" / "benchmark" / "release_v0_1.json"
ORACLE_CONFIG = REPO_ROOT / "configs" / "calibration" / "oracle_post_intervention_planner_v0_1.json"
HEURISTIC_CONFIG = REPO_ROOT / "configs" / "calibration" / "weak_heuristic_baseline_v0_1.json"
RECURRENT_CONFIG = REPO_ROOT / "configs" / "calibration" / "monolithic_recurrent_world_model_v0_1.json"
MEMORY_CONFIG = REPO_ROOT / "configs" / "calibration" / "persistent_memory_world_model_v0_1.json"
RELATIONAL_CONFIG = REPO_ROOT / "configs" / "calibration" / "relational_graph_world_model_v0_1.json"


def corridor_environment() -> Map2DEnvironment:
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
        environment_id="corridor-env",
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
        history=["corridor"],
        metadata={},
    )


def unreachable_environment() -> Map2DEnvironment:
    occupancy = [
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
    ]
    nodes = {
        "n0": Map2DNode(node_id="n0", row=0, col=0),
        "n1": Map2DNode(node_id="n1", row=0, col=2),
    }
    return Map2DEnvironment(
        environment_id="unreachable-env",
        motif_tag="handcrafted",
        split_name="test",
        seed=0,
        width_cells=3,
        height_cells=3,
        occupancy_grid=[list(row) for row in occupancy],
        nodes=nodes,
        adjacency={"n0": [], "n1": []},
        room_cells={"n0": [(0, 0)], "n1": [(0, 2)]},
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
        history=["unreachable"],
        metadata={},
    )


class CalibrationBaselineTests(unittest.TestCase):
    def test_oracle_shortest_path_matches_handcrafted_distance(self) -> None:
        environment = corridor_environment()
        config = load_baseline_run_config(ORACLE_CONFIG)
        oracle = instantiate_baseline(config)
        context = BaselineContext(model_name=oracle.name, exploration_budget_steps=config.exploration_budget_steps, seed=config.seed)
        exploration = run_exploration(oracle, environment, context)
        task = PlanningTask(
            task_type="shortest_path_to_target",
            horizon_steps=32,
            family="metric",
            start_node_id="n0",
            goal_node_id="n1",
            goal_token=None,
            goal_descriptor="reach n1",
        )

        result = run_evaluation(oracle, environment, task, exploration, context)

        self.assertTrue(result.success)
        self.assertEqual(result.oracle_length, 8.0)
        self.assertEqual(result.observed_length, 8.0)
        self.assertEqual(len(result.path_cells), 9)

    def test_oracle_reroutes_after_topology_change(self) -> None:
        bundle = load_release_bundle(ROOT_CONFIG)
        generator = Map2DGenerator(bundle.env2d)
        base_environment = generator.generate(seed=17, motif_tag="branching_chain").environment
        intervention = build_intervention("topology", bundle.interventions.families["topology"])
        transformed_environment = intervention.apply(base_environment, severity=2, seed=99).environment
        self.assertNotEqual(
            base_environment.shortest_path_length(base_environment.start_node_id, base_environment.goal_node_id),
            transformed_environment.shortest_path_length(transformed_environment.start_node_id, transformed_environment.goal_node_id),
        )

        config = load_baseline_run_config(ORACLE_CONFIG)
        oracle = instantiate_baseline(config)
        context = BaselineContext(model_name=oracle.name, exploration_budget_steps=config.exploration_budget_steps, seed=config.seed)
        exploration = run_exploration(oracle, base_environment, context)
        task = PlanningTask(
            task_type="reroute_after_blockage",
            horizon_steps=128,
            family="topology",
            start_node_id=transformed_environment.start_node_id,
            goal_node_id=transformed_environment.goal_node_id,
            goal_token=None,
            goal_descriptor="reroute to goal",
        )

        result = run_evaluation(oracle, transformed_environment, task, exploration, context)

        self.assertTrue(result.success)
        self.assertEqual(
            result.observed_length,
            transformed_environment.shortest_path_length(transformed_environment.start_node_id, transformed_environment.goal_node_id),
        )

    def test_weak_heuristic_is_deterministic_on_fixed_task(self) -> None:
        bundle = load_release_bundle(ROOT_CONFIG)
        generator = Map2DGenerator(bundle.env2d)
        environment = generator.generate(seed=13, motif_tag="simple_loop").environment
        config = load_baseline_run_config(HEURISTIC_CONFIG)
        heuristic = instantiate_baseline(config)
        context = BaselineContext(model_name=heuristic.name, exploration_budget_steps=config.exploration_budget_steps, seed=config.seed)
        exploration = run_exploration(heuristic, environment, context)
        task = PlanningTask(
            task_type="shortest_path_to_target",
            horizon_steps=64,
            family="metric",
            start_node_id=environment.start_node_id,
            goal_node_id=environment.goal_node_id,
            goal_token=None,
            goal_descriptor="reach goal",
        )

        first = run_evaluation(heuristic, environment, task, exploration, context)
        second = run_evaluation(heuristic, environment, task, exploration, context)

        self.assertEqual(first.to_dict(), second.to_dict())

    def test_recurrent_wrapper_is_config_driven_and_tracks_parameters(self) -> None:
        config = load_baseline_run_config(RECURRENT_CONFIG)
        model = instantiate_baseline(config)

        self.assertEqual(model.name, "monolithic_recurrent_world_model")
        self.assertGreater(model.parameter_count, 0)
        self.assertEqual(model.parameter_count, model.trainable_parameter_count)
        self.assertEqual(model.describe()["parameters"]["hidden_size"], 12)

    def test_memory_wrapper_is_config_driven_and_tracks_parameters(self) -> None:
        config = load_baseline_run_config(MEMORY_CONFIG)
        model = instantiate_baseline(config)

        self.assertEqual(model.name, "persistent_memory_world_model")
        self.assertGreater(model.parameter_count, 0)
        self.assertEqual(model.parameter_count, model.trainable_parameter_count)
        self.assertEqual(model.describe()["parameters"]["memory_slots"], 16)

    def test_relational_wrapper_is_config_driven_and_tracks_parameters(self) -> None:
        config = load_baseline_run_config(RELATIONAL_CONFIG)
        model = instantiate_baseline(config)

        self.assertEqual(model.name, "relational_graph_world_model")
        self.assertGreater(model.parameter_count, 0)
        self.assertEqual(model.parameter_count, model.trainable_parameter_count)
        self.assertEqual(model.describe()["parameters"]["message_passing_steps"], 2)

    def test_relational_baseline_detects_topology_change(self) -> None:
        bundle = load_release_bundle(ROOT_CONFIG)
        generator = Map2DGenerator(bundle.env2d)
        base_environment = generator.generate(seed=17, motif_tag="branching_chain").environment
        intervention = build_intervention("topology", bundle.interventions.families["topology"])
        transformed_environment = intervention.apply(base_environment, severity=2, seed=99).environment

        config = load_baseline_run_config(RELATIONAL_CONFIG)
        model = instantiate_baseline(config)
        context = BaselineContext(model_name=model.name, exploration_budget_steps=config.exploration_budget_steps, seed=config.seed)
        exploration = run_exploration(model, base_environment, context)
        task = InferenceTask(
            task_type="detect_topology_change",
            family="topology",
            query="Did the connectivity structure change between the explored and intervened environment?",
            expected_output_type="boolean",
            expected_answer=True,
        )

        result = run_evaluation(model, transformed_environment, task, exploration, context)

        self.assertTrue(result.correct)
        self.assertTrue(result.metadata["structural_shift_detected"])

    def test_oracle_detects_impossible_tasks(self) -> None:
        environment = unreachable_environment()
        config = load_baseline_run_config(ORACLE_CONFIG)
        oracle = instantiate_baseline(config)
        context = BaselineContext(model_name=oracle.name, exploration_budget_steps=config.exploration_budget_steps, seed=config.seed)
        exploration = run_exploration(oracle, environment, context)
        task = PlanningTask(
            task_type="shortest_path_to_target",
            horizon_steps=16,
            family="metric",
            start_node_id="n0",
            goal_node_id="n1",
            goal_token=None,
            goal_descriptor="reach n1",
        )

        result = run_evaluation(oracle, environment, task, exploration, context)

        self.assertFalse(result.success)
        self.assertFalse(result.solvable)
        self.assertTrue(result.metadata["impossible_for_oracle"])

    def test_calibration_suite_runs_end_to_end_for_five_baselines(self) -> None:
        bundle = load_release_bundle(ROOT_CONFIG)
        report = run_calibration_suite(
            release_bundle=bundle,
            baseline_run_configs=[ORACLE_CONFIG, HEURISTIC_CONFIG, RECURRENT_CONFIG, MEMORY_CONFIG, RELATIONAL_CONFIG],
            sample_count_per_motif=1,
            task_samples_per_class=1,
            severity_levels=(0, 1),
        )

        baseline_names = {record.baseline_name for record in report.records}
        self.assertIn("oracle_post_intervention_planner", baseline_names)
        self.assertIn("weak_heuristic_baseline", baseline_names)
        self.assertIn("monolithic_recurrent_world_model", baseline_names)
        self.assertIn("persistent_memory_world_model", baseline_names)
        self.assertIn("relational_graph_world_model", baseline_names)
        self.assertTrue(report.familywise_summary["rows"])
        self.assertIn("oracle_post_intervention_planner", report.baseline_metadata)
        self.assertIn("weak_heuristic_baseline", report.baseline_metadata)
        self.assertIn("monolithic_recurrent_world_model", report.baseline_metadata)
        self.assertIn("persistent_memory_world_model", report.baseline_metadata)
        self.assertIn("relational_graph_world_model", report.baseline_metadata)


if __name__ == "__main__":
    unittest.main()
