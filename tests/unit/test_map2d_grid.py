from __future__ import annotations

import unittest
from pathlib import Path

from mapshift.core.schemas import FAMILY_NAMES, load_release_bundle
from mapshift.envs.map2d.dynamics import DynamicsParameters2D
from mapshift.envs.map2d.generator import Map2DGenerator
from mapshift.envs.map2d.observation import observe_egocentric
from mapshift.envs.map2d.state import AgentPose2D, Map2DEnvironment, Map2DNode
from mapshift.interventions import build_intervention


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


class Map2DGridTests(unittest.TestCase):
    def test_shortest_path_correctness_on_hand_built_map(self) -> None:
        environment = handcrafted_environment()
        self.assertTrue(environment.reachable("n0", "n1"))
        self.assertEqual(environment.shortest_path_length("n0", "n1"), 8.0)

    def test_manifest_replay_is_exact(self) -> None:
        bundle = load_release_bundle(ROOT_CONFIG)
        generator = Map2DGenerator(bundle.env2d)
        result = generator.generate(seed=17, motif_tag="offset_bottleneck")

        replayed = generator.replay_from_manifest(result.manifest)
        self.assertEqual(replayed.to_dict(), result.environment.to_dict())

    def test_no_op_intervention_equivalence(self) -> None:
        bundle = load_release_bundle(ROOT_CONFIG)
        generator = Map2DGenerator(bundle.env2d)
        base = generator.generate(seed=5, motif_tag="asymmetric_multi_room_chain").environment

        for family in FAMILY_NAMES:
            intervention = build_intervention(family, bundle.interventions.families[family])
            result = intervention.apply(base, severity=0, seed=100)
            transformed = result.environment

            self.assertEqual(transformed.geometry_signature(), base.geometry_signature(), family)
            self.assertEqual(transformed.semantic_signature(), base.semantic_signature(), family)
            self.assertEqual(transformed.dynamics_signature(), base.dynamics_signature(), family)

    def test_intervention_manifest_replay_is_exact(self) -> None:
        bundle = load_release_bundle(ROOT_CONFIG)
        generator = Map2DGenerator(bundle.env2d)
        base = generator.generate(seed=14, motif_tag="offset_bottleneck").environment
        intervention = build_intervention("topology", bundle.interventions.families["topology"])
        transformed = intervention.apply(base, severity=2, seed=52)

        replayed = Map2DEnvironment.from_manifest_metadata(transformed.manifest.metadata)
        self.assertEqual(replayed.to_dict(), transformed.environment.to_dict())

    def test_topology_changes_preserve_semantics(self) -> None:
        bundle = load_release_bundle(ROOT_CONFIG)
        generator = Map2DGenerator(bundle.env2d)
        base = generator.generate(seed=8, motif_tag="branching_chain").environment
        intervention = build_intervention("topology", bundle.interventions.families["topology"])
        transformed = intervention.apply(base, severity=2, seed=44).environment

        self.assertEqual(transformed.semantic_signature(), base.semantic_signature())
        self.assertNotEqual(transformed.geometry_signature(), base.geometry_signature())

    def test_semantic_changes_preserve_geometry(self) -> None:
        bundle = load_release_bundle(ROOT_CONFIG)
        generator = Map2DGenerator(bundle.env2d)
        base = generator.generate(seed=8, motif_tag="branching_chain").environment
        intervention = build_intervention("semantic", bundle.interventions.families["semantic"])
        transformed = intervention.apply(base, severity=2, seed=45).environment

        self.assertEqual(transformed.geometry_signature(), base.geometry_signature())
        self.assertNotEqual(transformed.semantic_signature(), base.semantic_signature())

    def test_observation_model_returns_local_patch(self) -> None:
        bundle = load_release_bundle(ROOT_CONFIG)
        generator = Map2DGenerator(bundle.env2d)
        base = generator.generate(seed=12, motif_tag="simple_loop").environment

        frame = observe_egocentric(base)

        self.assertTrue(frame.geometry_patch)
        self.assertEqual(len(frame.geometry_patch), len(frame.semantic_patch))
        self.assertGreaterEqual(len(frame.visible_landmarks), 0)


if __name__ == "__main__":
    unittest.main()
