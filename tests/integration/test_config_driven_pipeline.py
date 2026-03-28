from __future__ import annotations

import unittest
from pathlib import Path

from mapshift.core.schemas import FAMILY_NAMES, load_release_bundle
from mapshift.envs.map2d.generator import Map2DGenerator
from mapshift.envs.map2d.validation import validate_map2d_instance
from mapshift.interventions import build_intervention
from mapshift.tasks.samplers import TaskSampler


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_CONFIG = REPO_ROOT / "configs" / "benchmark" / "release_v0_1.json"


class ConfigDrivenPipelineTests(unittest.TestCase):
    def test_generate_intervene_sample_for_each_family(self) -> None:
        bundle = load_release_bundle(ROOT_CONFIG)
        generator = Map2DGenerator(bundle.env2d)
        sampler = TaskSampler(bundle.tasks)

        base_result = generator.generate(seed=13, motif_tag="simple_loop")
        self.assertEqual(validate_map2d_instance(base_result.environment), [])

        for index, family in enumerate(FAMILY_NAMES, start=1):
            intervention = build_intervention(family, bundle.interventions.families[family])
            intervened = intervention.apply(base_result.environment, severity=index if family != "semantic" else 2, seed=13 + index)

            self.assertIsNotNone(intervened.environment)
            self.assertEqual(intervened.manifest.intervention_family, family)

            sampled = sampler.sample(base_result.environment, intervened.environment, family, seed=100 + index, task_class="planning")

            self.assertEqual(sampled.manifest.task_class, "planning")
            self.assertEqual(sampled.task.family, family)
            self.assertEqual(sampled.manifest.base_environment_id, base_result.environment.environment_id)
            self.assertEqual(sampled.manifest.intervened_environment_id, intervened.environment.environment_id)

    def test_topology_shift_changes_graph_structure(self) -> None:
        bundle = load_release_bundle(ROOT_CONFIG)
        generator = Map2DGenerator(bundle.env2d)
        base_result = generator.generate(seed=21, motif_tag="two_room_connector")
        base_environment = base_result.environment

        intervention = build_intervention("topology", bundle.interventions.families["topology"])
        intervened = intervention.apply(base_environment, severity=2, seed=99)

        self.assertNotEqual(base_environment.edge_list(), intervened.environment.edge_list())

        sampler = TaskSampler(bundle.tasks)
        sampled = sampler.sample(
            base_environment=base_environment,
            intervened_environment=intervened.environment,
            family="topology",
            seed=199,
            task_class="planning",
        )

        self.assertTrue(sampled.task.metadata["route_changed"])
        self.assertEqual(sampled.manifest.metadata["route_changed"], sampled.task.metadata["route_changed"])


if __name__ == "__main__":
    unittest.main()
