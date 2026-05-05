from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mapshift.envs.minigrid_port import (
    MiniGridPortDependencyError,
    MiniGridPortGenerator,
    build_minigrid_intervention,
    validate_minigrid_intervention_pair,
)
from mapshift.envs.minigrid_port.state import MiniGridPortEnvironment
from mapshift.analysis.minigrid_cpe import run_minigrid_cpe_study


class MiniGridPortTests(unittest.TestCase):
    def test_generation_roundtrip_and_solvability(self) -> None:
        result = MiniGridPortGenerator().generate(seed=100, motif_tag="two_room_door")
        environment = result.environment
        replayed = MiniGridPortEnvironment.from_dict(environment.to_dict())
        self.assertEqual(environment.state_signature(), replayed.state_signature())
        self.assertGreater(len(environment.shortest_path()), 0)
        self.assertEqual(result.manifest.tier, "minigrid_port")

    def test_intervention_invariants(self) -> None:
        generator = MiniGridPortGenerator()
        self.assertEqual(len(generator.motif_tags), 6)
        for motif in generator.motif_tags:
            base = generator.generate(seed=101, motif_tag=motif).environment
            for family in ("metric", "topology", "dynamics", "semantic"):
                with self.subTest(motif=motif, family=family):
                    transformed = build_minigrid_intervention(family).apply(base, severity=2, seed=202).environment
                    validation = validate_minigrid_intervention_pair(base, transformed, family=family, severity=2)
                    self.assertTrue(validation.ok, validation.issues)
                    if family == "metric":
                        self.assertTrue(validation.metrics["metric_changed"])
                        self.assertFalse(validation.metrics["topology_changed"])
                    if family == "topology":
                        self.assertTrue(validation.metrics["topology_changed"])
                        self.assertFalse(validation.metrics["semantic_changed"])
                    if family == "dynamics":
                        self.assertTrue(validation.metrics["dynamics_changed"])
                        self.assertFalse(validation.metrics["metric_changed"])
                    if family == "semantic":
                        self.assertTrue(validation.metrics["semantic_changed"])
                        self.assertFalse(validation.metrics["topology_changed"])

    def test_minigrid_cpe_study_writes_reviewer_tables(self) -> None:
        with TemporaryDirectory() as tmpdir:
            result = run_minigrid_cpe_study(output_dir=Path(tmpdir), seeds_per_motif=1, severity_levels=(0, 2))
            health = result.summary["health_summary"]
            self.assertEqual(health["motif_count"], 6)
            self.assertEqual(health["validation_failures"], 0)
            self.assertEqual(health["reference_solvability"], 1.0)
            self.assertFalse(health["weak_baseline_saturated"])
            semantic = next(row for row in result.summary["protocol_sensitivity"] if row["family"] == "semantic")
            self.assertGreater(semantic["protocol_delta"], 0.0)
            self.assertTrue(Path(result.artifact_paths["protocol_sensitivity_md"]).exists())

    def test_optional_minigrid_backend_is_lazy(self) -> None:
        environment = MiniGridPortGenerator().generate(seed=102, motif_tag="corridor_bend").environment
        try:
            env = environment.to_minigrid_env()
        except MiniGridPortDependencyError:
            return
        observation, _info = env.reset(seed=102)
        self.assertIsNotNone(observation)


if __name__ == "__main__":
    unittest.main()
