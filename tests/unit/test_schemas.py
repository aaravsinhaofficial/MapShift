from __future__ import annotations

import unittest
from pathlib import Path

from mapshift.core.schemas import FAMILY_NAMES, TASK_CLASS_NAMES, load_release_bundle


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_CONFIG = REPO_ROOT / "configs" / "benchmark" / "release_v0_1.json"


class ReleaseSchemaTests(unittest.TestCase):
    def test_release_bundle_loads(self) -> None:
        bundle = load_release_bundle(ROOT_CONFIG)

        self.assertEqual(bundle.root.benchmark_name, "MapShift")
        self.assertEqual(bundle.root.primary_tier, "mapshift_2d")
        self.assertEqual(bundle.env2d.exploration.canonical_budget_steps, 800)
        self.assertEqual(bundle.env3d.platform, "ProcTHOR")
        self.assertEqual(tuple(bundle.interventions.families.keys()), FAMILY_NAMES)
        self.assertEqual(tuple(bundle.tasks.classes.keys()), TASK_CLASS_NAMES)

    def test_split_motifs_are_disjoint(self) -> None:
        bundle = load_release_bundle(ROOT_CONFIG)
        train = set(bundle.env2d.splits.train_motifs)
        val = set(bundle.env2d.splits.val_motifs)
        test = set(bundle.env2d.splits.test_motifs)

        self.assertFalse(train & val)
        self.assertFalse(train & test)
        self.assertFalse(val & test)


if __name__ == "__main__":
    unittest.main()
