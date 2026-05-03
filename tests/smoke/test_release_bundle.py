from __future__ import annotations

import unittest
from pathlib import Path

from mapshift.metrics.statistics import bootstrap_mean_interval
from mapshift.core.schemas import load_release_bundle


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_CONFIG = REPO_ROOT / "configs" / "benchmark" / "release_v0_1.json"


class ReleaseBundleSmokeTests(unittest.TestCase):
    def test_summary_contains_expected_keys(self) -> None:
        bundle = load_release_bundle(ROOT_CONFIG)
        summary = bundle.summary()

        self.assertIn("benchmark_name", summary)
        self.assertIn("release_name", summary)
        self.assertEqual(summary["baseline_count"], 6)

    def test_bootstrap_helper_returns_interval(self) -> None:
        point, lower, upper = bootstrap_mean_interval([0.2, 0.4, 0.6, 0.8], resamples=200, seed=7)

        self.assertGreaterEqual(point, lower)
        self.assertLessEqual(point, upper)


if __name__ == "__main__":
    unittest.main()
