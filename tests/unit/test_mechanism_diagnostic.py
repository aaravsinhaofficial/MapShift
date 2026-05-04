import unittest

from mapshift.analysis.mechanism_diagnostic import (
    MechanismDiagnosticConfig,
    analyze_mechanism_diagnostic_bundle,
)


def record(
    *,
    baseline_name: str,
    protocol_name: str,
    motif_tag: str,
    family: str,
    success: bool,
) -> dict:
    return {
        "baseline_name": baseline_name,
        "baseline_run_id": baseline_name,
        "run_name": baseline_name,
        "protocol_name": protocol_name,
        "family": family,
        "severity": 3,
        "split_name": "test",
        "motif_tag": motif_tag,
        "task_class": "planning",
        "task_type": "counterfactual_path",
        "environment_id": f"{motif_tag}-{family}-s3",
        "base_environment_id": f"{motif_tag}-seed100",
        "task_id": f"{motif_tag}-{family}-{protocol_name}-{baseline_name}",
        "model_seed": 0,
        "environment_model_seed_id": f"{motif_tag}-seed100|{baseline_name}",
        "task_horizon_steps": 64,
        "success": success,
        "solvable": True,
        "primary_score": 1.0 if success else 0.0,
        "observed_length": 10.0 if success else None,
        "oracle_length": 10.0,
        "path_efficiency": 1.0 if success else 0.0,
        "oracle_gap": 0.0 if success else None,
        "oracle_success": True,
        "oracle_primary_score": 1.0,
        "adaptation_curve": [],
        "metadata": {},
    }


class MechanismDiagnosticAnalysisTests(unittest.TestCase):
    def test_heldout_consistency_and_paired_bootstrap(self) -> None:
        records = []
        for motif in ("test_a", "test_b"):
            for family in ("topology", "semantic"):
                records.extend(
                    [
                        record(
                            baseline_name="classical_belief_update_planner",
                            protocol_name="cep",
                            motif_tag=motif,
                            family=family,
                            success=True,
                        ),
                        record(
                            baseline_name="stale_map_planner",
                            protocol_name="cep",
                            motif_tag=motif,
                            family=family,
                            success=False,
                        ),
                        record(
                            baseline_name="classical_belief_update_planner",
                            protocol_name="same_environment",
                            motif_tag=motif,
                            family=family,
                            success=False,
                        ),
                        record(
                            baseline_name="stale_map_planner",
                            protocol_name="same_environment",
                            motif_tag=motif,
                            family=family,
                            success=True,
                        ),
                    ]
                )

        payload = {
            "protocol_reports": {
                "cep": {"records": [item for item in records if item["protocol_name"] == "cep"]},
                "same_environment": {
                    "records": [item for item in records if item["protocol_name"] == "same_environment"]
                },
            }
        }
        analysis = analyze_mechanism_diagnostic_bundle(
            payload,
            config=MechanismDiagnosticConfig(bootstrap_resamples=100, families=("topology", "semantic")),
        )

        self.assertEqual(analysis["record_count"], 16)
        self.assertEqual(len(analysis["heldout_motif_consistency"]["rows"]), 4)
        for row in analysis["heldout_motif_consistency"]["rows"]:
            self.assertGreater(row["belief_update_minus_stale_cep"], 0.0)
            self.assertLess(row["belief_update_minus_stale_same_environment"], 0.0)
            self.assertTrue(row["reversal_or_substantial_reduction"])

        bootstrap_rows = analysis["paired_delta_bootstrap"]["rows"]
        self.assertEqual(len(bootstrap_rows), 6)
        reversal_rows = [row for row in bootstrap_rows if row["contrast"] == "protocol_reversal_delta"]
        self.assertEqual(len(reversal_rows), 2)
        for row in reversal_rows:
            self.assertGreater(row["point_estimate"], 0.0)
            self.assertGreater(row["lower"], 0.0)
            self.assertEqual(row["unit_count"], 2)


if __name__ == "__main__":
    unittest.main()
