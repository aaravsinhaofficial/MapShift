from __future__ import annotations

import unittest
from pathlib import Path

from mapshift.core.manifests import SplitManifest
from mapshift.core.schemas import load_release_bundle
from mapshift.envs.map2d.dynamics import DynamicsParameters2D
from mapshift.envs.map2d.state import AgentPose2D, Map2DEnvironment, Map2DNode
from mapshift.splits.builders import build_canonical_release_split_bundle, validate_release_split_artifacts
from mapshift.splits.leakage_checks import generate_leakage_report
from mapshift.splits.motifs import motif_tags_for_environment, semantic_template_metadata, structural_signature_for_environment


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_CONFIG = REPO_ROOT / "configs" / "benchmark" / "release_v0_1.json"


def handcrafted_loop_environment() -> Map2DEnvironment:
    nodes = {
        "n0": Map2DNode(node_id="n0", row=0, col=0),
        "n1": Map2DNode(node_id="n1", row=0, col=4),
        "n2": Map2DNode(node_id="n2", row=4, col=4),
        "n3": Map2DNode(node_id="n3", row=4, col=0),
    }
    occupancy = [[0 for _ in range(5)] for _ in range(5)]
    return Map2DEnvironment(
        environment_id="loop-env",
        motif_tag="hand_loop",
        split_name="train",
        seed=0,
        width_cells=5,
        height_cells=5,
        occupancy_grid=occupancy,
        nodes=nodes,
        adjacency={
            "n0": ["n1", "n3"],
            "n1": ["n0", "n2"],
            "n2": ["n1", "n3"],
            "n3": ["n0", "n2"],
        },
        room_cells={node_id: [node.cell] for node_id, node in nodes.items()},
        edge_corridors={
            "n0|n1": [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
            "n1|n2": [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4)],
            "n2|n3": [(4, 4), (4, 3), (4, 2), (4, 1), (4, 0)],
            "n0|n3": [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
        },
        start_node_id="n0",
        goal_node_id="n2",
        landmark_by_node={"n0": "red_tower", "n1": "blue_gate"},
        goal_tokens={"target_alpha": "n2", "target_beta": "n3"},
        dynamics=DynamicsParameters2D(),
        occupancy_resolution_m=1.0,
        geometry_scale=1.0,
        observation_radius_m=2.0,
        field_of_view_deg=180.0,
        semantic_channels=True,
        agent_pose=AgentPose2D(x=0.0, y=0.0, theta_deg=0.0),
        history=["handcrafted_loop"],
        metadata={"motif_family": "loop"},
    )


class SplitControlTests(unittest.TestCase):
    def test_motif_tagging_and_structural_signature_on_handcrafted_loop(self) -> None:
        environment = handcrafted_loop_environment()

        tags = motif_tags_for_environment(environment)
        signature = structural_signature_for_environment(environment)

        self.assertIn("loop", tags)
        self.assertGreater(signature.cycle_rank, 0)
        self.assertEqual(signature.connectivity_hash, structural_signature_for_environment(environment.clone()).connectivity_hash)

    def test_semantic_template_metadata_is_stable_and_changes_on_remap(self) -> None:
        environment = handcrafted_loop_environment()
        original = semantic_template_metadata(environment)
        stable_copy = semantic_template_metadata(environment.clone())

        self.assertEqual(original["semantic_template_id"], stable_copy["semantic_template_id"])

        remapped = environment.clone(environment_id="loop-env-remapped")
        remapped.goal_tokens["target_alpha"] = "n3"
        remapped.landmark_by_node["n0"] = "green_beacon"
        for key in (
            "landmark_layout_template_id",
            "goal_token_template_id",
            "semantic_template_id",
            "landmark_layout_template",
            "goal_token_template",
        ):
            remapped.metadata.pop(key, None)
        changed = semantic_template_metadata(remapped)

        self.assertNotEqual(original["semantic_template_id"], changed["semantic_template_id"])

    def test_canonical_split_bundle_is_deterministic(self) -> None:
        bundle = load_release_bundle(ROOT_CONFIG)
        first = build_canonical_release_split_bundle(bundle, sample_count_per_motif=1, task_samples_per_class=1)
        second = build_canonical_release_split_bundle(bundle, sample_count_per_motif=1, task_samples_per_class=1)

        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(len(first.manifests["train"].environment_ids), 10)
        self.assertEqual(len(first.manifests["val"].environment_ids), 6)
        self.assertEqual(len(first.manifests["test"].environment_ids), 8)

    def test_leakage_report_detects_duplicate_environment_entries(self) -> None:
        entry = {
            "environment_id": "dup-env",
            "motif_tag": "dup_motif",
            "motif_family": "loop",
            "geometry_hash": "geo123",
            "normalized_structural_fingerprint": "norm123",
            "semantic_template_id": "sem123",
            "goal_token_template_id": "goal123",
            "landmark_layout_template_id": "land123",
        }
        report = generate_leakage_report(
            environment_entries_by_split={"train": [entry], "val": [dict(entry)], "test": []},
            task_entries_by_split={},
            intervention_entries_by_split={},
        )

        categories = {finding.category for finding in report.errors}
        self.assertIn("motif_instance_overlap", categories)
        self.assertIn("structural_exact_overlap", categories)
        self.assertIn("semantic_template_overlap", categories)

    def test_split_health_summary_contains_expected_coverage_fields(self) -> None:
        bundle = load_release_bundle(ROOT_CONFIG)
        split_bundle = build_canonical_release_split_bundle(bundle, sample_count_per_motif=1, task_samples_per_class=1)

        train_summary = split_bundle.coverage_summary["train"]
        val_summary = split_bundle.coverage_summary["val"]
        test_summary = split_bundle.coverage_summary["test"]

        self.assertIn("motif_family_counts", train_summary)
        self.assertIn("semantic_template_counts", train_summary)
        self.assertIn("intervention_family_severity_counts", train_summary)
        self.assertEqual(train_summary["environment_count"], 10)
        self.assertEqual(len(train_summary["missing_task_cells"]), 0)
        self.assertEqual(len(val_summary["missing_task_cells"]), 0)
        self.assertEqual(len(test_summary["missing_task_cells"]), 0)

    def test_release_split_validation_fails_on_intentional_leakage(self) -> None:
        bundle = load_release_bundle(ROOT_CONFIG)
        split_bundle = build_canonical_release_split_bundle(bundle, sample_count_per_motif=1, task_samples_per_class=1)
        train_manifest = split_bundle.manifests["train"]
        val_manifest = split_bundle.manifests["val"]

        leaked_entry = dict(train_manifest.metadata["environment_entries"][0])
        leaked_val_manifest = SplitManifest(
            artifact_id=val_manifest.artifact_id,
            artifact_type=val_manifest.artifact_type,
            benchmark_version=val_manifest.benchmark_version,
            code_version=val_manifest.code_version,
            config_hash=val_manifest.config_hash,
            created_at=val_manifest.created_at,
            parent_ids=list(val_manifest.parent_ids),
            seed_values=list(val_manifest.seed_values),
            metadata={**val_manifest.metadata, "environment_entries": [leaked_entry]},
            split_name=val_manifest.split_name,
            tier=val_manifest.tier,
            environment_ids=[leaked_entry["environment_id"]],
            release_name=val_manifest.release_name,
        )
        leakage_report = generate_leakage_report(
            environment_entries_by_split={
                "train": train_manifest.metadata["environment_entries"],
                "val": leaked_val_manifest.metadata["environment_entries"],
                "test": split_bundle.manifests["test"].metadata["environment_entries"],
            },
            task_entries_by_split={
                "train": train_manifest.metadata["task_entries"],
                "val": leaked_val_manifest.metadata.get("task_entries", []),
                "test": split_bundle.manifests["test"].metadata["task_entries"],
            },
            intervention_entries_by_split={
                "train": train_manifest.metadata["intervention_entries"],
                "val": leaked_val_manifest.metadata.get("intervention_entries", []),
                "test": split_bundle.manifests["test"].metadata["intervention_entries"],
            },
        )
        issues = validate_release_split_artifacts(
            release_bundle=bundle,
            manifests={
                "train": train_manifest,
                "val": leaked_val_manifest,
                "test": split_bundle.manifests["test"],
            },
            leakage_report=leakage_report,
        )

        self.assertTrue(any("split leakage error" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
