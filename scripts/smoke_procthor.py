#!/usr/bin/env python3
"""Generate and summarize one ProcTHOR-compatible MapShift-3D scene."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mapshift.core.schemas import load_release_bundle
from mapshift.envs.procthor.generator import ProcTHORGenerator
from mapshift.envs.procthor.observation import observe_scene
from mapshift.envs.procthor.validation import validate_procthor_scene


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="configs/benchmark/release_v0_1.json")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--motif", default=None)
    args = parser.parse_args()

    bundle = load_release_bundle(args.config)
    generator = ProcTHORGenerator(bundle.env3d)
    result = generator.sample(seed=args.seed, motif_tag=args.motif)
    scene = result.scene
    if scene is None:
        print(json.dumps({"ok": False, "error": "generator_returned_no_scene"}, indent=2, sort_keys=True))
        return 1

    observation = observe_scene(scene)
    issues = validate_procthor_scene(scene)
    payload = {
        "ok": not issues,
        "scene_id": scene.scene_id,
        "motif_tag": scene.motif_tag,
        "split_name": scene.split_name,
        "room_count": scene.room_count(),
        "object_count": scene.object_count(),
        "goal_tokens": dict(scene.goal_tokens),
        "backend_status": dict(generator.backend_status),
        "observation": {
            "room_id": observation.room_id,
            "visible_objects": list(observation.visible_objects),
            "visible_categories": list(observation.visible_categories),
            "visible_semantic_labels": list(observation.visible_semantic_labels),
            "pose_token": observation.pose_token,
        },
        "validation_issues": issues,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
