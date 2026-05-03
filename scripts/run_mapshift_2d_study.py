#!/usr/bin/env python3
"""Run the first full MapShift-2D benchmark study bundle."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mapshift.analysis.study import (
    load_mapshift_2d_study_config,
    run_mapshift_2d_study,
    write_mapshift_2d_study_bundle,
)


def _configure_study_logging(log_file: Path) -> logging.Logger:
    """Configure console and file logging for long-running study jobs."""

    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    return logging.getLogger("mapshift.run_mapshift_2d_study")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "study_config",
        nargs="?",
        default="configs/analysis/mapshift_2d_full_study_v0_1.json",
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--log-file", default="", help="Study log path. Defaults to <output-dir>/logs/run_mapshift_2d_study.log.")
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args()

    study_config = load_mapshift_2d_study_config(args.study_config)
    output_dir = Path(args.output_dir or study_config.output_subdir or f"outputs/studies/{study_config.study_name}").resolve()
    log_path = Path(args.log_file).resolve() if args.log_file else output_dir / "logs" / "run_mapshift_2d_study.log"
    logger = _configure_study_logging(log_path)
    logger.info("Starting MapShift study run")
    logger.info("Study config: %s", Path(args.study_config).resolve())
    logger.info("Output directory: %s", output_dir)
    logger.info("Study log: %s", log_path)
    logger.info("MAPSHIFT_TORCH_DEVICE=%s", os.environ.get("MAPSHIFT_TORCH_DEVICE", "<unset>"))
    logger.info("MAPSHIFT_CHECKPOINT_DIR=%s", os.environ.get("MAPSHIFT_CHECKPOINT_DIR", "<unset>"))
    logger.info(
        "Study grid: baseline_configs=%d samples_per_motif=%d task_samples_per_class=%d severity_levels=%s protocols=%s motifs=%s families=%s",
        len(study_config.baseline_run_configs),
        study_config.sample_count_per_motif,
        study_config.task_samples_per_class,
        list(study_config.severity_levels),
        list(study_config.protocol_names),
        list(study_config.motif_tags) if study_config.motif_tags else "<all>",
        list(study_config.family_names) if study_config.family_names else "<all>",
    )
    bundle = run_mapshift_2d_study(study_config)
    artifact_paths = write_mapshift_2d_study_bundle(bundle, output_dir)
    logger.info("Study artifacts written: %s", output_dir)
    logger.info(
        "Study finished: P1=%s P2=%s P3=%s",
        bundle.proposition_support["P1"]["status"],
        bundle.proposition_support["P2"]["status"],
        bundle.proposition_support["P3"]["status"],
    )

    if args.print_summary:
        summary = {
            "study_name": bundle.study_name,
            "release_name": bundle.release_name,
            "p1_status": bundle.proposition_support["P1"]["status"],
            "p2_status": bundle.proposition_support["P2"]["status"],
            "p3_status": bundle.proposition_support["P3"]["status"],
            "log_file": str(log_path),
            "artifact_paths": artifact_paths,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
