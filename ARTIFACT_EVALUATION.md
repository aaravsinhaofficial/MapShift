# MapShift Artifact Evaluation Guide

This guide is the shortest path for reviewers to verify that the MapShift code artifact is installable, executable, and able to regenerate the paper-facing outputs. The artifact is an executable benchmark/generator, not a hosted static dataset.

## Environment

Use Python 3.10 or newer. A CPU environment is sufficient for validation, unit tests, the reviewer smoke build, and the deterministic mechanism diagnostic. A CUDA-capable PyTorch install is recommended for the full reproduction run.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Optional reviewed-environment pins are provided in `requirements-lock.txt`:

```bash
python -m pip install -r requirements-lock.txt
```

For CUDA runs, verify PyTorch sees the GPU:

```bash
python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
print("device_0:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

To force learned baselines onto a GPU and isolate checkpoints:

```bash
export MAPSHIFT_TORCH_DEVICE=cuda:0
export MAPSHIFT_CHECKPOINT_DIR=/tmp/mapshift_learned_baselines_review
```

## One-Command Audit

The quick audit validates configs, runs tests, builds the reviewer smoke artifact, checks the generated manifest, verifies benchmark-health gates, and renders paper-facing outputs:

```bash
python3 scripts/audit_artifact.py --quick --output-dir outputs/audit/mapshift_quick
```

Expected result: the command exits with status 0 and prints `Artifact audit passed`.

## Reviewer Smoke Build

The smoke build is the main low-cost artifact check:

```bash
python3 scripts/build_benchmark.py \
  --tier mapshift_2d \
  --study-config configs/analysis/mapshift_2d_full_study_smoke_v0_1.json \
  --output-dir outputs/releases/mapshift_2d_v0_1_smoke \
  --print-summary
```

Expected runtime: minutes on a laptop or CPU VM. It writes:

```text
outputs/releases/mapshift_2d_v0_1_smoke/
  logs/build_benchmark.log
  health/benchmark_health.json
  study/study_bundle.json
  study/raw/cep_report.json
  study/raw/protocol_comparison_report.json
  study/tables/*.json
  paper_outputs/tables/*.md
  paper_outputs/figures/*.svg
  manifests/release_manifest.json
```

## Deterministic Mechanism Diagnostic

This run reproduces the stale-map, weak-heuristic, and classical belief-update diagnostic reported in the paper:

```bash
python3 scripts/run_mapshift_2d_study.py \
  configs/analysis/mapshift_2d_belief_update_diagnostic_v0_1.json \
  --print-summary
```

Expected output directory:

```text
outputs/studies/mapshift_2d_belief_update_diagnostic_v0_1/
```

Inspect the reported diagnostic:

```bash
python3 - <<'PY'
import json
from pathlib import Path

b = json.loads(Path("outputs/studies/mapshift_2d_belief_update_diagnostic_v0_1/study_bundle.json").read_text())
print(json.dumps(b["proposition_support"], indent=2))
for row in b["raw_reports"]["cep_report"]["familywise_summary"]["rows"]:
    print(row["baseline_name"], row["family"], round(row["family_primary_score"], 3))
PY
```

Generate the held-out consistency and paired-bootstrap delta tables for the stale-map versus belief-update claims:

```bash
python3 scripts/analyze_mechanism_diagnostic.py \
  outputs/studies/mapshift_2d_belief_update_diagnostic_v0_1/study_bundle.json \
  --output-dir outputs/studies/mapshift_2d_belief_update_diagnostic_v0_1/mechanism_diagnostic_analysis \
  --split test \
  --family topology \
  --family semantic \
  --resamples 1000 \
  --print-summary
```

The key outputs are:

```text
mechanism_diagnostic_analysis/tables/heldout_motif_summary.md
mechanism_diagnostic_analysis/tables/heldout_motif_consistency.md
mechanism_diagnostic_analysis/tables/paired_delta_bootstrap.md
```

## High-Capacity Learned World-Model Add-On

This optional add-on evaluates only the 1.14M-parameter pretrained structured graph world model on the CEP grid. It does not rerun the older baselines. An implicit oracle reference is still evaluated internally so oracle fields in the raw records remain populated. The generated report contains all severities; the paper table reports the non-identity subset for this high-capacity learned row.

```bash
export MAPSHIFT_TORCH_DEVICE=cuda:0
export MAPSHIFT_CHECKPOINT_DIR=/tmp/mapshift_pretrained_graph_world_model_1m_v0_1

python3 scripts/generate_calibration_report.py \
  configs/benchmark/release_v0_1.json \
  --tier mapshift_2d \
  --run-config configs/calibration/pretrained_structured_graph_world_model_1m_v0_1.json \
  --model-seed 0 \
  --samples-per-motif 1 \
  --task-samples-per-class 3 \
  --output outputs/studies/pretrained_structured_graph_world_model_1m_v0_1/cep_report.json \
  --log-file outputs/studies/pretrained_structured_graph_world_model_1m_v0_1/logs/run.log \
  --print-summary
```

Expected scale: the learned model contributes 3456 all-severity episode records, of which 2592 are non-identity severity episodes used for the paper row. The implicit oracle contributes 3456 additional reference records. The high-capacity config has approximately 1.14M trainable parameters.

Extract the family-wise learned-baseline row:

```bash
python3 - <<'PY'
import json
from pathlib import Path

from mapshift.runners.evaluate import EvaluationRecord, _aggregate_metric_rows, _long_horizon_threshold

p = Path("outputs/studies/pretrained_structured_graph_world_model_1m_v0_1/cep_report.json")
payload = json.loads(p.read_text())
records = [
    EvaluationRecord(**{**record, "adaptation_curve": tuple(record.get("adaptation_curve", ()))})
    for record in payload["records"]
    if record["baseline_name"] == "pretrained_structured_graph_world_model" and int(record["severity"]) > 0
]
rows = _aggregate_metric_rows(records, ("protocol_name", "baseline_name", "family"), _long_horizon_threshold(records))
for row in rows:
    print(row["baseline_name"], row["family"], round(row["family_primary_score"], 3), "episodes=", row["episode_count"])
PY
```

## Full Reproduction

The full run regenerates the paper-facing tables, figures, raw records, protocol comparisons, benchmark health reports, and provenance manifest:

```bash
python3 scripts/build_benchmark.py \
  --tier mapshift_2d \
  --study-config configs/analysis/mapshift_2d_full_study_v0_1.json \
  --output-dir outputs/releases/mapshift_2d_v0_1_full \
  --print-summary
```

Expected runtime: the expanded full study should be budgeted at roughly 30-36 wall-clock hours on one NVIDIA L4 GPU with 23GB memory, with faster completion expected on L40S/H100-class GPUs. CPU-only full reproduction is possible but not recommended for deadline-sensitive review.

Monitor progress:

```bash
tail -f outputs/releases/mapshift_2d_v0_1_full/logs/build_benchmark.log
grep -c "evaluating family=" outputs/releases/mapshift_2d_v0_1_full/logs/build_benchmark.log
```

The full config runs one primary CEP sweep plus four protocol-comparison sweeps. Each sweep has 24 motifs x 4 families, so a completed run logs roughly 480 `evaluating family=` chunks.

## Paper Output Mapping

| Paper item | Reproducible source |
|---|---|
| Benchmark health table | `outputs/releases/mapshift_2d_v0_1_full/study/tables/benchmark_health_summary.json` |
| Main family-wise results | `outputs/releases/mapshift_2d_v0_1_full/study/tables/familywise_main_results.json` |
| Full-run protocol sensitivity | `outputs/releases/mapshift_2d_v0_1_full/study/tables/protocol_sensitivity_and_rank_correlation.json` |
| Severity-response curves | `outputs/releases/mapshift_2d_v0_1_full/study/tables/severity_response.json` and `paper_outputs/figures/severity_response_curves.svg` |
| Deterministic mechanism diagnostic | `outputs/studies/mapshift_2d_belief_update_diagnostic_v0_1/study_bundle.json` |
| High-capacity learned add-on | `outputs/studies/pretrained_structured_graph_world_model_1m_v0_1/cep_report.json` |
| Raw episode records | `study/raw/cep_report.json` and `study/raw/protocol_comparison_report.json` |
| Rendered tables/figures | `outputs/releases/<run_name>/paper_outputs/` |

Regenerate rendered Markdown/SVG outputs from an existing bundle:

```bash
python3 scripts/render_paper_outputs.py \
  outputs/releases/mapshift_2d_v0_1_full/study/study_bundle.json \
  --output-dir outputs/releases/mapshift_2d_v0_1_full/paper_outputs \
  --print-summary
```

## Health Gates

A valid artifact run should report:

- split validation ok
- fatal leakage count 0
- benchmark-health validator failures 0
- task coverage present for all configured cells
- oracle solvability reported in the health summary
- paper-facing tables and figures rendered

If a full run differs numerically across platforms, compare the generated `manifests/release_manifest.json`, `study/study_bundle.json`, and `logs/build_benchmark.log` to identify config, seed, dependency, or device changes.
