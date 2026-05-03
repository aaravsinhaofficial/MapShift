# MapShift: A Benchmark for Counterfactual Embodied Planning

MapShift is an executable benchmark and evaluation protocol for testing whether environment knowledge acquired during reward-free exploration remains useful after a structured environmental intervention. In the Counterfactual Embodied Planning (CEP) protocol, an agent explores a base environment without task reward, the environment is changed along one controlled intervention family, and the agent is evaluated on post-intervention planning, inference, and adaptation tasks.

This repository is the code artifact for the current MapShift release. It contains the environment generator, intervention operators, task samplers, benchmark health checks, baselines, study runner, raw-record outputs, and paper table/figure renderers.

## Code-Release Summary

For NeurIPS E&D review, this repository should be supplied through the OpenReview **Code URL** field. It is an executable benchmark/generator, not a hosted static dataset. Do not check the OpenReview dataset box unless you separately package generated instances as a dataset with the required metadata. Written appendices belong in the paper PDF, not in supplementary material.

| Requirement | Where it is handled |
|---|---|
| Reviewer smoke command | `python3 scripts/build_benchmark.py --tier mapshift_2d --study-config configs/analysis/mapshift_2d_full_study_smoke_v0_1.json --output-dir outputs/releases/mapshift_2d_v0_1_smoke --print-summary` |
| Full reproduction command | `python3 scripts/build_benchmark.py --tier mapshift_2d --study-config configs/analysis/mapshift_2d_full_study_v0_1.json --output-dir outputs/releases/mapshift_2d_v0_1_full --print-summary` |
| Expected runtime | Smoke: minutes on CPU. Full: single-GPU run recommended; the reference L4 run should be treated as an overnight job. |
| CPU/GPU requirements | CPU works for validation and smoke. Full study works on CPU but is intended for a CUDA-capable PyTorch install when available. |
| Disk usage | Reserve 5GB for generated outputs and checkpoints; reserve more if installing CUDA PyTorch wheels into a fresh environment. |
| Output paths | `outputs/releases/<run_name>/health`, `study`, `paper_outputs`, `manifests`, and `logs`. |
| Tables/figures regenerate | `python3 scripts/render_paper_outputs.py <study_bundle.json> --output-dir <paper_outputs> --print-summary` |
| Version/config hash | Package version `0.1.0`; schema version `0.1.0`; release manifest records `config_hash`. |
| Dependency installation | `python3 -m pip install -e .` from a fresh Python 3.10+ environment. |
| Baseline hyperparameters | See `configs/calibration/*.json` and the table below. |
| Raw record schema | See the "Raw Record Schema" section below. |

## Installation

Use Python 3.10 or newer. The package dependencies are bounded in `pyproject.toml`:

- `numpy>=1.26,<3.0`
- `torch>=2.2,<3.0`

CPU install:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

CUDA install:

Install a CUDA-enabled PyTorch build appropriate for the host first if needed, then install MapShift in editable mode:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Verify the selected Torch device:

```bash
python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
print("device_0:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

To force the learned baselines onto a particular device and isolate checkpoints:

```bash
export MAPSHIFT_TORCH_DEVICE=cuda:0
export MAPSHIFT_CHECKPOINT_DIR=/tmp/mapshift_learned_baselines_full_v0_1
```

If `MAPSHIFT_TORCH_DEVICE` is unset, learned baselines use their config value. The release configs use `torch_device: "auto"`, which selects CUDA only when PyTorch reports that CUDA is available.

## Quick Validation

Validate the release config bundle:

```bash
python3 scripts/validate_benchmark.py \
  --tier mapshift_2d \
  configs/benchmark/release_v0_1.json
```

Generate a benchmark health preflight:

```bash
python3 scripts/generate_benchmark_health.py \
  configs/benchmark/release_v0_1.json \
  --samples-per-motif 1 \
  --task-samples-per-class 3 \
  --output outputs/releases/mapshift_2d_v0_1_health_preflight.json
```

Run tests:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

## Reviewer Smoke Run

The smoke command builds a small release artifact: config copies, split manifests, benchmark health, a learned-baseline smoke study, JSON tables, SVG figures, and manifests.

```bash
python3 scripts/build_benchmark.py \
  --tier mapshift_2d \
  --study-config configs/analysis/mapshift_2d_full_study_smoke_v0_1.json \
  --output-dir outputs/releases/mapshift_2d_v0_1_smoke \
  --print-summary
```

Expected runtime: a few minutes on a typical laptop or CPU VM. It is intended for reviewer sanity checking, not for reproducing paper-scale confidence intervals.

Expected outputs:

```text
outputs/releases/mapshift_2d_v0_1_smoke/
  logs/build_benchmark.log
  health/benchmark_health.json
  study/study_bundle.json
  study/raw/cep_report.json
  study/raw/protocol_comparison_report.json
  study/raw/benchmark_health_report.json
  study/tables/*.json
  study/figures/*.json
  paper_outputs/tables/*.md
  paper_outputs/figures/*.svg
  manifests/release_manifest.json
```

## Full Reproduction Run

The full command regenerates the paper-facing results from the release configs:

```bash
python3 scripts/build_benchmark.py \
  --tier mapshift_2d \
  --study-config configs/analysis/mapshift_2d_full_study_v0_1.json \
  --output-dir outputs/releases/mapshift_2d_v0_1_full \
  --print-summary
```

Recommended hardware:

- CPU: enough for validation, health checks, and smoke.
- GPU: a single CUDA GPU is recommended for the full study. The reference run was launched on one NVIDIA L4 with 23GB memory.
- RAM: 16GB is recommended for the full build because raw records are accumulated before the study bundle is written.
- Disk: reserve at least 5GB for outputs, logs, and checkpoints. A fresh CUDA PyTorch install may require several additional GB in the virtual environment or pip cache.

Expected runtime:

- Smoke: minutes.
- Full single-GPU study: overnight-scale. Runtime is dominated by task generation, evaluation loops, bootstrap aggregation, and rendering. The learned training jobs are small.
- Full CPU-only study: possible but not recommended for deadline-sensitive reproduction.

Monitor progress:

```bash
tail -f outputs/releases/mapshift_2d_v0_1_full/logs/build_benchmark.log
```

Count evaluation chunks:

```bash
grep -c "evaluating family=" outputs/releases/mapshift_2d_v0_1_full/logs/build_benchmark.log
```

The full config currently runs one primary CEP sweep plus five protocol-comparison sweeps. Each sweep has 8 motifs x 4 intervention families = 32 logged family chunks, so a completed run should log roughly 192 `evaluating family=` lines.

Check completion:

```bash
grep -E "Study complete|Study bundle written|Rendered|Release manifest written" \
  outputs/releases/mapshift_2d_v0_1_full/logs/build_benchmark.log
```

Core result files are written after the long study stage completes:

```text
outputs/releases/mapshift_2d_v0_1_full/study/raw/cep_report.json
outputs/releases/mapshift_2d_v0_1_full/study/raw/protocol_comparison_report.json
outputs/releases/mapshift_2d_v0_1_full/study/tables/familywise_main_results.json
outputs/releases/mapshift_2d_v0_1_full/study/tables/severity_response.json
outputs/releases/mapshift_2d_v0_1_full/study/tables/protocol_sensitivity_and_rank_correlation.json
outputs/releases/mapshift_2d_v0_1_full/paper_outputs/tables/main_familywise_results.md
outputs/releases/mapshift_2d_v0_1_full/paper_outputs/figures/familywise_main_results.svg
outputs/releases/mapshift_2d_v0_1_full/paper_outputs/figures/severity_response_curves.svg
outputs/releases/mapshift_2d_v0_1_full/paper_outputs/figures/protocol_rank_reversal_comparison.svg
```

## Regenerating Tables and Figures

After a smoke or full run has produced `study/study_bundle.json`, regenerate paper-facing Markdown tables and SVG figures without rerunning evaluation:

```bash
python3 scripts/render_paper_outputs.py \
  outputs/releases/mapshift_2d_v0_1_full/study/study_bundle.json \
  --output-dir outputs/releases/mapshift_2d_v0_1_full/paper_outputs \
  --print-summary
```

This writes:

```text
paper_outputs/tables/main_familywise_results.md
paper_outputs/tables/benchmark_health_summary.md
paper_outputs/tables/baseline_metadata.md
paper_outputs/tables/protocol_sensitivity_summary.md
paper_outputs/figures/cep_protocol_diagram.svg
paper_outputs/figures/intervention_examples_2d.svg
paper_outputs/figures/familywise_main_results.svg
paper_outputs/figures/severity_response_curves.svg
paper_outputs/figures/protocol_rank_reversal_comparison.svg
paper_outputs/paper_outputs_manifest.json
```

To render only the data-free protocol diagram:

```bash
python3 scripts/render_paper_outputs.py \
  --protocol-diagram-only \
  --output-dir outputs/figure1 \
  --print-summary
```

## Release Version and Config Hash

Primary version identifiers:

- Package version: `0.1.0` in `pyproject.toml`
- Config schema version: `0.1.0` in `configs/**/*.json`
- Release identity: `mapshift_2d_v0_1` in `configs/benchmark/release_v0_1.json`
- Primary tier: `mapshift_2d`

The build writes a release manifest with a config hash:

```text
outputs/releases/<run_name>/manifests/release_manifest.json
```

Inspect the manifest:

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path("outputs/releases/mapshift_2d_v0_1_full/manifests/release_manifest.json")
m = json.loads(p.read_text())
print("artifact_id:", m["artifact_id"])
print("benchmark_version:", m["benchmark_version"])
print("config_hash:", m["config_hash"])
print("tier:", m["metadata"]["tier"])
PY
```

## Baselines and Hyperparameters

The main study uses seven scientific baseline classes. Deterministic reference baselines have no trainable parameters.

| Baseline | Config | Key hyperparameters |
|---|---|---|
| `oracle_post_intervention_planner` | `configs/calibration/oracle_post_intervention_planner_v0_1.json` | oracle access; no training |
| `same_environment_upper_baseline` | `configs/calibration/same_environment_upper_baseline_v0_1.json` | same-environment reference; no training |
| `weak_heuristic_baseline` | `configs/calibration/weak_heuristic_baseline_v0_1.json` | visited-state heuristic; no training |
| `monolithic_recurrent_world_model` | `configs/calibration/monolithic_recurrent_world_model_v0_1.json` | hidden size 12, observation stride 2, 6 epochs, lr 0.01, max rollout 8 |
| `persistent_memory_world_model` | `configs/calibration/persistent_memory_world_model_v0_1.json` | 16 memory slots, slot stride 3, readout width 8, 8 epochs, lr 0.01, max rollout 8 |
| `relational_graph_world_model` | `configs/calibration/relational_graph_world_model_v0_1.json` | hidden size 10, 2 message-passing steps, 10 epochs, lr 0.01 |
| `structured_dynamics_world_model` | `configs/calibration/structured_dynamics_world_model_v0_1.json` | geometry width 10, dynamics width 6, 10 epochs, lr 0.01 |

Learned baselines train separately for each base environment and model seed from the reward-free exploration trace. They are not pretrained across environments. The full study expands learned baselines over seeds `[0, 1, 2, 3, 4]` using `configs/analysis/mapshift_2d_full_study_v0_1.json`.

Training targets are derived from the exploration-time graph representation:

- edge existence
- normalized geometric path cost
- normalized traversal cost
- semantic-token location

The shared learned-baseline loss combines edge binary cross-entropy, geometry mean-squared error, traversal mean-squared error, and token binary cross-entropy. Checkpoints are keyed by baseline name, environment id, model seed, and hyperparameter hash.

## Raw Record Schema

Raw episode records are written in:

```text
outputs/releases/<run_name>/study/raw/cep_report.json
outputs/releases/<run_name>/study/raw/protocol_comparison_report.json
```

In `cep_report.json`, records are stored at:

```text
records[]
```

In `protocol_comparison_report.json`, records are stored at:

```text
protocol_reports.<protocol_name>.records[]
```

Each record has the schema below. The implementation source is `mapshift/runners/evaluate.py::EvaluationRecord`.

| Field | Type | Meaning |
|---|---|---|
| `baseline_name` | string | Scientific baseline class. |
| `baseline_run_id` | string | Concrete run id, including seed-expanded runs. |
| `run_name` | string | Alias for the concrete run used in manifests. |
| `protocol_name` | string | `cep`, `same_environment`, `no_exploration`, `short_horizon`, or `long_horizon`. |
| `family` | string | Intervention family: `metric`, `topology`, `dynamics`, or `semantic`. |
| `severity` | integer | Severity level 0, 1, 2, or 3. |
| `split_name` | string | `train`, `val`, or `test`. |
| `motif_tag` | string | Structural motif id. |
| `task_class` | string | `planning`, `inference`, or `adaptation`. |
| `task_type` | string | Concrete task type sampled within the class. |
| `environment_id` | string | Evaluated environment id. |
| `base_environment_id` | string | Pre-intervention environment id. |
| `task_id` | string | Deterministic task identifier. |
| `model_seed` | integer | Model/random seed for the baseline run. |
| `environment_model_seed_id` | string | Bootstrap grouping key combining environment and baseline run. |
| `task_horizon_steps` | integer | Evaluation horizon after protocol multipliers. |
| `success` | boolean | Baseline task success. |
| `solvable` | boolean | Whether the sampled task is oracle-solvable. |
| `primary_score` | number | Task-normalized primary score for the record. |
| `observed_length` | number or null | Baseline path length when applicable. |
| `oracle_length` | number or null | Oracle path length when applicable. |
| `path_efficiency` | number | Path-efficiency score. |
| `oracle_gap` | number or null | Gap to oracle path length/score when applicable. |
| `oracle_success` | boolean | Oracle success on the same task. |
| `oracle_primary_score` | number | Oracle primary score for the same task. |
| `predicted_answer` | any | Baseline answer for inference-style tasks. |
| `expected_answer` | any | Expected answer for inference-style tasks. |
| `correct` | boolean or null | Whether the answer is correct when applicable. |
| `adaptation_curve` | array[number] | Recovery/score curve for adaptation tasks. |
| `metadata` | object | Task- and baseline-specific diagnostic metadata. |

Quick raw-record inspection:

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path("outputs/releases/mapshift_2d_v0_1_full/study/raw/cep_report.json")
report = json.loads(p.read_text())
print("records:", len(report["records"]))
print("first keys:", sorted(report["records"][0]))
print("first record:", json.dumps(report["records"][0], indent=2)[:2000])
PY
```

## Benchmark Design

The current release uses:

- 96x96 occupancy maps
- `T_exp=800` reward-free exploration steps
- eight structural motifs
- motif-level train/validation/test splits
- four intervention families: metric, topology, dynamics, semantic
- severity levels 0, 1, 2, 3
- planning, inference, and adaptation tasks
- 1000 bootstrap resamples with grouping by `environment_model_seed_id`

The 3D-compatible files remain in the repository as prototype/future-work code and are not required for the current release claims. Use `--tier mapshift_2d` for validation and artifact building so prototype 3D status does not block the primary artifact.

## Repository Structure

```text
mapshift/
  core/              Config schemas, manifests, logging, registry
  envs/map2d/        Map generator, dynamics, renderer
  interventions/     Metric, topology, dynamics, semantic interventions
  tasks/             Planning, inference, adaptation task definitions/samplers
  baselines/         Baseline API and implementations
  runners/           Exploration, evaluation, protocol comparisons
  metrics/           Planning/inference/adaptation metrics
  analysis/          Study orchestration, bootstrap, ranking, figure data
  splits/            Motif splits and leakage checks

configs/
  benchmark/         Top-level release config
  env2d/             Environment config
  interventions/     Family definitions and severity ladders
  tasks/             Task classes and sampling
  calibration/       Per-baseline run configs
  analysis/          Smoke and full study configs
  schema/            JSON schemas

scripts/
  validate_benchmark.py
  generate_benchmark_health.py
  build_benchmark.py
  render_paper_outputs.py
  run_mapshift_2d_study.py
  run_protocol_comparison.py

docs/
  benchmark_spec.md
  release_freeze_v0_1.md
  evaluation_card.md
  release_checklist.md
```

## Troubleshooting

CUDA is not used:

```bash
python3 - <<'PY'
import torch
print(torch.__version__, torch.cuda.is_available())
PY
```

If CUDA is available but MapShift uses CPU, set:

```bash
export MAPSHIFT_TORCH_DEVICE=cuda:0
```

Checkpoint reuse is confusing:

```bash
export MAPSHIFT_CHECKPOINT_DIR=/tmp/mapshift_learned_baselines_new_run
```

Build appears stalled:

```bash
tail -f outputs/releases/mapshift_2d_v0_1_full/logs/build_benchmark.log
grep -c "evaluating family=" outputs/releases/mapshift_2d_v0_1_full/logs/build_benchmark.log
```

Large output directory:

The raw JSON records and protocol comparison reports are the largest generated artifacts. Remove or archive `outputs/releases/<run_name>` after copying the files needed for submission. Learned checkpoints are in `MAPSHIFT_CHECKPOINT_DIR` or the config default `/tmp/mapshift_learned_baselines`.

Validation fails due to 3D prototype:

Use the primary tier path:

```bash
python3 scripts/validate_benchmark.py --tier mapshift_2d configs/benchmark/release_v0_1.json
```

## License

This project is licensed under the [MIT License](LICENSE).
