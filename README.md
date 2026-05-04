# MapShift: Controlled Post-Intervention Evaluation for Embodied Planning

MapShift is an executable benchmark and evaluation protocol for testing whether environment knowledge acquired during reward-free exploration remains useful after a structured environmental intervention. In the controlled post-intervention evaluation (CEP) protocol, an agent explores a base environment without task reward, the environment is changed along one controlled intervention family, and the agent is evaluated on post-intervention planning, inference, and adaptation tasks.

This repository is the code artifact for the current MapShift release. It contains the environment generator, intervention operators, task samplers, benchmark health checks, baselines, study runner, raw-record outputs, and paper table/figure renderers.

## Artifact Overview

This repository is a self-contained executable artifact. It regenerates benchmark instances, health checks, raw evaluation records, summary tables, rendered figures, and provenance manifests from versioned source code and configuration files. Generated instances are benchmark artifacts produced by the code, not a separately hosted static dataset.

| Artifact component | Location or command |
|---|---|
| Smoke command | `python3 scripts/build_benchmark.py --tier mapshift_2d --study-config configs/analysis/mapshift_2d_full_study_smoke_v0_1.json --output-dir outputs/releases/mapshift_2d_v0_1_smoke --print-summary` |
| One-command artifact audit | `python3 scripts/audit_artifact.py --quick --output-dir outputs/audit/mapshift_quick` |
| Full reproduction command | `python3 scripts/build_benchmark.py --tier mapshift_2d --study-config configs/analysis/mapshift_2d_full_study_v0_1.json --output-dir outputs/releases/mapshift_2d_v0_1_full --print-summary` |
| Deterministic mechanism diagnostic | `python3 scripts/run_mapshift_2d_study.py configs/analysis/mapshift_2d_belief_update_diagnostic_v0_1.json --print-summary` |
| High-capacity world-model add-on | `python3 scripts/generate_calibration_report.py configs/benchmark/release_v0_1.json --tier mapshift_2d --run-config configs/calibration/pretrained_structured_graph_world_model_1m_v0_1.json --model-seed 0 --samples-per-motif 1 --task-samples-per-class 3 --output outputs/studies/pretrained_structured_graph_world_model_1m_v0_1/cep_report.json --log-file outputs/studies/pretrained_structured_graph_world_model_1m_v0_1/logs/run.log --print-summary` |
| Expected runtime | Smoke: minutes on CPU. Deterministic diagnostic: short CPU/GPU run. Full expanded study: single-GPU run recommended; budget roughly 30-36 wall-clock hours on one NVIDIA L4, with faster completion expected on L40S/H100-class GPUs. |
| CPU/GPU requirements | CPU works for validation and smoke. Full study works on CPU but is intended for a CUDA-capable PyTorch install when available. |
| Disk usage | Reserve 5GB for generated outputs and checkpoints; reserve more if installing CUDA PyTorch wheels into a fresh environment. |
| Output paths | `outputs/releases/<run_name>/health`, `study`, `paper_outputs`, `manifests`, and `logs`. |
| Tables/figures regenerate | `python3 scripts/render_paper_outputs.py <study_bundle.json> --output-dir <paper_outputs> --print-summary` |
| Version/config hash | Package version `0.1.0`; schema version `0.1.0`; release manifest records `config_hash`. |
| Dependency installation | `python3 -m pip install -e .` from a fresh Python 3.10+ environment. |
| Reviewed dependency pins | `requirements-lock.txt` |
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

Reviewed-environment pins are provided for reviewers who want a tighter dependency target:

```bash
python -m pip install -r requirements-lock.txt
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

For the artifact runbook, see [`ARTIFACT_EVALUATION.md`](ARTIFACT_EVALUATION.md).

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

Run the one-command artifact audit:

```bash
python3 scripts/audit_artifact.py \
  --quick \
  --output-dir outputs/audit/mapshift_quick
```

The audit validates the release config, runs tests, builds the reviewer smoke artifact, verifies required output paths, checks zero fatal leakage and zero validator failures, and confirms paper-facing rendered outputs exist.

## Smoke Run

The smoke command builds a small release artifact: config copies, split manifests, benchmark health, a learned-baseline smoke study, JSON tables, SVG figures, and manifests.

```bash
python3 scripts/build_benchmark.py \
  --tier mapshift_2d \
  --study-config configs/analysis/mapshift_2d_full_study_smoke_v0_1.json \
  --output-dir outputs/releases/mapshift_2d_v0_1_smoke \
  --print-summary
```

Expected runtime: a few minutes on a typical laptop or CPU VM. It is intended for artifact sanity checking, not for reproducing paper-scale confidence intervals.

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
- Full single-GPU study: overnight-scale to multi-day depending on GPU. The expanded 24-motif release should be budgeted at roughly 30-36 wall-clock hours on one NVIDIA L4, with faster completion expected on L40S/H100-class GPUs. Runtime is dominated by task generation, evaluation loops, bootstrap aggregation, and rendering. The learned training jobs are small.
- Full CPU-only study: possible but not recommended for deadline-sensitive reproduction.

Monitor progress:

```bash
tail -f outputs/releases/mapshift_2d_v0_1_full/logs/build_benchmark.log
```

Count evaluation chunks:

```bash
grep -c "evaluating family=" outputs/releases/mapshift_2d_v0_1_full/logs/build_benchmark.log
```

The full config currently runs one primary CEP sweep plus four protocol-comparison sweeps. Each sweep has 24 motifs x 4 intervention families = 96 logged family chunks, so a completed run should log roughly 480 `evaluating family=` lines.

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

## Deadline-Friendly Study Runs

The complete `mapshift_2d_full_study_v0_1` configuration runs the full baseline roster under five protocol variants. When wall-clock time is constrained, use two grid-aligned runs instead:

1. A CEP-only full-roster run for the main family-wise results table.
2. A deterministic full-protocol diagnostic for stale-map, weak-heuristic, and belief-update protocol sensitivity.

This preserves the 24 structural motifs, 10/6/8 split, all four intervention families, all severity levels, and the full task mix, while avoiding five repeated full-roster sweeps.

CEP-only full-roster run:

```bash
export MAPSHIFT_TORCH_DEVICE=cuda:0
export MAPSHIFT_CHECKPOINT_DIR=/tmp/mapshift_learned_baselines_cep_only_v0_1

python3 scripts/build_benchmark.py \
  --tier mapshift_2d \
  --study-config configs/analysis/mapshift_2d_main_cep_only_v0_1.json \
  --output-dir outputs/releases/mapshift_2d_v0_1_cep_only \
  --print-summary
```

Deterministic full-protocol diagnostic:

```bash
python3 scripts/run_mapshift_2d_study.py \
  configs/analysis/mapshift_2d_belief_update_diagnostic_full_protocols_v0_1.json \
  --output-dir outputs/studies/mapshift_2d_belief_update_diagnostic_full_protocols_v0_1 \
  --log-file outputs/studies/mapshift_2d_belief_update_diagnostic_full_protocols_v0_1/logs/run.log \
  --print-summary
```

Then generate held-out motif and paired-bootstrap delta tables:

```bash
python3 scripts/analyze_mechanism_diagnostic.py \
  outputs/studies/mapshift_2d_belief_update_diagnostic_full_protocols_v0_1/study_bundle.json \
  --output-dir outputs/studies/mapshift_2d_belief_update_diagnostic_full_protocols_v0_1/mechanism_diagnostic_analysis \
  --split test \
  --family topology \
  --family semantic \
  --resamples 1000 \
  --print-summary
```

The CEP-only run has about 96 logged family chunks instead of the full run's 480. The deterministic diagnostic has the same 480 family chunks but only four deterministic/reference methods and no learned baseline training, so it is much cheaper than the full-roster protocol sweep.

## Reproducing Paper Claims

The paper's empirical claims come from two executable study paths. The submitted `paper.pdf` contains the publication figures; the artifact also regenerates code-produced protocol and intervention-example SVGs for reviewer inspection.

| Paper item | Reproduction source |
|---|---|
| Figure 1, CEP protocol diagram | Included in `paper.pdf`; a data-free SVG version can be regenerated with `scripts/render_paper_outputs.py --protocol-diagram-only`. |
| Figure 2, matched intervention pairs | Included in `paper.pdf`; generated intervention examples are rendered by `scripts/render_paper_outputs.py` as `intervention_examples_2d.svg`. |
| Benchmark health summary | Full reproduction run, `outputs/releases/mapshift_2d_v0_1_full/study/tables/benchmark_health_summary.json`. |
| Full-run family-wise calibration table | Full reproduction run, `outputs/releases/mapshift_2d_v0_1_full/study/tables/familywise_main_results.json` and `paper_outputs/tables/main_familywise_results.md`. |
| Full-run protocol sensitivity | Full reproduction run, `outputs/releases/mapshift_2d_v0_1_full/study/tables/protocol_sensitivity_and_rank_correlation.json`. |
| Severity-response and family stress profiles | Full reproduction run, `outputs/releases/mapshift_2d_v0_1_full/study/tables/severity_response.json` and `paper_outputs/figures/severity_response_curves.svg`. |
| Deterministic mechanism diagnostic | Belief-update diagnostic run, `outputs/studies/mapshift_2d_belief_update_diagnostic_v0_1/study_bundle.json`. |
| Conclusion audit | Combination of the full reproduction run and the belief-update diagnostic run. |
| Baseline metadata and hyperparameters | `configs/calibration/*.json`, `outputs/releases/mapshift_2d_v0_1_full/study/raw/cep_report.json`, and `paper_outputs/tables/baseline_metadata.md`. |

Run the full reproduction path first:

```bash
python3 scripts/build_benchmark.py \
  --tier mapshift_2d \
  --study-config configs/analysis/mapshift_2d_full_study_v0_1.json \
  --output-dir outputs/releases/mapshift_2d_v0_1_full \
  --print-summary
```

Then run the deterministic mechanism diagnostic used for the stale-map versus belief-update claims:

```bash
python3 scripts/run_mapshift_2d_study.py \
  configs/analysis/mapshift_2d_belief_update_diagnostic_v0_1.json \
  --print-summary
```

Expected diagnostic outputs:

```text
outputs/studies/mapshift_2d_belief_update_diagnostic_v0_1/
  logs/run_mapshift_2d_study.log
  raw/cep_report.json
  raw/protocol_comparison_report.json
  raw/benchmark_health_report.json
  tables/familywise_main_results.json
  tables/protocol_sensitivity_and_rank_correlation.json
  tables/benchmark_health_summary.json
  figures/*.json
  manifests/study_manifest.json
  study_bundle.json
```

Inspect the full-run health gates cited in the paper:

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path("outputs/releases/mapshift_2d_v0_1_full/study/tables/benchmark_health_summary.json")
print(json.dumps(json.loads(p.read_text()), indent=2))
PY
```

Inspect the full-run protocol sensitivity table:

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path("outputs/releases/mapshift_2d_v0_1_full/study/tables/protocol_sensitivity_and_rank_correlation.json")
print(json.dumps(json.loads(p.read_text()), indent=2))
PY
```

Inspect the deterministic mechanism diagnostic scores and rank changes:

```bash
python3 - <<'PY'
import json
from pathlib import Path
b = json.loads(Path("outputs/studies/mapshift_2d_belief_update_diagnostic_v0_1/study_bundle.json").read_text())

print("=== Proposition support ===")
print(json.dumps(b["proposition_support"], indent=2))

print("\n=== Family-wise CEP scores ===")
rows = b["raw_reports"]["cep_report"]["familywise_summary"]["rows"]
for row in rows:
    print(
        row["baseline_name"],
        row["family"],
        round(row["family_primary_score"], 3),
        "episodes=", row["episode_count"],
    )

print("\n=== Protocol comparisons ===")
for name, comp in b["protocol_sensitivity"]["pairwise_comparisons"].items():
    print(name, "tau=", comp.get("kendall_tau"), "best_changes=", comp.get("best_method_changes"))
PY
```

After the 24-motif deterministic diagnostic finishes, generate the held-out motif consistency table and paired-bootstrap delta CIs used for the central stale-map versus belief-update claims:

```bash
python3 scripts/analyze_mechanism_diagnostic.py \
  outputs/studies/mapshift_2d_belief_update_diagnostic_v0_1_24motif/study_bundle.json \
  --output-dir outputs/studies/mapshift_2d_belief_update_diagnostic_v0_1_24motif/mechanism_diagnostic_analysis \
  --split test \
  --family topology \
  --family semantic \
  --resamples 1000 \
  --print-summary
```

This writes:

```text
mechanism_diagnostic_analysis/p2_p3_summary.json
mechanism_diagnostic_analysis/tables/heldout_motif_consistency.json
mechanism_diagnostic_analysis/tables/heldout_motif_consistency.md
mechanism_diagnostic_analysis/tables/heldout_motif_summary.json
mechanism_diagnostic_analysis/tables/heldout_motif_summary.md
mechanism_diagnostic_analysis/tables/paired_delta_bootstrap.json
mechanism_diagnostic_analysis/tables/paired_delta_bootstrap.md
```

The held-out consistency table reports, per test motif and family, `BeliefUpdate - StaleMap` under CEP and same-environment evaluation, plus protocol deltas for each method. The paired-bootstrap table reports 95% CIs for `BeliefUpdate_CEP - StaleMap_CEP`, `BeliefUpdate_same_env - StaleMap_same_env`, and the protocol-reversal contrast `(BeliefUpdate_CEP - StaleMap_CEP) - (BeliefUpdate_same_env - StaleMap_same_env)`.

## High-Capacity Learned World-Model Add-On

The main full-study artifact leaves the original baseline roster unchanged. To add the higher-capacity learned row without recomputing the older baselines, run a CEP-only calibration report for the 1.14M-parameter pretrained structured graph world model. The generated report contains all severities; the paper table reports the non-identity severity subset for this row.

On a CUDA host:

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

This command trains one global pretrained world model per seed across generated train motifs, evaluates only that learned baseline on the CEP grid, and leaves all previously reported baselines untouched. The evaluator also runs an implicit oracle reference internally to populate oracle fields, so the output includes an oracle row; use the `pretrained_structured_graph_world_model` rows for the append-only learned-baseline table entry.

Expected output:

```text
outputs/studies/pretrained_structured_graph_world_model_1m_v0_1/
  cep_report.json
  logs/run.log
```

Expected scale: 24 motifs x 4 families x 4 severities x 3 task classes x 3 task samples. The learned baseline contributes 3456 all-severity episode records, of which 2592 non-identity severity records are used for the paper row; the implicit oracle contributes 3456 reference records. This is substantially shorter than the full expanded reproduction because it skips the older baselines and protocol-comparison sweeps.

Extract the family-wise scores for the new row:

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
    print(
        row["baseline_name"],
        row["family"],
        "episodes=", row["episode_count"],
        "score=", round(row["family_primary_score"], 3),
    )
PY
```

Inspect trainable parameter count and device metadata:

```bash
python3 - <<'PY'
import json
from pathlib import Path

p = Path("outputs/studies/pretrained_structured_graph_world_model_1m_v0_1/cep_report.json")
metadata = json.loads(p.read_text())["baseline_metadata"]["pretrained_structured_graph_world_model"]
print("parameter_count_min:", metadata["parameter_count_min"])
print("parameter_count_max:", metadata["parameter_count_max"])
for run_name, run in sorted(metadata["runs"].items()):
    print(run_name, "seed=", run["seed"], "device=", run.get("torch_device_resolved"))
PY
```

The submitted `paper.pdf` is built from the author manuscript and is not required to execute the artifact. The commands above regenerate the data products used to fill the paper tables.

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

The main full study uses eight scientific baseline classes. Deterministic reference and classical baselines have no trainable parameters. The high-capacity pretrained structured graph world model is an append-only learned baseline that can be evaluated separately with the command above.

| Baseline | Config | Key hyperparameters |
|---|---|---|
| `oracle_post_intervention_planner` | `configs/calibration/oracle_post_intervention_planner_v0_1.json` | oracle access; no training |
| `same_environment_upper_baseline` | `configs/calibration/same_environment_upper_baseline_v0_1.json` | same-environment reference; no training |
| `weak_heuristic_baseline` | `configs/calibration/weak_heuristic_baseline_v0_1.json` | visited-state heuristic; no training |
| `classical_belief_update_planner` | `configs/calibration/classical_belief_update_planner_v0_1.json` | occupancy-map plus belief update; edge/cost/dynamics/token updates; no training |
| `monolithic_recurrent_world_model` | `configs/calibration/monolithic_recurrent_world_model_v0_1.json` | hidden size 12, observation stride 2, 6 epochs, lr 0.01, max rollout 8 |
| `persistent_memory_world_model` | `configs/calibration/persistent_memory_world_model_v0_1.json` | 16 memory slots, slot stride 3, readout width 8, 8 epochs, lr 0.01, max rollout 8 |
| `relational_graph_world_model` | `configs/calibration/relational_graph_world_model_v0_1.json` | hidden size 10, 2 message-passing steps, 10 epochs, lr 0.01 |
| `structured_dynamics_world_model` | `configs/calibration/structured_dynamics_world_model_v0_1.json` | geometry width 10, dynamics width 6, 10 epochs, lr 0.01 |
| `pretrained_structured_graph_world_model` | `configs/calibration/pretrained_structured_graph_world_model_1m_v0_1.json` | Approximately 1.14M trainable parameters; hidden size 256, 6 message-passing steps, pair width 256, dynamics width 128, 120 epochs, lr 0.0005, batch size 64, 4000 generated train environments, 800 generated validation environments; paper row reports 2592 non-identity evaluation episodes |

Except for `pretrained_structured_graph_world_model`, learned baselines train separately for each base environment and model seed from the reward-free exploration trace and are not pretrained across environments. The full study expands the original learned baselines over seeds `[0, 1, 2, 3, 4]` using `configs/analysis/mapshift_2d_full_study_v0_1.json`; the append-only pretrained baseline is evaluated with the high-capacity command above.

For a faster deterministic diagnostic that isolates stale maps, local heuristics, and explicit belief updates, run:

```bash
python3 scripts/run_mapshift_2d_study.py \
  configs/analysis/mapshift_2d_belief_update_diagnostic_v0_1.json \
  --print-summary
```

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
- 24 structural motifs
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
  README.md          Documentation map and reviewer-facing entry points
  internal/          Historical planning notes, superseded by this README and configs
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

The raw JSON records and protocol comparison reports are the largest generated artifacts. Generated release directories can be archived after preserving the files needed for reproduction. Learned checkpoints are in `MAPSHIFT_CHECKPOINT_DIR` or the config default `/tmp/mapshift_learned_baselines`.

Validation fails due to 3D prototype:

Use the primary tier path:

```bash
python3 scripts/validate_benchmark.py --tier mapshift_2d configs/benchmark/release_v0_1.json
```

## License

The MapShift source code is licensed under the [MIT License](LICENSE). Generated benchmark outputs, raw episode records, health reports, rendered result tables, and rendered result figures produced by the artifact commands are licensed under CC BY 4.0. Third-party dependencies and tooling are summarized in [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).

Citation metadata is provided in [`CITATION.cff`](CITATION.cff).
