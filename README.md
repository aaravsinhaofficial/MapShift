# MapShift: A Benchmark for Counterfactual Embodied Planning

MapShift is a diagnostic benchmark for evaluating whether world models learned during reward-free exploration remain useful after structured environmental interventions. It introduces the **Counterfactual Embodied Planning (CEP)** evaluation protocol: an agent first explores an environment without task reward, the environment is then modified along a controlled intervention axis, and the agent must plan, infer, or adapt in the changed environment using only its previously learned world model.

The frozen **MapShift-2D v0.1** release provides four orthogonal intervention families — **metric**, **topology**, **dynamics**, and **semantic** — each with calibrated severity ladders, enabling fine-grained diagnosis of which aspects of a world model are brittle or robust under distributional shift. The benchmark emphasizes family-wise reporting, deterministic reproducibility, and explicit provenance tracking for all generated artifacts.

## Overview

Existing embodied AI benchmarks typically evaluate agents in the same environment they trained in, or under unconstrained domain randomization. Neither setup isolates *which* aspect of a learned world model fails when the environment changes. MapShift fills this gap by formalizing the evaluation question:

> After an agent explores an environment without task reward, can it still plan correctly when the environment is changed in a structured way?

The benchmark is built around three premises:

1. **P1 — Missing capability:** Existing protocols do not directly measure post-intervention planning ability after reward-free exploration.
2. **P2 — Construct validity:** Metric, topology, dynamics, and semantic interventions probe distinct weaknesses rather than one undifferentiated notion of robustness.
3. **P3 — Scientific consequence:** Model rankings and scientific conclusions change when evaluation is done under CEP instead of standard same-environment performance.

## Benchmark Description

### The CEP Protocol

The benchmark pipeline has five stages:

```
1. GENERATE        2. INTERVENE        3. EXPLORE         4. EVALUATE         5. ANALYZE
   a 2D map  ──▶  change something  ──▶  agent walks  ──▶  give it tasks  ──▶  measure
   from seed       in the map            around freely     in changed map      performance
```

**Stage 1 — Environment Generation.**
A procedural generator (`mapshift/envs/map2d/generator.py`) creates a 96×96 occupancy grid from a deterministic seed. Each map contains navigable cells, named anchor nodes, an adjacency graph, landmarks, and dynamics parameters. Maps are instantiated from **motif templates** — structural patterns like `simple_loop`, `branching_chain`, or `deceptive_shortcut` — defined in the environment config (`configs/env2d/release_v0_1.json`).

**Stage 2 — Intervention.**
The base environment is modified along exactly one of four intervention families, each targeting a different aspect of the world model:

| Family | What Changes | What Is Preserved | Severity Range | Code |
|--------|-------------|-------------------|----------------|------|
| **Metric** | Geometry scale, step size, odometry bias | Topology, semantics | 1.0× → 1.5× | `mapshift/interventions/metric.py` |
| **Topology** | Graph connectivity (blocked doors, inserted walls) | Geometry, semantics | 0 → 3 edges | `mapshift/interventions/topology.py` |
| **Dynamics** | Friction, inertia, action asymmetry | Geometry, topology | 1.0 → 0.6 | `mapshift/interventions/dynamics.py` |
| **Semantic** | Landmark identities, goal token meanings | Geometry, topology | 0% → 50% remapped | `mapshift/interventions/semantic.py` |

All interventions share a common interface (`mapshift/interventions/base.py`) and produce provenance metadata via `InterventionManifest`.

**Stage 3 — Reward-Free Exploration.**
An agent explores the **base** (unmodified) environment for a fixed budget (default: 800 steps) with no task reward. The exploration runner (`mapshift/runners/explore.py`) records visited cells and reached nodes.

**Stage 4 — Post-Intervention Evaluation.**
The agent must perform tasks in the **intervened** environment using only the world model it built during exploration. Three task classes are supported:

| Task Class | What the Agent Does | Example | Code |
|------------|-------------------|---------|------|
| **Planning** | Navigate from start to goal | Find shortest path after a wall was inserted | `mapshift/tasks/planning.py` |
| **Inference** | Detect or predict environmental changes | "Has the topology changed?" | `mapshift/tasks/inference.py` |
| **Adaptation** | Re-learn with a small post-shift interaction budget | Replan after dynamics changed, given 32 extra steps | `mapshift/tasks/adaptation.py` |

Tasks are sampled by `mapshift/tasks/samplers.py`, which enforces solvability and rejects trivial instances.

**Stage 5 — Metrics and Analysis.**
Performance is measured with family-specific metrics (`mapshift/metrics/`):

- **Planning:** success rate, normalized path efficiency, counterfactual planning accuracy
- **Inference:** accuracy, change detection AUROC
- **Adaptation:** sample efficiency, recovery vs. budget

Results are reported **per intervention family** with bootstrap confidence intervals and rank stability checks (`mapshift/analysis/`). Protocol comparison tools (`mapshift/runners/compare_protocols.py`) enable direct comparison of CEP against same-environment and no-exploration baselines.

### Hypotheses

MapShift is designed to test four specific hypotheses:

- **H1:** Performance under topology shifts correlates less with standard same-environment reward than performance under metric shifts.
- **H2:** Memory-augmented systems outperform monolithic recurrent systems more on topology and semantic shifts than on mild metric shifts.
- **H3:** Structured-dynamics systems improve more on metric and dynamics shifts than on semantic shifts.
- **H4:** Evaluation protocol choice changes model rankings; family-specific reporting is more informative than a single pooled score.

### Splits and Leakage Control

Train, validation, and test sets are split by **structural motif**, not by random environment instance. This ensures that the test set contains structurally novel layouts, not just new random seeds of familiar patterns. Leakage checks (`mapshift/splits/leakage_checks.py`) verify separation using structural signatures computed over connectivity and geometry hashes.

### Tiers

- **MapShift-2D v0.1** (primary): Frozen procedural 2D grid benchmark release with full intervention, health, evaluation, and artifact-building support.
- **MapShift-3D** (prototype/future work): ProcTHOR-compatible files remain in the repository, but 3D is not part of the v0.1 scientific claim.

## Getting Started

### Requirements

- Python ≥ 3.10
- CPU dependencies from `pyproject.toml`: `numpy>=1.26,<3.0` and `torch>=2.2,<3.0`

### Installation

```bash
git clone https://github.com/aaravsinhaofficial/MapShift.git
cd MapShift
python3 -m pip install --upgrade pip
python3 -m pip install -e .
```

### Validation

Validate the frozen 2D release config bundle:

```bash
python3 scripts/validate_benchmark.py --tier mapshift_2d configs/benchmark/release_v0_1.json
```

### Reviewer Smoke

Build a small reviewer artifact, including health checks, split manifests, a learned-baseline study, paper tables, and SVG figures:

```bash
python3 scripts/build_benchmark.py --tier mapshift_2d --study-config configs/analysis/mapshift_2d_full_study_smoke_v0_1.json --output-dir outputs/releases/mapshift_2d_v0_1_smoke --print-summary
```

The smoke config is designed for CPU execution with small seeds. Runtime depends on the installed PyTorch build and host CPU.

### Full Reproduction

Regenerate the full v0.1 paper-facing tables and figures from frozen configs:

```bash
python3 scripts/build_benchmark.py --tier mapshift_2d --study-config configs/analysis/mapshift_2d_full_study_v0_1.json --output-dir outputs/releases/mapshift_2d_v0_1_full --print-summary
```

Learned baselines use `torch_device: "auto"` and will select CUDA when a CUDA-enabled PyTorch build reports `torch.cuda.is_available()`. To pin a GPU explicitly, set:

```bash
export MAPSHIFT_TORCH_DEVICE=cuda:0
```

To render paper tables/figures again from saved study JSON without rerunning model evaluation:

```bash
python3 scripts/render_paper_outputs.py outputs/releases/mapshift_2d_v0_1_full/study/study_bundle.json --output-dir outputs/releases/mapshift_2d_v0_1_full/paper_outputs --print-summary
```

### Running Tests

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

## Usage

### Generating Environments and Applying Interventions

The benchmark is config-driven. The top-level release config (`configs/benchmark/release_v0_1.json`) references all sub-configs for environments, interventions, tasks, baselines, and analysis. Scripts in `scripts/` provide entry points:

| Script | Purpose |
|--------|---------|
| `validate_benchmark.py` | Validate release config structure and integrity |
| `smoke_map2d.py` | Quick end-to-end smoke test |
| `build_benchmark.py` | Frozen 2D release artifact builder |
| `generate_release_splits.py` | Build train/val/test splits with leakage detection |
| `run_eval.py` | Execute full evaluation on baselines |
| `run_mapshift_2d_study.py` | Orchestrate a complete MapShift-2D study |
| `run_protocol_comparison.py` | Compare CEP vs. same-environment vs. no-exploration protocols |
| `render_paper_outputs.py` | Render Markdown tables and SVG figures from saved study JSON |

### Baselines

The benchmark defines a common baseline interface (`mapshift/baselines/api.py`) with implementations for:

- **Oracle planner** — has access to the true intervened environment
- **Same-environment upper/reference** — oracle-style same-environment reference
- **Heuristic baseline** — simple reactive policy
- **Recurrent world model** — monolithic learned dynamics
- **Memory world model** — explicit memory-augmented architecture
- **Relational graph world model** — graph-structured world model
- **Structured-dynamics world model** — factorized geometry/traversal/dynamics learned baseline

Baseline configs are in `configs/calibration/`. `prismx_reference_model` is explicitly omitted from the frozen v0.1 main study because it is not implemented.

## Repository Structure

```
mapshift/                          # Main Python package
├── core/                          #   Config schemas, manifests, logging, registry
├── envs/map2d/                    #   2D grid environment (generator, dynamics, renderer)
├── interventions/                 #   Four intervention families + base interface
├── tasks/                         #   Task definitions and sampling
├── baselines/                     #   Baseline model interface and implementations
├── runners/                       #   Exploration & evaluation execution
├── metrics/                       #   Performance measurement (planning, inference, adaptation)
├── analysis/                      #   Study orchestration, statistics, bootstrap, ranking
└── splits/                        #   Train/val/test splitting with motif tagging & leakage checks

configs/                           # JSON configuration files
├── benchmark/                     #   Top-level release config
├── env2d/                         #   2D environment specification
├── interventions/                 #   Intervention family definitions & severity ladders
├── tasks/                         #   Task class definitions & metrics
├── baselines/                     #   Baseline roster & hyperparameters
├── calibration/                   #   Per-baseline calibration configs
├── analysis/                      #   Study and analysis parameters
└── schema/                        #   JSON schemas for config validation

scripts/                           # Entry-point scripts
tests/                             # Unit, smoke, integration, and regression tests
docs/                              # Documentation (spec, evaluation card, implementation plan)
outputs/                           # Generated artifacts (manifests, reports, figures, tables)
```

## Design Principles

- **Deterministic from seed.** Every environment, intervention, and task is reproducible from a `(seed, config_hash)` pair. Seeding logic: `mapshift/core/seeding.py`.
- **Provenance tracking.** Every generated artifact produces an `ArtifactManifest` with a unique ID, timestamp, parent references, and config hash. See `mapshift/core/manifests.py`.
- **CPU-first reproducibility.** The release uses bounded `numpy` and `torch` dependencies and runs without GPU-specific code paths.
- **Family-wise reporting.** Metrics are always disaggregated by intervention family to avoid masking conclusions in pooled scores.
- **Motif-based splitting.** Structural novelty in test sets, not just random seed variation.

## Documentation

- [Benchmark Specification](docs/benchmark_spec.md) — Full scientific specification with formal CEP definition, intervention taxonomy, and freeze policy
- [Release Freeze](docs/release_freeze_v0_1.md) — Frozen MapShift-2D v0.1 values and canonical reviewer commands
- [Evaluation Card](docs/evaluation_card.md) — What MapShift measures and does not measure, supported claims, known limitations, and responsible-use guidance
- [Implementation Plan](docs/IMPLEMENTATION_PLAN.md) — Implementation blueprint and scope
- [Release Checklist](docs/release_checklist.md) — Pre-release validation checklist

## Project Status

Version **MapShift-2D v0.1**. Current state:

- ✅ 2D environment generator with motif templates
- ✅ All four intervention families with severity ladders
- ✅ Task sampler for planning, inference, and adaptation
- ✅ Config-driven pipeline from generation through evaluation
- ✅ Validation, benchmark health, split leakage checks, smoke tests, integration tests, and artifact builder
- ✅ Oracle, same-environment reference, weak heuristic, recurrent, memory, relational, and structured-dynamics 2D baselines
- ✅ Paper-facing tables/figures regenerate from saved JSON outputs
- 🔧 3D tier remains prototype/future work and is not part of the v0.1 claim

## Citation

If you use MapShift in your research, please cite:

```bibtex
@misc{mapshift2026,
  title   = {MapShift: A Benchmark for Counterfactual Embodied Planning},
  author  = {Sinha, Aarav},
  year    = {2026},
  url     = {https://github.com/aaravsinhaofficial/MapShift}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
