# MapShift

**MapShift** is a benchmark for testing how well AI agents can adapt when their environment changes. It answers one question:

> After an agent explores an environment without any task reward, can it still plan correctly when the environment is changed in a structured way?

Think of it like this: imagine a robot that freely wanders around a building and builds a mental map. Then someone rearranges the furniture, blocks a hallway, or swaps the room labels. Can the robot still navigate to where it needs to go? MapShift measures exactly that.

This concept is called **Counterfactual Embodied Planning (CEP)**.

## How It Works — The Big Picture

MapShift works in five stages. Here's the pipeline:

```
1. GENERATE        2. INTERVENE        3. EXPLORE         4. EVALUATE         5. ANALYZE
   a 2D map  ──▶  change something  ──▶  agent walks  ──▶  give it tasks  ──▶  measure
   from seed       in the map            around freely     in changed map      performance
```

### Stage 1 — Generate an Environment

A procedural generator (`mapshift/envs/map2d/generator.py`) creates a 96×96 grid-based 2D map from a **seed number**. The same seed always produces the same map — this makes experiments reproducible.

Each map contains:
- An **occupancy grid** — cells that are free (walkable) or occupied (walls)
- **Nodes** — named anchor points (like room centers) placed on the grid
- **Adjacency graph** — which nodes connect to each other
- **Landmarks and goal tokens** — semantic labels attached to nodes
- **Dynamics parameters** — how movement works (friction, step size, etc.)

Maps are built from **motif templates** — structural patterns like `simple_loop`, `branching_chain`, or `deceptive_shortcut`. There are 8 motif families defined in the environment config (`configs/env2d/release_v0_1.json`), and they're split across train/val/test sets to prevent data leakage.

### Stage 2 — Apply an Intervention

This is the "shift" in MapShift. The benchmark modifies the base environment in one of **four intervention families**, each targeting a different aspect of the world model:

| Family | What Changes | What Stays the Same | Code |
|--------|-------------|---------------------|------|
| **Metric** | Geometry scale, step size, odometry bias | Topology, semantics | `mapshift/interventions/metric.py` |
| **Topology** | Graph connectivity (block doors, add walls) | Geometry, semantics | `mapshift/interventions/topology.py` |
| **Dynamics** | Friction, inertia, action asymmetry | Geometry, topology | `mapshift/interventions/dynamics.py` |
| **Semantic** | Landmark names, goal token meanings | Geometry, topology | `mapshift/interventions/semantic.py` |

Each family has **4 severity levels** (0–3), where 0 means no change and 3 is the most aggressive. For example, a topology intervention at severity 3 removes corridors and inserts walls, while severity 1 just blocks a single doorway.

Every intervention is applied via a common interface (`mapshift/interventions/base.py`):

```python
result = intervention.apply(base_env, severity=2, seed=42)
# Returns: intervened_env + InterventionManifest (provenance metadata)
```

### Stage 3 — Reward-Free Exploration

Before the test, an agent gets to explore the **base** (unmodified) environment with a budget of steps (default: 800). It receives no task or reward — it just wanders and builds its internal world model.

This is managed by the exploration runner (`mapshift/runners/explore.py`), which tracks:
- Which grid cells were visited
- Which nodes were reached
- Total steps taken

### Stage 4 — Evaluate on Tasks

Now the agent must perform tasks in the **intervened** environment using the world model it built during exploration. There are three task classes:

| Task Class | What the Agent Does | Example | Code |
|------------|-------------------|---------|------|
| **Planning** | Navigate from A to B | Find shortest path after a wall was added | `mapshift/tasks/planning.py` |
| **Inference** | Answer questions about changes | "Has the topology changed?" | `mapshift/tasks/inference.py` |
| **Adaptation** | Re-learn with a small interaction budget | Replan after dynamics changed, given 32 extra steps | `mapshift/tasks/adaptation.py` |

Tasks are sampled by `mapshift/tasks/samplers.py`, which checks that each task is **solvable but not trivial** before including it.

### Stage 5 — Measure and Analyze

Performance is measured with family-specific metrics (`mapshift/metrics/`):

- **Planning**: success rate, path efficiency, counterfactual planning accuracy
- **Inference**: accuracy, change detection AUROC
- **Adaptation**: sample efficiency, recovery vs. budget

Results are reported **per intervention family** (not pooled), with bootstrap confidence intervals and rank stability checks (`mapshift/analysis/`).

## Repository Structure

```
mapshift/                          # Main Python package
├── core/                          # Config schemas, manifests, logging, registry
├── envs/map2d/                    # 2D grid environment (generator, dynamics, renderer)
├── interventions/                 # Four intervention families + base interface
├── tasks/                         # Task definitions and sampling
├── baselines/                     # Baseline model interface (oracle, heuristic, recurrent, etc.)
├── runners/                       # Exploration & evaluation execution
├── metrics/                       # Performance measurement
├── analysis/                      # Study orchestration, statistics, bootstrap
└── splits/                        # Train/val/test splitting with leakage checks

configs/                           # JSON configuration files
├── benchmark/release_v0_1.json    # Top-level release config (references everything else)
├── env2d/                         # 2D environment specification
├── interventions/                 # Intervention family definitions & severity ladders
├── tasks/                         # Task class definitions & metrics
├── baselines/                     # Baseline roster & hyperparameters
├── calibration/                   # Per-baseline calibration configs
├── analysis/                      # Study and analysis parameters
└── schema/                        # JSON schemas for config validation

scripts/                           # Entry-point scripts
├── validate_benchmark.py          # Validate release config bundle
├── smoke_map2d.py                 # Quick smoke test: generate + intervene + render
├── build_benchmark.py             # Full split generation and artifact creation
├── generate_release_splits.py     # Build train/val/test splits
├── run_eval.py                    # Run full evaluation
└── run_mapshift_2d_study.py       # Orchestrate a complete study

tests/                             # Test suite
├── unit/                          # Config, grid, metrics, splits tests
├── smoke/                         # Quick end-to-end validation
├── integration/                   # Full pipeline tests
└── regression/                    # (scaffolded)

docs/                              # Documentation
├── IMPLEMENTATION_PLAN.md         # Implementation blueprint
├── benchmark_spec.md              # Full scientific specification
├── evaluation_card.md             # Responsible-use guidance
└── release_checklist.md           # Pre-release validation checklist

outputs/                           # Generated artifacts (manifests, reports, figures, tables)
```

## Key Design Decisions

- **Zero dependencies** — The entire package is pure Python with no external libraries. This maximizes portability and avoids version conflicts.
- **Deterministic from seed** — Every environment, intervention, and task is reproducible from a `(seed, config_hash)` pair. The seeding logic lives in `mapshift/core/seeding.py`.
- **Provenance tracking** — Every generated artifact (environment, intervention, task) produces an `ArtifactManifest` with a unique ID, timestamp, parent references, and config hash (`mapshift/core/manifests.py`).
- **Motif-based splits** — Train/val/test sets are split by structural motif (not by random environment instance), with explicit leakage checks (`mapshift/splits/leakage_checks.py`).
- **Family-wise reporting** — Metrics are always reported per intervention family to avoid hiding conclusions in pooled scores.

## Quickstart

**Requirements:** Python ≥ 3.10, no other dependencies needed.

Validate the draft release config bundle:

```bash
python3 scripts/validate_benchmark.py configs/benchmark/release_v0_1.json
```

Run a quick smoke test (generates a map, applies all 4 interventions, renders output):

```bash
python3 scripts/smoke_map2d.py
```

Run the full test suite:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

## Project Status

This is version **0.1-draft**. The current state:

- ✅ 2D environment generator with 8 motif templates
- ✅ All 4 intervention families with severity ladders
- ✅ Task sampler for planning, inference, and adaptation
- ✅ Config-driven pipeline from generation through evaluation
- ✅ Validation and smoke-test entry points
- 🔧 Baselines, runners, and analysis stack are partially scaffolded
- 🔧 3D tier (ProcTHOR) is scaffolded but not yet functional

## Project Documents

- [Implementation Plan](docs/IMPLEMENTATION_PLAN.md)
- [Benchmark Specification](docs/benchmark_spec.md) — Full scientific spec with formal CEP definition
- [Evaluation Card](docs/evaluation_card.md) — What MapShift measures, limitations, and responsible use
- [Release Checklist](docs/release_checklist.md)

## License

[MIT](LICENSE)
