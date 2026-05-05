# MapShift Benchmark Specification

## 1. Document Status

- Benchmark: MapShift
- Version: `mapshift_2d_v0_1`
- Status: Frozen 2D benchmark release contract
- Scope: 2D-first scientific and protocol specification
- Companion documents:
  - [Implementation Plan](../IMPLEMENTATION_PLAN.md)
  - [Evaluation Card](../evaluation_card.md)
  - [Release Checklist](../release_checklist.md)

This document is the source of truth for what the frozen MapShift-2D v0.1 release is intended to measure, how it is implemented, and how results must be reported. The ProcTHOR-compatible 3D files are prototype/future-work artifacts and do not carry the v0.1 scientific claim.

### Normative language

The terms `MUST`, `SHOULD`, and `MAY` are normative.

- `MUST` means required for a valid MapShift result.
- `SHOULD` means strongly recommended unless there is a documented reason not to do it.
- `MAY` means optional.

## 2. Scientific Objective

MapShift is a benchmark and evaluation protocol for Counterfactual Embodied Planning (CEP).

The benchmark addresses the following scientific question:

> After an agent explores an environment without task reward, can it still plan correctly when the environment is changed in a structured way?

MapShift is designed to test whether world knowledge acquired during reward-free exploration remains useful after a structured intervention changes the environment.

## 3. Intended Claims and Non-Claims

### Supported claims

Valid MapShift results may support claims about:

- robustness of learned world knowledge under structured intervention
- planning utility of world models after environment change
- diagnostic separation of failure modes across metric, topology, dynamics, and semantic intervention families
- protocol sensitivity, including whether evaluation design changes model rankings and scientific conclusions

### Unsupported claims

MapShift does not, by itself, support claims about:

- general real-world robotics reliability
- language-conditioned planning ability
- broad commonsense reasoning
- unrestricted transfer beyond the MapShift task and environment design assumptions

## 4. Benchmark Theses

MapShift is only successful as a benchmark if experiments support all three theses below.

### P1. Missing capability

Existing evaluation protocols do not directly measure post-intervention planning ability after reward-free exploration.

### P2. Construct validity

Metric, topology, dynamics, and semantic interventions probe distinct weaknesses rather than one undifferentiated notion of robustness.

### P3. Scientific consequence

Model rankings and scientific conclusions change when evaluation is done under CEP instead of standard same-environment performance.

## 5. Formal Problem Definition

Let `E` denote a base environment distribution.

1. A base environment `e ~ E` is sampled from a generator.
2. The agent interacts with `e` during an exploration phase of length `T_exp`.
3. No downstream task reward is provided during exploration.
4. An intervention operator `I_f(sigma)` is applied, where:
   - `f` is the intervention family in `{metric, topology, dynamics, semantic}`
   - `sigma` is intervention severity
5. The modified environment is `e' = I_f(sigma)(e)`.
6. A post-intervention task `q` is sampled from a task family `Q(e, e')`.
7. The evaluated system `M` is scored on performance after exploration in `e` and execution or inference in `e'`.

The central benchmark target is:

`CEP(M) = E_{e,f,sigma,q}[Perf(M; e -> e', q)]`

MapShift MUST not treat a single scalar aggregate as the only primary endpoint. The scientific core of the benchmark is family-wise performance.

## 6. Benchmark Entities and Artifacts

MapShift results MUST be recoverable from explicit artifacts.

### Core benchmark entities

- **Environment specification:** generator settings, motif labels, and substrate metadata
- **Base environment:** a concrete sampled instance `e`
- **Intervention specification:** family, severity, measurable intervention parameter, and preservation constraints
- **Intervened environment:** a concrete modified instance `e'`
- **Task specification:** task class, start state distribution, target definition, horizon, and success conditions
- **Run specification:** model, seed, protocol, compute metadata, and config hashes
- **Metrics artifact:** per-episode metrics and aggregated metrics

### Provenance requirements

Every benchmark artifact MUST store:

- artifact ID
- parent artifact IDs where applicable
- benchmark version
- code version or git commit
- config hash
- random seed or seeds
- creation time

## 7. Benchmark Structure

MapShift has a frozen 2D tier and a prototype 3D tier.

### Tier 1. MapShift-2D

This is the frozen v0.1 benchmark and MUST carry the main scientific weight.

Properties:

- partially observed
- procedurally generated
- cheap enough for many seeds
- able to produce matched intervention pairs
- able to control geometry, topology, dynamics, and semantics explicitly

### Tier 2. MapShift-3D Prototype

This is prototype/future work and is not part of the frozen v0.1 claim.

Properties:

- intended to preserve the same logical evaluation protocol as MapShift-2D
- intended to use the same intervention-family vocabulary
- intended to use the same reporting hierarchy
- substrate-specific restrictions must be documented before any future 3D claim

## 8. MapShift-2D Substrate Specification

### 8.1 Reference state representation

The reference 2D environment SHOULD use:

- a 2D navigable occupancy map
- continuous agent pose `(x, y, theta)` or equivalent semi-continuous state
- transition parameters that can be manipulated for dynamics shifts
- a semantic layer for landmarks, cues, or symbol identities

### 8.2 Reference action interface

The reference 2D implementation SHOULD use semi-continuous navigation:

- actions are discrete primitives
- state transitions are continuous or semi-continuous
- action-response parameters are exposed so metric and dynamics interventions can be applied cleanly

### 8.3 Observation model

The environment MUST be partially observed.

The reference observation model SHOULD include:

- egocentric local geometry
- bounded field of view or observation radius
- optional semantic landmark or cue observations

The observation model MUST not expose a global map during exploration or evaluation unless a run is explicitly marked as an oracle baseline.

### 8.4 Generator requirements

The 2D generator MUST support:

- exact reproducibility from config and seed
- motif tagging
- path planning on the base and intervened environment
- visibility computation
- geometry serialization and reloading

The motif grammar SHOULD include:

- loops
- bottlenecks
- room chains
- connector structures
- shortcuts
- deceptive or nested routing structures

## 9. MapShift-3D Substrate Specification

MapShift-3D SHOULD use ProcTHOR as the reference substrate.

The 3D implementation MUST preserve the same logical benchmark structure:

- reward-free exploration in base environment `e`
- structured intervention producing `e'`
- post-intervention task sampling from `Q(e, e')`
- family-wise evaluation and reporting

If a 3D intervention family cannot exactly preserve every corresponding 2D invariant, the deviation MUST be documented explicitly in both the benchmark card and experiment protocol.

## 10. Exploration Protocol

### 10.1 Exploration phase

During exploration:

- the agent MUST receive no downstream task reward
- the exploration budget MUST be fixed by the release config
- the observation interface MUST match the benchmark substrate
- all agent actions, observations, and internal checkpoints SHOULD be logged for auditability

### 10.2 Exploration budget

MapShift MUST define one canonical `T_exp` for the main benchmark table.

MapShift MAY additionally define sensitivity sweeps over exploration budget, but these are supplementary and MUST not replace the canonical budget in the main benchmark.

### 10.3 Exploration resets and side information

The release config MUST state:

- whether exploration episodes allow resets
- whether start states are fixed or sampled
- what, if any, privileged state is available
- what memory persistence is allowed across exploration and evaluation

## 11. Intervention Families

Each intervention family MUST change one major causal factor while preserving others as much as possible.

Every family MUST expose four severity levels:

- `0`: no intervention
- `1`: mild
- `2`: moderate
- `3`: strong

Severity MUST be monotone in a measurable intervention parameter.

### 11.1 Metric shifts

Purpose:

- test internal geometric stability
- test motion calibration

Allowed reference interventions:

- isotropic or controlled spatial scale change
- action gain perturbation
- odometry bias
- observation-range distortion

Primary expected failure mode:

- degradation in systems that memorize local transitions or appearance without stable geometry

### 11.2 Topology shifts

Purpose:

- test map revision
- test route recomputation under changed connectivity

Allowed reference interventions:

- doorway blocked
- wall inserted
- shortcut opened
- corridor removed

Primary expected failure mode:

- degradation in systems that do not update connectivity structure

### 11.3 Dynamics shifts

Purpose:

- test transition-model robustness independent of map quality

Allowed reference interventions:

- friction change
- inertial lag
- action asymmetry
- moving-obstacle rule change where supported

Primary expected failure mode:

- degradation in systems with weak transition structure even when geometry is intact

### 11.4 Semantic shifts

Purpose:

- test separation between world structure and cue meaning

Allowed reference interventions:

- landmark identity swap
- goal cue remapping
- reward-semantic reassignment
- appearance change with preserved geometry

Primary expected failure mode:

- degradation in systems that over-depend on surface cues

## 12. Intervention Invariants and Validator Requirements

The implementation MUST provide validator tests for intervention isolation.

### Required invariants

- geometry-only shifts do not alter topology labels unless explicitly intended
- topology-only shifts do not alter landmark identities
- semantic-only shifts do not alter navigable geometry
- dynamics-only shifts do not alter geometry
- severity parameters are monotone
- severity `0` reproduces the base environment exactly

### Required validator outputs

For every intervention family, validation reports SHOULD include:

- intervention parameter value by severity
- preserved attributes
- changed attributes
- eligibility failures
- counts of rejected or repaired intervention attempts

## 13. Task Suite

Tasks are sampled after exploration and after intervention.

MapShift defines three task classes.

### 13.1 Planning tasks

Reference tasks:

- shortest-path to target
- reroute after blockage
- reach target with changed dynamics
- navigate under changed cue semantics

Primary metrics:

- success rate
- normalized path efficiency

### 13.2 Inference tasks

Reference tasks:

- detect whether a topology change occurred
- predict masked regions after intervention
- answer a counterfactual reachability query

Primary metrics:

- accuracy
- AUROC where class imbalance makes it more appropriate

### 13.3 Adaptation tasks

Reference tasks:

- limited post-shift interaction budget followed by re-evaluation
- few-shot replanning after action-gain or dynamics change

Primary metrics:

- performance as a function of adaptation interaction budget
- adaptation sample efficiency

## 14. Task Sampling Rules

Task generation MUST control confounders wherever possible.

The task sampler SHOULD hold fixed across matched comparisons:

- start state distribution
- goal distribution
- visibility distribution
- episode horizon

Task generation MUST include eligibility checks that reject:

- impossible tasks unless explicitly marked as impossibility-evaluation tasks
- trivial tasks with near-zero planning demand
- tasks whose difficulty changes for reasons unrelated to the intended intervention family

## 15. Split Policy and Leakage Control

MapShift MUST split by generator seed and structural motif, not only by final environment instances.

### Required split principles

- train, validation, and test MUST differ in motif composition
- semantic remapping templates MUST also be split
- matched intervention descendants MUST inherit the split of the base environment
- no benchmark report may tune on the test split

### Required leakage checks

The benchmark implementation MUST check for leakage across:

- motif family
- geometry signature
- topology signature
- semantic template signature
- task template signature

### Benchmark health reporting

Every canonical split SHOULD ship with a summary of:

- motif counts
- intervention-family coverage
- severity coverage
- task-class coverage
- path-length and horizon distributions

## 16. Metrics and Reporting Hierarchy

### 16.1 Primary benchmark metrics

Primary metrics MUST be reported per intervention family:

- success rate
- normalized path efficiency
- adaptation sample efficiency
- counterfactual planning accuracy

### 16.2 Secondary diagnostics

Secondary diagnostics SHOULD include:

- change-detection AUROC
- masked-state inference accuracy
- long-horizon rollout consistency

### 16.3 Benchmark-health diagnostics

Benchmark-health diagnostics SHOULD include:

- ranking spread across model classes
- bootstrap rank stability
- metric correlation matrix
- severity-response monotonicity

### 16.4 Aggregation policy

The reporting policy is:

- family-wise metrics are primary
- pooled scores are supplementary
- rankings MUST be reported by family
- any global aggregate MUST state its construction explicitly

## 17. Hypotheses

MapShift adopts the following primary hypotheses.

### H1

Performance under topology shifts is less correlated with standard same-environment reward than performance under metric shifts.

Observable evidence:

- family-wise correlation analysis
- protocol-comparison rank correlation

### H2

Memory-heavy systems outperform monolithic recurrent systems more on topology and semantic shifts than on mild metric shifts.

Observable evidence:

- baseline family interaction effects by intervention family
- family-specific effect sizes

### H3

Structured-dynamics systems improve more on metric and dynamics shifts than on semantic shifts.

Observable evidence:

- baseline-by-family performance profile
- severity curves for dynamics-aware models

### H4

Evaluation protocol choices change model rankings, and family-specific reporting is more informative than a single pooled score.

Observable evidence:

- Kendall tau between protocols
- explicit rank reversals
- leave-one-family-out ranking analysis

## 18. Baseline Policy

MapShift SHOULD evaluate a small set of conceptually distinct system classes:

- monolithic recurrent world model
- persistent-memory world model
- structured-dynamics world model
- relational or graph world model
- structured reference models only if fully implemented; `prismx_reference_model` is omitted from v0.1

### Fairness requirements

The benchmark protocol MUST define:

- matched exploration budget across methods
- shared validation protocol
- fixed hyperparameters before final test runs
- reported compute and parameter counts

The benchmark SHOULD also include:

- an oracle planner
- a same-environment upper baseline
- a weak heuristic baseline

## 19. Statistical Analysis Policy

Final benchmark reporting MUST include:

- mean performance across environment seeds and model seeds
- `95%` bootstrap confidence intervals
- paired comparisons where base environments are shared
- bootstrap units that include both environment identity and model seed
- rank stability under bootstrap
- severity-response analysis

Significance testing MAY be reported, but it is supplementary. The main story MUST not rely solely on null-hypothesis testing.

## 20. Protocol Comparison Requirements

A valid MapShift benchmark study MUST compare at least the following protocol choices:

- same-environment evaluation vs post-intervention evaluation
- no-exploration evaluation vs reward-free exploration
- pooled-score reporting vs family-wise reporting
- short-horizon vs long-horizon evaluation

The implementation MUST support automatic computation of:

- rank correlations between protocols
- rank reversals
- per-family conclusion stability

## 21. Freeze Policy

This specification freezes logical benchmark decisions. Exact numeric values for the release benchmark MUST be frozen in canonical config files and referenced by release name.

### Freeze A. Scientific specification

Freeze before baseline training:

- supported claims
- non-claims
- formal CEP definition
- intervention families
- task classes
- reporting hierarchy
- main hypotheses

### Freeze B. Protocol specification

Freeze before final validation sweeps:

- canonical exploration budget
- test-horizon distribution
- severity parameterization
- task sampling policy
- generator grammar
- split rules
- primary metrics

### Freeze C. Final evaluation

Freeze before any final test runs:

- baseline roster
- hyperparameter policy
- seed counts
- bootstrap procedure
- plotting templates

Any post-freeze change MUST be logged with:

- changed item
- prior value
- new value
- reason
- expected impact

## 22. Versioning and Valid Results

Each public result MUST state:

- benchmark version
- substrate tier
- split version
- intervention families included
- task classes included
- canonical config names
- baseline version or commit

A result MUST NOT be called a main MapShift result if it omits family-wise reporting.

## 23. Frozen v0.1 Values

The frozen values are recorded in `docs/release_freeze_v0_1.md` and the canonical configs under `configs/`. They include `T_exp=800`, 96x96 maps, motif splits, severity ladders, task mix, horizons, adaptation budgets, model-seed counts, metric definitions, bootstrap settings, and the baseline roster. Any post-freeze change must be logged before it is used in a reported result.
