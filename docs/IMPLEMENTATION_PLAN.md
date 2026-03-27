# MapShift Implementation Plan

## 1. Purpose

This document is the implementation blueprint for MapShift: a benchmark and evaluation protocol for Counterfactual Embodied Planning (CEP).

The implementation must support the paper's core claim:

> After an agent explores an environment without task reward, can it still plan correctly when the environment is changed in a structured way?

The project is only successful if the implemented benchmark can support all three benchmark propositions:

1. **P1. Missing capability:** existing evaluation protocols do not directly measure post-intervention planning ability.
2. **P2. Construct validity:** metric, topology, dynamics, and semantic interventions probe distinct weaknesses.
3. **P3. Scientific consequence:** model rankings and conclusions change when evaluation is done with CEP rather than standard same-environment performance.

## 2. Scope and Non-Goals

### In scope

- A fully specified 2D benchmark with procedural generation, matched intervention pairs, task generation, metrics, and analysis tools.
- A 3D benchmark tier built on ProcTHOR with the same evaluation logic.
- Intervention families: metric, topology, dynamics, semantic.
- Task classes: planning, inference, adaptation.
- Family-wise reporting, protocol-comparison tools, and release-ready benchmark artifacts.
- Baseline interfaces and evaluation harnesses for the planned system families.

### Out of scope

- Claims about general real-world robotic reliability.
- Claims about language-conditioned planning.
- Claims about broad commonsense reasoning.
- Model invention as the main paper contribution.

## 3. Implementation Principles

1. **Benchmark-first, model-second.** Freeze the benchmark specification before large-scale baseline tuning.
2. **Every result must be reproducible from manifests.** Environment generation, intervention generation, task generation, and evaluation runs must all be driven by explicit configuration and logged provenance.
3. **Interventions must be controlled.** Each intervention family changes one major causal factor while preserving others as much as possible.
4. **Family-wise metrics are primary.** A pooled score may exist, but all code, analysis, and reporting must treat family-wise evaluation as the scientific default.
5. **Statistical validity is part of the implementation.** Bootstrap CIs, paired comparisons, ranking stability, and severity monotonicity are not afterthoughts.
6. **Releaseability is a hard requirement.** The codebase must be executable, documented, and submission-ready as a reusable evaluation artifact.

## 4. Benchmark Contract and Freeze Policy

MapShift must define a benchmark contract before final experiments. The contract has three freeze points.

### Freeze A: Scientific specification

Lock the following before baseline training:

- benchmark claims and non-claims
- formal CEP definition
- intervention families
- task classes
- reporting hierarchy
- split policy principles
- core hypotheses H1-H4

### Freeze B: Benchmark protocol

Lock the following after pilot debugging and before final validation sweeps:

- exploration budget `T_exp`
- test horizon distribution
- severity ladder definitions
- task sampling policy
- environment generation grammar
- train/val/test motif split rules
- primary and secondary metrics
- episode termination rules

### Freeze C: Final evaluation protocol

Lock the following before any final test runs:

- baseline roster
- shared training budget policy
- hyperparameter selection policy
- seed counts
- bootstrap procedure
- plotting and table templates

Any change after the relevant freeze point must be recorded in a changelog with rationale and impact assessment.

## 5. Target Repository Structure

The project should be implemented with a clean separation between benchmark specification, environment generation, evaluation execution, and paper analysis.

```text
MapShift/
  README.md
  docs/
    IMPLEMENTATION_PLAN.md
    benchmark_spec.md
    evaluation_card.md
    release_checklist.md
  configs/
    benchmark/
    env2d/
    env3d/
    interventions/
    tasks/
    baselines/
    analysis/
  mapshift/
    core/
      registry.py
      manifests.py
      schemas.py
      seeding.py
      logging.py
    envs/
      map2d/
        generator.py
        state.py
        dynamics.py
        observation.py
        renderer.py
        validation.py
      procthor/
        generator.py
        wrappers.py
        observation.py
        validation.py
    interventions/
      base.py
      metric.py
      topology.py
      dynamics.py
      semantic.py
      validators.py
    tasks/
      planning.py
      inference.py
      adaptation.py
      samplers.py
    metrics/
      planning_metrics.py
      inference_metrics.py
      adaptation_metrics.py
      ranking.py
      statistics.py
    splits/
      motifs.py
      builders.py
      leakage_checks.py
    baselines/
      api.py
      recurrent.py
      memory.py
      structured_dynamics.py
      relational.py
      prismx.py
    runners/
      explore.py
      evaluate.py
      compare_protocols.py
      batch.py
    analysis/
      bootstrap.py
      construct_validity.py
      rank_stability.py
      severity.py
      failure_taxonomy.py
      figures.py
  scripts/
    build_benchmark.py
    validate_benchmark.py
    run_eval.py
    run_protocol_comparison.py
    export_release.py
  tests/
    unit/
    integration/
    regression/
    smoke/
  outputs/
    manifests/
    reports/
    figures/
    tables/
```

## 6. Artifact Model

Every stage of the pipeline must produce explicit artifacts.

### Core artifacts

- **Environment spec:** defines generator parameters and motif identifiers.
- **Base environment manifest:** defines a sampled environment instance `e`.
- **Intervention manifest:** defines `I_f(sigma)` applied to a base environment.
- **Task manifest:** defines task family, start, goal, horizon, and evaluation metadata.
- **Run manifest:** defines model, seed, protocol, and config hashes.
- **Metrics artifact:** per-episode and aggregated metrics with bootstrap-ready tables.
- **Analysis artifact:** ranking outputs, correlation matrices, failure labels, plots, and paper tables.

### Provenance requirements

Every artifact must store:

- unique ID
- parent artifact IDs
- code version or git commit
- config hash
- random seed(s)
- creation timestamp
- benchmark version

## 7. Phase-by-Phase Implementation Plan

### Phase 0. Design Freeze and Benchmark Specification

#### Goal

Write the benchmark specification that prevents implementation drift.

#### Work

1. Create `docs/benchmark_spec.md` as the benchmark source of truth.
2. State intended claims, non-claims, assumptions, and limitations.
3. Formalize CEP notation and the evaluation pipeline `e -> e' -> q`.
4. Define intervention families and severity ladder semantics.
5. Define task classes and primary metrics.
6. Define split policy at the level of generator motifs, not just environment instances.
7. Define reporting hierarchy with family-wise metrics as primary endpoints.
8. Define the hypothesis table for H1-H4, including measurable observables for each hypothesis.

#### Deliverables

- benchmark specification document
- benchmark versioning policy
- hypothesis matrix
- reporting template

#### Exit criteria

- A new contributor can read the spec and know exactly what counts as a valid MapShift result.
- The benchmark claims and limitations are explicit enough to place directly in the paper.

### Phase 1. Core Software Foundation

#### Goal

Build the project skeleton, configuration system, manifest system, and validation utilities that everything else will rely on.

#### Work

1. Create the package layout under `mapshift/`, `configs/`, `scripts/`, and `tests/`.
2. Implement typed schemas for environment specs, interventions, tasks, and runs.
3. Implement deterministic seeding utilities.
4. Implement manifest serialization and config hashing.
5. Implement a registry for environments, intervention families, task samplers, and metrics.
6. Implement a logging format for benchmark generation and evaluation runs.
7. Create smoke tests that instantiate a benchmark config end to end without model training.

#### Deliverables

- core schemas and manifest classes
- config loading and validation
- run logging and provenance tracking
- CI-friendly smoke tests

#### Exit criteria

- Benchmark components can be instantiated entirely from config files.
- Two runs with the same configs and seeds reproduce identical artifacts.

### Phase 2. 2D MapShift Environment Generator

#### Goal

Implement the core 2D benchmark environment as the main experimental substrate.

#### Work

1. Define the 2D state representation:
   - occupancy grid or semi-continuous navigable map
   - agent pose
   - transition parameters
   - semantic landmarks and cue assignments
2. Implement procedural motif generation:
   - loops
   - bottlenecks
   - connectors
   - shortcuts
   - room-chain patterns
   - disconnected or nearly disconnected subregions where appropriate
3. Implement agent dynamics for continuous or semi-continuous navigation.
4. Implement partial observability:
   - egocentric local observation
   - visibility clipping
   - optional landmark channel
5. Implement exact environment serialization so a base environment can be replayed and transformed deterministically.
6. Implement map rendering and debugging views.
7. Implement generator diagnostics:
   - connectivity statistics
   - path-length distributions
   - visibility distributions
   - collision and dead-end checks

#### Deliverables

- procedural 2D generator
- observation model
- environment serializer
- environment visualization tools
- generator diagnostics report

#### Exit criteria

- The 2D environment can generate diverse but controlled instances at scale.
- Every base environment can be reloaded exactly from its manifest.
- The generator produces motif families that are separable enough to support train/val/test splitting.

### Phase 3. Intervention Family Implementation

#### Goal

Implement the four intervention families so that each changes one intended factor while preserving others as much as possible.

#### Work

1. Implement a shared intervention API:
   - input: base environment manifest
   - output: transformed environment manifest plus intervention metadata
2. Implement metric shifts:
   - arena scaling
   - action gain perturbation
   - odometry bias
   - camera or observation-range analog for the chosen observation model
3. Implement topology shifts:
   - insert wall
   - block doorway
   - open shortcut
   - remove corridor
4. Implement dynamics shifts:
   - friction change
   - inertial lag
   - action asymmetry
   - moving obstacle rule changes if supported cleanly
5. Implement semantic shifts:
   - landmark identity swaps
   - cue remapping
   - goal-semantic reassignment
   - appearance-only changes with preserved geometry
6. Implement severity ladders with monotone measurable parameters.
7. Implement intervention validators for each family.

#### Mandatory validator suite

- geometry-only shifts do not alter topology labels unless intended
- topology-only shifts do not alter landmark identities
- semantic-only shifts do not alter navigable geometry
- dynamics-only shifts do not alter geometry and only alter reachable sets when intended
- severity values induce monotone change in the intervention parameter
- no-op severity level reproduces the base environment exactly

#### Deliverables

- four intervention modules
- severity ladder registry
- intervention validation suite
- matched-pair generation tool

#### Exit criteria

- Interventions can be applied automatically to any eligible base environment.
- Each family passes validator tests on a large random sample.
- Intervention manifests fully describe what changed.

### Phase 4. Task Suite Implementation

#### Goal

Implement the planning, inference, and adaptation task families that run after exploration and intervention.

#### Work

1. Implement reward-free exploration episodes and exploration logging.
2. Implement planning tasks:
   - shortest-path to target
   - reroute after blockage
   - reach target under changed dynamics
   - reach target under changed cue semantics
3. Implement inference tasks:
   - detect whether a topology change occurred
   - predict masked regions after intervention
   - answer counterfactual reachability queries
4. Implement adaptation tasks:
   - limited post-shift interaction budget
   - few-shot replanning under changed dynamics
5. Implement task samplers that condition on `e`, `e'`, and intervention family.
6. Ensure task generation holds fixed, wherever possible:
   - start state distribution
   - goal distribution
   - visibility distribution
   - episode horizon
7. Implement per-task eligibility checks so that impossible or trivial tasks are filtered before evaluation.

#### Deliverables

- exploration runner
- planning tasks
- inference tasks
- adaptation tasks
- task sampler and task validator

#### Exit criteria

- Each task class can be sampled reproducibly from manifests.
- Task difficulty distributions are measurable and stable.
- The post-intervention task set is not dominated by degenerate cases.

### Phase 5. Split Design and Leakage Control

#### Goal

Build the split machinery so the benchmark is resistant to structural overfitting.

#### Work

1. Define structural motif tags for each base environment.
2. Implement split builders for train, validation, and test by motif family and seed.
3. Implement semantic-template splits so remapping templates do not leak across splits.
4. Implement leakage checks across:
   - motif family
   - geometry signature
   - topology signature
   - semantic template signature
   - task template signature
5. Build canonical benchmark manifests for the fixed release split.
6. Generate benchmark health summaries for each split:
   - motif counts
   - path-length distributions
   - intervention coverage
   - severity coverage
   - task-class coverage

#### Deliverables

- split builder
- leakage checker
- canonical benchmark manifests
- split health report

#### Exit criteria

- The split policy can be explained precisely in the paper.
- Leakage checks pass on the release benchmark.
- Each split has adequate family and severity coverage.

### Phase 6. Metrics, Reporting, and Statistical Analysis Tooling

#### Goal

Implement the evaluation metrics and analysis tooling that make MapShift an evaluation-science benchmark rather than a scoreboard.

#### Work

1. Implement primary metrics reported per intervention family:
   - success rate
   - normalized path efficiency
   - adaptation sample efficiency
   - counterfactual planning accuracy
2. Implement secondary diagnostics:
   - change-detection AUROC
   - masked-state inference accuracy
   - long-horizon rollout consistency
3. Implement benchmark-health metrics:
   - ranking spread
   - rank stability under bootstrap
   - metric correlation matrix
   - severity-response monotonicity
4. Implement bootstrap CI tooling over environment seeds and model seeds.
5. Implement paired comparison utilities using matched environments where appropriate.
6. Implement protocol-comparison metrics:
   - Kendall tau between protocols
   - leave-one-family-out ranking analysis
   - rank-reversal detection
7. Implement standard export tables and figure-ready data frames.

#### Deliverables

- metric library
- bootstrap and rank analysis utilities
- report generator for tables and plots
- fixed reporting templates

#### Exit criteria

- Any benchmark run can be converted automatically into paper-ready family-wise tables.
- Statistical summaries are reproducible from raw run manifests and metrics artifacts.

### Phase 7. Baseline Interfaces and Evaluation Wrappers

#### Goal

Build a common API and wrappers for the planned baseline families so that comparisons are fair and mechanically consistent.

#### Work

1. Define a baseline API covering:
   - reward-free exploration
   - representation or memory export
   - planning or control under the post-intervention task
   - limited adaptation episodes where applicable
2. Implement wrappers for the five planned system classes:
   - monolithic recurrent world model
   - persistent-memory world model
   - structured-dynamics world model
   - relational or graph world model
   - reference structured model such as PRISM-X
3. Standardize exploration budget handling across all wrappers.
4. Standardize logging of compute, parameters, and run failures.
5. Implement hyperparameter locking via config snapshots.
6. Implement sanity baselines:
   - oracle planner with full post-intervention map access
   - same-environment upper baseline
   - weak heuristic baseline

#### Deliverables

- baseline API
- baseline wrappers
- model run manifests
- shared evaluation scripts

#### Exit criteria

- Every baseline can be executed by the same evaluation harness.
- Fairness constraints are explicit and enforceable by config validation.
- At least one oracle and one weak baseline are available for calibration.

### Phase 8. Main 2D Experimental Program

#### Goal

Run the complete experimental program for the core 2D benchmark and generate all primary paper evidence.

#### Block 1. Construct validity

Implement analyses for:

- family-wise degradation profiles
- cross-family metric correlations
- severity monotonicity
- matched-pair ablations
- classifier-based separability of failure profiles

Success condition:

- intervention families induce measurably distinct stress patterns across model classes

#### Block 2. Discriminative power

Implement analyses for:

- model rankings by family
- effect sizes between model classes
- benchmark saturation checks
- bootstrap rank distributions

Success condition:

- the benchmark separates model classes without collapsing into a single dominant winner

#### Block 3. Protocol sensitivity

Implement protocol comparisons for:

- same-environment evaluation vs CEP evaluation
- no-exploration evaluation vs reward-free exploration
- single scalar reporting vs family-wise reporting
- short-horizon vs long-horizon evaluation

Success condition:

- ranking and scientific conclusions change under protocol choice

#### Block 4. Failure taxonomy

Implement automatic or semi-automatic failure labeling for:

- local-geometry failure
- connectivity-update failure
- dynamics-calibration failure
- semantic-remapping failure
- relocalization failure

Success condition:

- failure categories are measurable and informative enough to explain model differences

#### Deliverables

- full 2D benchmark result tables
- protocol-comparison study
- failure taxonomy outputs
- main paper figures

#### Exit criteria

- P1, P2, and P3 all have direct empirical support in the 2D tier.
- The 2D results are independently strong enough to anchor the paper even without 3D.

### Phase 9. 3D MapShift on ProcTHOR

#### Goal

Implement the 3D external-validity tier using ProcTHOR while preserving the same scientific structure as the 2D benchmark.

#### Work

1. Implement ProcTHOR scene sampling and serialization wrappers.
2. Implement the 3D observation interface consistent with the benchmark API.
3. Map the four intervention families into the 3D substrate:
   - metric: scale-like or calibration-like perturbations where supported
   - topology: blocked doors, opened paths, inserted blockers
   - dynamics: control-response changes and motion parameter changes
   - semantic: cue identity swaps, appearance changes with preserved layout
4. Implement matched scene-pair generation and validation.
5. Implement the same task classes in the 3D tier, with any substrate-specific restrictions documented.
6. Reuse the same reporting hierarchy and protocol-comparison tooling.

#### Deliverables

- ProcTHOR wrapper layer
- 3D intervention suite
- 3D task suite
- 3D evaluation manifests

#### Exit criteria

- The 3D tier is not a separate benchmark; it is the same benchmark logic in a second substrate.
- At least one full evaluation path exists for all four intervention families.
- The 3D tier provides external-validity evidence that the 2D findings are not an artifact of the simplified substrate.

### Phase 10. Mechanistic Analysis and Reference Structured Model

#### Goal

Use a structured or interpretable model to show that MapShift supports mechanism-level diagnosis rather than only system ranking.

#### Work

1. Implement latent probes aligned to intervention family labels.
2. Implement module-level lesions for:
   - memory
   - geometry or metric module
   - dynamics module
   - semantic or context module
3. Implement optional latent editing where technically feasible.
4. Integrate probe and lesion outputs into the failure taxonomy analysis.

#### Deliverables

- probe scripts
- lesion scripts
- mechanistic analysis report

#### Exit criteria

- At least one model can be analyzed in a way that connects benchmark failures to internal mechanisms.
- Mechanistic analyses support the claim that family-wise failures are scientifically meaningful.

### Phase 11. Release Package and Submission Readiness

#### Goal

Ship MapShift as a release-ready benchmark artifact with complete documentation and reproducibility support.

#### Work

1. Write top-level setup and quickstart documentation.
2. Write the benchmark card or evaluation card:
   - intended use
   - supported claims
   - assumptions
   - limitations
   - known failure modes
   - split policy
   - reporting guidance
3. Write executable examples for:
   - generating benchmark instances
   - applying interventions
   - running evaluation
   - reproducing key figures
4. Package canonical configs for all main experiments.
5. Package benchmark manifests or generation recipes for the release split.
6. Provide runbook documentation for cluster execution, even if the heavy jobs are run externally.
7. Provide a submission checklist for code accessibility, executability, and documentation completeness.

#### Deliverables

- release-ready repository
- benchmark card or evaluation card
- reproducibility runbook
- example configs and scripts
- submission checklist

#### Exit criteria

- A reviewer can clone the repository and run a documented benchmark smoke test.
- The codebase is organized around reuse and inspection, not only around one paper run.

## 8. Cross-Cutting Validation Plan

Validation is a separate workstream that runs in parallel with implementation.

### Unit tests

- schema validation
- manifest serialization
- deterministic seeding
- intervention parameter monotonicity
- metric correctness on hand-constructed examples

### Integration tests

- generate base environment -> apply intervention -> sample task -> run oracle evaluation
- produce family-wise metrics from a toy baseline
- reproduce a benchmark manifest exactly from stored provenance

### Regression tests

- no-op intervention equivalence
- unchanged split composition under frozen configs
- stable metric outputs on pinned fixtures

### Benchmark health tests

- coverage by family and severity
- task difficulty balance
- oracle solvability
- heuristic non-saturation
- absence of trivial impossible cases

## 9. Experiment Governance

The following rules must be followed during experimentation.

1. Never tune on the test split.
2. Never change intervention definitions after final validation tuning.
3. Never report only pooled scores in internal or external summaries.
4. Always store environment IDs and parent environment IDs in result tables.
5. Always run bootstrap analyses for final reported numbers.
6. Always include at least one protocol-comparison analysis in any major result bundle.
7. Always report compute and parameter counts for baselines.

## 10. Paper-Driven Output Plan

The implementation should be organized around producing the following paper outputs.

### Core figures

1. CEP evaluation pipeline figure.
2. Intervention family figure with matched base/intervened environments.
3. Family-wise degradation curves by severity.
4. Ranking comparison across evaluation protocols.
5. Metric-correlation or failure-profile separability figure.
6. Failure taxonomy visualization.
7. 2D-to-3D external-validity summary.

### Core tables

1. Benchmark specification table.
2. Baseline roster and fairness table.
3. Family-wise main results.
4. Protocol-sensitivity and rank-correlation table.
5. Split and benchmark-health summary.
6. Reproducibility and compute summary.

## 11. Order of Execution

Implementation should proceed in the following order, without skipping the freeze gates.

1. Write and freeze the benchmark specification.
2. Build the core software foundation.
3. Implement the 2D generator.
4. Implement and validate all four intervention families.
5. Implement exploration and the three task classes.
6. Build splits and leakage checks.
7. Implement metrics and statistical tooling.
8. Integrate baseline wrappers and sanity baselines.
9. Run the full 2D benchmark program.
10. Implement the 3D ProcTHOR tier.
11. Run the 3D external-validity program.
12. Run mechanistic analyses.
13. Finalize the release package and submission materials.

## 12. Definition of Done

MapShift is implemented only when all of the following are true.

1. The benchmark spec, code, configs, and documentation agree.
2. The 2D tier supports all four intervention families, all three task classes, and the full reporting hierarchy.
3. The 3D tier reproduces the same benchmark logic on ProcTHOR.
4. The validator suite demonstrates clean intervention isolation.
5. The split machinery and leakage checks are operational and documented.
6. The evaluation harness produces family-wise metrics, bootstrap CIs, and rank-stability analyses automatically.
7. The main experiments support P1, P2, and P3 directly.
8. The release package is executable, documented, and review-ready.

## 13. Immediate Next Documents to Write

The next documents to create after this plan are:

1. `docs/benchmark_spec.md`
2. `docs/evaluation_card.md`
3. `docs/release_checklist.md`

These documents should be written before any major baseline training begins.
