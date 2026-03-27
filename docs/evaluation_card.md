# MapShift Evaluation Card

## 1. Overview

MapShift is a benchmark and evaluation protocol for Counterfactual Embodied Planning (CEP).

It evaluates whether an agent can:

1. explore an environment without task reward
2. retain useful world knowledge from that exploration
3. plan or infer correctly after the environment is changed in a structured way

MapShift is intended as an evaluation artifact, not as a claim that a particular model family is universally best.

## 2. Artifact Summary

- Name: MapShift
- Artifact type: benchmark and evaluation protocol
- Primary domain: embodied planning and world-model evaluation
- Benchmark tiers:
  - MapShift-2D
  - MapShift-3D on ProcTHOR
- Intervention families:
  - metric
  - topology
  - dynamics
  - semantic
- Task classes:
  - planning
  - inference
  - adaptation

## 3. Evaluative Intent

MapShift is designed to improve how embodied systems are evaluated.

Specifically, it is intended to test:

- whether reward-free exploration produces world knowledge that remains useful after intervention
- whether structured interventions reveal weaknesses hidden by same-environment evaluation
- whether different intervention families expose different failure modes
- whether benchmark design choices change scientific conclusions

## 4. Supported Claims

Results on MapShift may support claims about:

- post-intervention planning robustness
- diagnostic robustness across intervention families
- relative strengths of memory-heavy, recurrent, relational, and structured-dynamics systems under the benchmark assumptions
- protocol sensitivity, including mis-ranking under standard same-environment evaluation

## 5. Unsupported Claims

Results on MapShift should not be used as evidence for:

- broad real-world robotics safety or reliability
- language-grounded instruction following
- unrestricted transfer to unseen physical embodiments
- human-like commonsense reasoning
- unconstrained semantic understanding outside the benchmark's cue system

## 6. Benchmark Design Summary

MapShift evaluates systems in two stages:

1. **Reward-free exploration:** the agent explores a base environment `e` for a fixed budget `T_exp`.
2. **Post-intervention evaluation:** an intervention `I_f(sigma)` modifies the environment into `e'`, then a task `q` is sampled from `Q(e, e')`.

The benchmark measures performance after the agent has to use or revise what it learned from exploration in the presence of controlled change.

## 7. Benchmark Composition

### Environments

MapShift contains:

- a 2D procedural environment tier for control and statistical power
- a 3D ProcTHOR tier for external validity

### Intervention families

- **Metric:** geometry or calibration changes
- **Topology:** connectivity changes
- **Dynamics:** transition-model changes
- **Semantic:** cue-meaning or identity changes

### Severity ladder

Each family includes four severity levels:

- `0`: none
- `1`: mild
- `2`: moderate
- `3`: strong

Severity is monotone in a measurable intervention parameter.

### Tasks

- **Planning tasks:** reach, reroute, or navigate under change
- **Inference tasks:** detect, infer, or answer counterfactual queries about change
- **Adaptation tasks:** use a limited post-shift interaction budget to recover performance

## 8. Primary Metrics

MapShift's primary metrics are reported per intervention family:

- success rate
- normalized path efficiency
- adaptation sample efficiency
- counterfactual planning accuracy

Secondary diagnostics may include:

- change-detection AUROC
- masked-state inference accuracy
- long-horizon rollout consistency

## 9. Reporting Guidance

MapShift SHOULD be reported family-wise first.

Recommended order of reporting:

1. family-wise primary metrics
2. severity-response curves
3. rank stability and protocol-comparison results
4. pooled score, if used

MapShift results should not be summarized only through a single weighted average.

## 10. Intended Users

MapShift is intended for:

- researchers studying embodied world models
- researchers studying planning under partial observability
- evaluation and benchmark researchers
- authors comparing evaluation protocols in embodied AI

It is not intended as a turnkey robotics deployment assessment.

## 11. Key Assumptions

MapShift depends on the following assumptions:

- reward-free exploration can produce useful latent or memory-based world knowledge
- the intervention families are sufficiently isolated to support diagnostic interpretation
- structural motif splits are enough to reduce trivial benchmark overfitting
- the benchmark environments are simplified but still scientifically informative for embodied evaluation

## 12. Known Limitations

- The 2D tier trades realism for control.
- The 3D tier inherits limitations from the underlying simulator.
- Semantic shifts are benchmark-defined and not equivalent to open-world semantics.
- Family isolation is an engineering target, not a metaphysical guarantee.
- Benchmark conclusions depend on the chosen exploration budget, horizon distribution, and task mix.
- Good performance on MapShift does not imply general real-world robustness.

## 13. Known Failure Modes the Benchmark Is Meant to Diagnose

MapShift is specifically designed to surface:

- local-geometry failure
- connectivity-update failure
- dynamics-calibration failure
- semantic-remapping failure
- relocalization failure

If these failure types do not emerge distinctly in practice, that is evidence against the benchmark's construct validity and should be reported.

## 14. Split and Leakage Policy

MapShift splits by:

- generator seed
- structural motif
- semantic remapping template where applicable

Users should not:

- tune on the test split
- collapse train, validation, and test motif families
- treat descendant intervention instances as independent of their base environment lineage

## 15. Reproducibility Expectations

A reusable MapShift release should provide:

- executable benchmark code
- environment generation code
- intervention generation code
- canonical configs
- evaluation scripts
- analysis scripts for core tables and figures
- provenance-bearing manifests for benchmark instances or generation recipes
- documentation on supported claims and limitations

## 16. Responsible Use Guidance

Users SHOULD:

- report family-wise metrics
- disclose benchmark version and split version
- document any deviation from the canonical protocol
- describe what conclusions the benchmark does and does not justify

Users SHOULD NOT:

- use MapShift as the sole evidence of real-world deployment readiness
- hide protocol changes behind pooled metrics
- compare methods under mismatched exploration budgets without explicit disclosure

## 17. Maintenance and Versioning

Each MapShift result should state:

- benchmark version
- tier used
- canonical config names
- code commit
- split version
- baseline versions

Breaking changes to benchmark semantics should increment the benchmark version and should not be silently mixed with earlier results.

## 18. Recommended Citation Surface

When releasing or using MapShift, accompanying materials should point readers to:

- the benchmark specification
- the evaluation card
- the release checklist
- the canonical configs used for reported results

These documents together define the intended use and interpretation of the benchmark.
