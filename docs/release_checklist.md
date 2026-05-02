# MapShift Release Checklist

## 1. Purpose

This checklist is the release gate for MapShift.

It is intended to ensure that:

- the benchmark contract is frozen before large-scale experiments
- the codebase is executable and inspectable
- the benchmark is documented well enough for external review
- benchmark claims, assumptions, and limitations are explicit
- the repository is ready for artifact release and paper submission

This checklist should be reviewed before:

- internal release
- large-scale benchmark runs
- paper submission
- camera-ready artifact finalization

## 2. Release Stages

### Stage A. Benchmark freeze

- [ ] `docs/benchmark_spec.md` exists and is up to date
- [ ] supported claims and non-claims are frozen
- [ ] intervention families are frozen
- [ ] task classes are frozen
- [ ] reporting hierarchy is frozen
- [ ] hypotheses H1-H4 are frozen
- [ ] freeze changes are tracked in a changelog

### Stage B. Protocol freeze

- [ ] canonical exploration budget `T_exp` is frozen
- [ ] canonical horizon distributions are frozen
- [ ] severity ladders are frozen for all families
- [ ] split rules are frozen
- [ ] task sampling policy is frozen
- [ ] primary metrics are frozen
- [ ] episode termination rules are frozen

### Stage C. Final evaluation freeze

- [ ] baseline roster is frozen
- [ ] hyperparameter policy is frozen
- [ ] canonical seed counts are frozen
- [ ] bootstrap procedure is frozen
- [ ] figure and table templates are frozen

## 3. Repository and Code Structure

- [ ] repository structure matches the implementation plan or documented deviations
- [ ] environment code is separated from intervention code
- [ ] task code is separated from metric code
- [ ] baseline wrappers use a common API
- [ ] configs exist for benchmark generation, evaluation, baselines, and analysis
- [ ] provenance or manifest utilities exist
- [ ] no critical benchmark logic depends on undocumented ad hoc scripts

## 4. Executability

- [ ] installation instructions exist
- [ ] environment dependencies are pinned or version-ranged explicitly
- [ ] a documented smoke test exists
- [ ] smoke test runs from a clean checkout
- [ ] benchmark generation can be executed from config
- [ ] evaluation can be executed from config
- [ ] analysis can be executed from saved outputs
- [ ] failures produce actionable logs rather than silent corruption

## 5. Benchmark Artifacts

- [ ] canonical benchmark configs exist
- [ ] canonical split manifests or generation recipes exist
- [ ] intervention manifests or intervention recipes exist
- [ ] task manifests or task recipes exist
- [ ] run manifests include benchmark version, config hash, code version, and seeds
- [ ] example outputs are included for inspection

## 6. Validation and Quality Gates

### Unit and integration tests

- [ ] schema validation tests exist
- [ ] manifest round-trip tests exist
- [ ] deterministic seeding tests exist
- [ ] metric correctness tests exist
- [ ] end-to-end integration tests exist for environment -> intervention -> task -> evaluation

### Intervention validation

- [ ] severity `0` reproduces the base environment
- [ ] geometry-only changes preserve topology labels unless intended
- [ ] topology-only changes preserve semantic identities
- [ ] semantic-only changes preserve navigable geometry
- [ ] dynamics-only changes preserve geometry
- [ ] severity monotonicity is tested for each family

### Benchmark health

- [ ] split coverage reports exist
- [ ] task difficulty reports exist
- [ ] oracle solvability checks exist
- [ ] heuristic non-saturation checks exist
- [ ] leakage checks exist for motifs, geometry, topology, semantics, and task templates

## 7. Documentation

- [ ] top-level `README.md` explains what MapShift is
- [ ] `docs/benchmark_spec.md` is current
- [ ] `docs/evaluation_card.md` is current
- [ ] `docs/release_checklist.md` is current
- [ ] setup instructions exist
- [ ] quickstart instructions exist
- [ ] example commands exist for generation, evaluation, and analysis
- [ ] benchmark versioning policy is documented
- [ ] protocol deviations are documented

## 8. Reporting Readiness

- [ ] family-wise reporting is the default in scripts and docs
- [ ] pooled-score reporting, if any, is explicitly labeled supplementary
- [ ] bootstrap confidence interval tooling exists
- [ ] rank-stability tooling exists
- [ ] protocol-comparison tooling exists
- [ ] figure generation scripts exist for core plots
- [ ] table generation scripts exist for core tables
- [ ] compute and parameter reporting fields exist for baselines

## 9. Baseline Readiness

- [x] baseline API is documented
- [x] monolithic recurrent baseline wrapper exists
- [x] persistent-memory baseline wrapper exists
- [x] structured-dynamics baseline wrapper exists
- [x] relational or graph baseline wrapper exists
- [x] reference structured baseline is intentionally omitted with explanation
- [x] oracle baseline exists
- [x] same-environment upper/reference baseline exists
- [x] weak heuristic baseline exists
- [x] fairness policy is documented

## 10. Main Experimental Readiness

### Construct validity

- [ ] scripts exist for family-wise degradation analysis
- [ ] scripts exist for cross-family correlation analysis
- [ ] scripts exist for severity monotonicity analysis
- [ ] scripts exist for matched-pair ablations
- [ ] scripts exist for failure-profile separability analysis

### Discriminative power

- [ ] scripts exist for rank comparison by family
- [ ] scripts exist for effect-size reporting
- [ ] scripts exist for saturation checks
- [ ] scripts exist for bootstrap rank distributions

### Protocol sensitivity

- [ ] scripts exist for same-environment vs CEP comparison
- [ ] scripts exist for no-exploration vs exploration comparison
- [ ] scripts exist for pooled-score vs family-wise comparison
- [ ] scripts exist for short-horizon vs long-horizon comparison

### Failure taxonomy

- [ ] failure labeling logic exists
- [ ] failure taxonomy definitions are documented
- [ ] taxonomy outputs can be exported with run IDs and environment IDs

## 11. 3D Tier Prototype Status

- [x] ProcTHOR-compatible files are documented as prototype/future work
- [x] 3D is excluded from the frozen v0.1 scientific claim
- [ ] future 3D scene generation, interventions, tasks, and reporting require a separate release claim

## 12. Mechanistic Analysis Readiness

- [ ] probe scripts exist or omission is documented
- [ ] lesion scripts exist or omission is documented
- [ ] at least one interpretable analysis path is documented

## 13. Accessibility and Submission Compliance

### General artifact accessibility

- [ ] code is hosted in a location accessible to reviewers
- [ ] reviewers do not need to request access personally
- [ ] code is documented and executable
- [ ] private assets, if any, are described with reviewer-access instructions

### NeurIPS ED-specific checks

The following reflect the official NeurIPS 2026 Evaluations & Datasets materials:

- [ ] submission uses the correct ED track
- [ ] all required code for the executable benchmark artifact is accessible at submission time
- [ ] if no new dataset is introduced, dataset-hosting and Croissant requirements are not incorrectly claimed as satisfied or required
- [ ] if any new dataset is introduced later, hosting and Croissant requirements are satisfied
- [ ] review mode choice is documented: double-blind by default, single-blind only if justified by artifact constraints

Reference links:

- NeurIPS 2026 ED CFP: `https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets`
- NeurIPS 2026 ED FAQ: `https://neurips.cc/Conferences/2026/EvaluationsDatasetsFAQ`

## 14. Camera-Ready Readiness

- [ ] public release plan exists
- [ ] benchmark version tag or release tag exists
- [ ] canonical configs are included in the release
- [ ] benchmark card and documentation are included in the release
- [ ] license is present
- [ ] paper figures and tables can be regenerated from released scripts and configs, subject to external compute

## 15. Sign-Off

Before a release is considered complete, the following should be recorded:

- [ ] benchmark version
- [ ] release date
- [ ] commit hash
- [ ] release owner
- [ ] checklist reviewer
- [ ] known limitations at release time
- [ ] deferred items, if any
