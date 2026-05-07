# MapShift v0.1 Release Status

This file is an internal status snapshot, not a reviewer runbook. The reviewer-facing artifact instructions are `README.md` and `ARTIFACT_EVALUATION.md`; versioned configs under `configs/` are the source of truth for executable settings.

## Current Release

- Release identity: `mapshift_2d_v0_1`
- Primary tier: `mapshift_2d`
- Scientific scope: controlled post-intervention evaluation (CPE) for embodied world models
- Motifs and splits: 24 motifs, with 10 train, 6 validation, and 8 held-out test motifs
- Families: metric, topology, dynamics, semantic
- Severities: 0, 1, 2, 3
- Task classes: planning, inference, adaptation
- Exploration budget: `T_exp=800`
- Bootstrap reporting: 1000 grouped resamples by `environment_model_seed_id`

## Completed Artifact Gates

- Root README, artifact evaluation guide, license, citation metadata, and dependency pins are present.
- Canonical benchmark, environment, intervention, task, baseline, analysis, and schema configs are tracked.
- Smoke/full reproduction, deterministic mechanism diagnostic, MiniGrid sanity-transfer, and capacity add-on commands are documented.
- Release builder can generate configs, split manifests, intervention/task recipes, benchmark health, study records, family-wise tables, protocol comparisons, rendered Markdown/SVG outputs, and provenance manifests.
- Validation tooling covers schema loading, split leakage, intervention validators, task rejection accounting, benchmark-health checks, metrics/reporting, MiniGrid adapter behavior, and integration-level study execution.
- The ProcTHOR-compatible code is documented as prototype/future work and excluded from the frozen v0.1 scientific claim.

## Deliberate Non-Claims

- The artifact is not a deployment-realism, robotics-safety, language-grounding, or commonsense-reasoning benchmark.
- Generated outputs are produced by artifact commands and are not a separately hosted static dataset in this repository.
- Optional generated output bundles may be hosted separately; if treated as dataset-like assets, they should carry Croissant/Responsible-AI metadata.

## Deferred Work

- Public camera-ready release tags and hosted generated-output bundles are separate release-management steps.
- Future 3D/ProcTHOR or language-conditioned claims require a new release claim and separate validation.
