# MapShift-2D v0.1 Release Freeze

Release identity: `mapshift_2d_v0_1`

Status: frozen 2D benchmark release. The ProcTHOR-compatible 3D files remain prototype/future work and are not part of the v0.1 scientific claim.

## Frozen Protocol Values

- Tier: `mapshift_2d`
- Map size: 96x96 cells
- Exploration budget: `T_exp=800` reward-free steps
- Motifs: `simple_loop`, `two_room_connector`, `branching_chain`, `asymmetric_multi_room_chain`, `offset_bottleneck`, `nested_bottleneck`, `deceptive_shortcut`, `disconnected_subregion`
- Splits: train uses `simple_loop`, `two_room_connector`, `branching_chain`; validation uses `asymmetric_multi_room_chain`, `offset_bottleneck`; test uses `nested_bottleneck`, `deceptive_shortcut`, `disconnected_subregion`
- Intervention families: metric, topology, dynamics, semantic
- Severity levels: 0, 1, 2, 3 for every family
- Task classes: planning, inference, adaptation
- Planning horizons: 64, 96, 128
- Inference horizons: 16, 32
- Adaptation budgets/horizons: 16, 32, 64
- Main study task samples per class: 3
- Main learned-baseline model seeds: 0, 1, 2, 3, 4
- Bootstrap: 1000 resamples, 95% intervals, resampling by `environment_model_seed_id`
- Learned-baseline Torch device: `torch_device="auto"` in calibration configs; set `MAPSHIFT_TORCH_DEVICE=cuda:0` to pin a CUDA GPU.

## Frozen Baseline Roster

- `oracle_post_intervention_planner`
- `same_environment_upper_baseline`
- `weak_heuristic_baseline`
- `monolithic_recurrent_world_model`
- `persistent_memory_world_model`
- `relational_graph_world_model`
- `structured_dynamics_world_model`

`prismx_reference_model` is intentionally omitted from the v0.1 main study because it is not implemented.

## Required Artifact Outputs

The release builder writes configs, split manifests, intervention and task recipes, benchmark health, raw study records, family-wise tables, protocol comparisons, severity curves, rendered Markdown tables, SVG figures, and a top-level release manifest.

Canonical reviewer smoke:

```bash
python3 scripts/build_benchmark.py --tier mapshift_2d --study-config configs/analysis/mapshift_2d_full_study_smoke_v0_1.json --output-dir outputs/releases/mapshift_2d_v0_1_smoke --print-summary
```

Canonical full reproduction:

```bash
python3 scripts/build_benchmark.py --tier mapshift_2d --study-config configs/analysis/mapshift_2d_full_study_v0_1.json --output-dir outputs/releases/mapshift_2d_v0_1_full --print-summary
```
