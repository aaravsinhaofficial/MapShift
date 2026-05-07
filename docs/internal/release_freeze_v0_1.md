# MapShift-2D v0.1 Release Freeze

Release identity: `mapshift_2d_v0_1`

Status: frozen 2D benchmark release. The ProcTHOR-compatible 3D files remain prototype/future work and are not part of the v0.1 scientific claim.

## Frozen Protocol Values

- Tier: `mapshift_2d`
- Map size: 96x96 cells
- Exploration budget: `T_exp=800` reward-free steps
- Motifs: `simple_loop`, `two_room_connector`, `branching_chain`, `asymmetric_multi_room_chain`, `offset_bottleneck`, `nested_bottleneck`, `deceptive_shortcut`, `disconnected_subregion`, `spiral_loop`, `ladder_loop`, `double_connector_rooms`, `zigzag_chain`, `forked_bottleneck`, `hub_spoke_deadends`, `parallel_corridor`, `ring_with_tail`, `offset_loop_bridge`, `split_hallway`, `braided_shortcut`, `narrow_gate_cluster`, `cul_de_sac_shortcut`, `island_bridge`, `broken_bridge_islands`, `asymmetric_loop_chain`
- Splits: 10 train motifs, 6 validation motifs, and 8 held-out test motifs, as specified in `configs/env2d/release_v0_1.json`
- Intervention families: metric, topology, dynamics, semantic
- Severity levels: 0, 1, 2, 3 for every family
- Task classes: planning, inference, adaptation
- Planning horizons: 64, 96, 128
- Inference horizons: 16, 32
- Adaptation budgets/horizons: 16, 32, 64
- Main study task samples per class: 3
- Main learned-baseline model seeds: 0, 1, 2, 3, 4
- Bootstrap: 1000 resamples, 95% intervals, resampling by `environment_model_seed_id`
- Learned-baseline Torch device: `torch_device="auto"` in calibration configs; set `MAPSHIFT_TORCH_DEVICE=cuda:0` to pin a CUDA GPU and `MAPSHIFT_CHECKPOINT_DIR` to isolate run checkpoints.

## Frozen Baseline Roster

- `oracle_post_intervention_planner`
- `same_environment_upper_baseline`
- `weak_heuristic_baseline`
- `stale_map_planner` for deterministic mechanism diagnostics
- `classical_belief_update_planner`
- `monolithic_recurrent_world_model`
- `persistent_memory_world_model`
- `relational_graph_world_model`
- `structured_dynamics_world_model`
- `pretrained_structured_graph_world_model` as an append-only capacity row

`prismx_reference_model` is intentionally omitted from the v0.1 study because it is not implemented. Large-capacity rows such as `persistent_memory_world_model_large_v0_1` and `pretrained_structured_graph_world_model_1m_v0_1` are evaluated as add-ons so the original full-roster study does not need to be recomputed.

## Required Artifact Outputs

The release builder can generate configs, split manifests, intervention and task recipes, benchmark health, study records, family-wise tables, protocol comparisons, severity curves, rendered Markdown tables, SVG figures, and a top-level release manifest.

Canonical reviewer smoke:

```bash
python3 scripts/build_benchmark.py --tier mapshift_2d --study-config configs/analysis/mapshift_2d_full_study_smoke_v0_1.json --output-dir outputs/releases/mapshift_2d_v0_1_smoke --print-summary
```

Canonical full reproduction:

```bash
python3 scripts/build_benchmark.py --tier mapshift_2d --study-config configs/analysis/mapshift_2d_full_study_v0_1.json --output-dir outputs/releases/mapshift_2d_v0_1_full --print-summary
```
