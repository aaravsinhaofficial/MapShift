# MapShift

MapShift is a benchmark and evaluation protocol for Counterfactual Embodied Planning.

It is designed to answer a specific evaluation question:

> After an agent explores an environment without task reward, can it still plan correctly when the environment is changed in a structured way?

## Project documents

- [Implementation Plan](/Users/aaravsinha/MapShift/docs/IMPLEMENTATION_PLAN.md)
- [Benchmark Specification](/Users/aaravsinha/MapShift/docs/benchmark_spec.md)
- [Evaluation Card](/Users/aaravsinha/MapShift/docs/evaluation_card.md)
- [Release Checklist](/Users/aaravsinha/MapShift/docs/release_checklist.md)

## Repository status

The repository currently contains:

- benchmark planning and documentation
- a Python package scaffold for the benchmark codebase
- canonical draft release configs and machine-readable config schemas
- a first executable config-driven 2D path for generation, intervention, and task sampling
- validation and smoke-test entry points

The 2D generator, intervention families, and task sampler now have a working first path. The broader simulator, baselines, runners, and analysis stack remain partially scaffolded.

## Quickstart

Validate the draft release config bundle:

```bash
python3 scripts/validate_benchmark.py configs/benchmark/release_v0_1.json
```

Run the current unit and smoke tests:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```
