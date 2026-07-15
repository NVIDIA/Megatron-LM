---
orphan: true
---

# Determinism Status

> Setup and supported configurations are in the
> [user guide](../../user-guide/deterministic-training.md).

## Deterministic Mode

`--deterministic-mode` (refer to `megatron/training/determinism.py`) validates the
determinism environment variables and fills canonical defaults, rejects the
features that have no deterministic path (cross-entropy fusion and TP
communication overlap), and enables `torch.use_deterministic_algorithms(True)`.
The library code then selects the deterministic branches listed in the
[op catalog](./op-catalog.md). Refer to the user guide for exact flags and
environment values.

## Validation

- **Module-level bit-exact suite** (`tests/unit_tests/determinism/`): Runs a
  model or block twice under restored RNG state and asserts bit-identical
  outputs and gradients. Coverage includes:

  - GPTModel, TransformerBlock, and HybridModel
  - Tensor parallelism, expert parallelism, fully sharded data parallel, pipeline
    parallelism, and virtual pipeline parallelism
  - FP8 and FP4 recipes
  - Scheduling stressors to surface latent ordering races
- **Performance gate**
  (`tests/performance_tests/shell_test_utils/determinism/`): Runs a small
  recipe in deterministic and default mode under Nsight Systems, reports a
  per-range leaderboard, and fails when the deterministic step time exceeds the
  documented threshold.
- **End-to-end verification**: Compares full-precision training metrics across
  two independent runs (refer to the glossary's "Verification" note). Extending
  checked-in coverage to production-scale architectures is a roadmap item.

## Performance

Deterministic mode increases step time by roughly 15%, varying by model and
precision. The goal is under 10%, with a stretch goal near 5%, so you can leave
determinism on in production runs. The hotspot list and optimization progress
live in the [op catalog](./op-catalog.md) and
[issue #5785](https://github.com/NVIDIA/Megatron-LM/issues/5785).
