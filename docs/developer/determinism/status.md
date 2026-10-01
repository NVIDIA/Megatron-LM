---
orphan: true
---

# Determinism Status

> Setup and supported configurations are in the
> [user guide](../../user-guide/deterministic-training.md).

## Deterministic Mode

`--deterministic-mode` (refer to `megatron/training/determinism.py`) does the
following:

- Validates the determinism environment variables and fills canonical defaults
- Rejects features that have no deterministic path (cross-entropy fusion and
  tensor parallelism (TP) communication overlap)
- Enables `torch.use_deterministic_algorithms(True)`

The library code then selects the deterministic branches listed in the
[op catalog](./op-catalog.md). Refer to the user guide for exact flags and
environment values.

## Validation

- **Kernel-level bit-exact suite** (`tests/unit_tests/determinism/kernels/`):
  Replays every kernel Megatron dispatches (fused activations, Triton fusions,
  apex extensions, Transformer Engine wrappers, MoE, SSM, optimizer and
  inference kernels) on identical inputs and asserts byte-identical outputs and
  gradients. `manifest.py` registers each kernel with its test; the unit tests
  and the `linting` CI job fail when a kernel file is unregistered or a kernel
  change ships without a test change. Refer to [`testing.md`](./testing.md).
- **Module-level bit-exact suite** (`tests/unit_tests/determinism/correctness/`): Runs a
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
  two independent runs (refer to the glossary's "Verification" note). The
  functional tests do the same against checked-in golden values: every
  pretraining case that does not opt out (`NON_DETERMINSTIC_RESULTS: 1` or
  `NVTE_ALLOW_NONDETERMINISTIC_ALGO: 1` in its `model_config.yaml`, which makes
  `run_ci_test.sh` compare approximately) is compared bit-exactly by
  `DeterministicTest` in `tests/functional_tests/python_test_utils/common.py`.
  Newly written golden values keep the full `float32` precision of the
  TensorBoard scalars and record it as `"value_precision": "full"`; files
  written before this convention carry no marker and are compared at five
  decimals until they are regenerated (refer to the glossary's "Verification"
  note). Extending
  checked-in coverage to production-scale architectures is a roadmap item.

## Performance

Deterministic mode increases step time by roughly 15%, varying by model and
precision. The goal is under 10%, with a stretch goal near 5%, so you can leave
determinism on in production runs. The hotspot list and optimization progress
live in the [op catalog](./op-catalog.md) and
[issue #5785](https://github.com/NVIDIA/Megatron-LM/issues/5785).
