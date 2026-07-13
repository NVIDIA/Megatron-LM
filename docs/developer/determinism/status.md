---
orphan: true
---

# Determinism status

> **Audience:** Megatron developers. Setup and supported configurations are in
> the [user guide](../../user-guide/deterministic-training.md); definitions are
> in the [glossary](./glossary.md); per-operation detail is in the
> [op catalog](./op-catalog.md). The live roadmap is
> [issue #5785](https://github.com/NVIDIA/Megatron-LM/issues/5785).

## What deterministic mode does today

`--deterministic-mode` (see `megatron/training/determinism.py`) validates the
determinism environment variables and fills canonical defaults, rejects the
features that have no deterministic path (cross-entropy fusion and TP
communication overlap), and enables `torch.use_deterministic_algorithms(True)`.
Library code then selects the deterministic branches listed in the
[op catalog](./op-catalog.md). The user guide documents the exact flags and
environment values — they are not repeated here.

## Validation

- **Module-level bit-exact suite** (`tests/unit_tests/determinism/`): runs a
  model or block twice under restored RNG state and asserts bit-identical
  outputs and gradients — GPTModel, TransformerBlock, and HybridModel across a
  TP/EP/FSDP/PP/VPP matrix, including FP8 and FP4 recipes, with scheduling
  stressors to surface latent ordering races.
- **Performance gate**
  (`tests/performance_tests/shell_test_utils/determinism/`): runs a small
  recipe in deterministic and default mode under Nsight Systems, reports a
  per-range leaderboard, and fails when the deterministic step time exceeds the
  documented threshold.
- **End-to-end verification** is done by comparing full-precision training
  metrics across two independent runs (see the glossary's "Verification"
  note). Extending checked-in coverage to production-scale architectures is a
  roadmap item.

## Performance

Deterministic mode currently costs roughly 15% step time, varying by model and
precision; the goal is under 10%, with a stretch goal near 5%, so determinism
can be left on in production runs. The hotspot list and optimization progress
live in the [op catalog](./op-catalog.md) and
[issue #5785](https://github.com/NVIDIA/Megatron-LM/issues/5785).
