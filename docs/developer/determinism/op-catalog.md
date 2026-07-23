---
orphan: true
---

# Determinism Operation Catalog

> This content is for developers reviewing or extending deterministic-mode
> coverage. Terms are defined in the [glossary](./glossary.md).

The catalog is organized into two buckets. The project goal is to keep shrinking
the second bucket (operations that deterministic mode cannot support yet) and to keep
making the first bucket (operations with a deterministic code path) faster.

Most operations in a training step need no entry here at all. Elementwise ops,
GEMMs under the pinned cuBLAS workspace, rank-indexed collectives (all-gather,
all-to-all, broadcast), stable sorts, and unique-index writes are deterministic
as-is. The tables list only the operations where a choice is made.

## Deterministic Code Path Operations

Selected by `torch.are_deterministic_algorithms_enabled()` or
`config.deterministic_mode`. The default-mode path stays in the other branch.

| Operation | Where | Deterministic Path | Default Path |
| --- | --- | --- | --- |
| MoE token unpermute (combine) | `megatron/core/transformer/moe/moe_utils.py:517` | `index_add_` — deterministic under torch deterministic algorithms and CUDA-graph safe | `scatter_add_` (atomic accumulation) |
| MoE routing map and probabilities | `megatron/core/transformer/moe/moe_utils.py:823` | `index_put_(accumulate=False)` row-wise writes | out-of-place `scatter` |
| Vocab-parallel embedding | `megatron/core/tensor_parallel/layers.py:299` | direct indexing `weight[idx]` (deterministic backward) | `F.embedding` (non-deterministic atomic backward) |
| Gated-delta-net kernel | `megatron/core/ssm/gated_delta_net.py:212` | torch `chunk_gated_delta_rule` | FLA fused kernel |
| Gated-delta-net causal conv1d | `megatron/core/ssm/gated_delta_net.py:429` | `F.conv1d` (plus transposes) | FLA `causal_conv1d` |
| Mamba/SSM Triton ops | `megatron/core/ssm/ops/determinism.py` | one fixed autotune config plus a zero-initialized tiled workspace reduced with an ordered `sum` | timing-based autotune, uninitialized workspace |
| Transformer Engine attention | `megatron/core/extensions/transformer_engine.py:1698` | requires `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`, under which TE selects only backends that support deterministic execution (including deterministic FlashAttention backward) | TE picks freely, including atomic-accumulation attention backward |
| Inference DP scheduling and RL rollout order | `megatron/core/inference/engines/dynamic_engine.py`, `megatron/rl/rl_utils.py` | sort by stable key | completion order |

The following environment controls make the rest of the step deterministic.
The flag `--deterministic-mode` validates and defaults these settings. For
details, refer to `megatron/training/determinism.py`.

- `NCCL_ALGO=Ring`. Tree is rejected because its reduction order is not
  user-controllable.
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` (or `:16:8`).
- `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`.
- `MAMBA_DETERMINISTIC` must not be disabled.

## Operations Without Determinism Support

Deterministic mode either rejects these at validation (fails closed) or they
are known open gaps.

| Operation or Feature | Where Enforced or Observed | Status |
| --- | --- | --- |
| Fused cross-entropy loss (`--cross-entropy-loss-fusion`) | rejected by `--deterministic-mode` (`megatron/training/determinism.py`) | The fused kernel is non-deterministic. Whether a deterministic variant is feasible remains an open question. Until then, the framework uses the native vocab-parallel path. |
| TP communication overlap (`--tp-comm-overlap`) | rejected by `--deterministic-mode` | Overlapped collective ordering is not reproducible. |
| Packed sequence (`thd`) in gated-delta-net | `megatron/core/ssm/gated_delta_net.py:313` assert | No deterministic packed-sequence SSM path exists yet. |
| Cross-allocation floating-point collectives (TP all-reduce, DP grad reduce-scatter) | open gap | `NCCL_ALGO=Ring` pins the algorithm but not the physical ring an allocation receives. The environment variable alone does not guarantee bit-exactness across *different* allocations for these reductions. Runs repeated within one allocation, or on allocations with identical topology, remain bit-exact. |

## Performance Notes

The deterministic paths above cost roughly 15% of step time compared to
default mode, which varies by model. MoE-heavy models pay more. Measured
examples range from approximately 4% on a large dense model to approximately
17% on a hybrid MoE model. The measured hotspots are the deterministic MoE
scatter and unpermute path, the sorted router top-k, attention backward, and
grouped-GEMM wgrad.

Reducing this cost is a tracked workstream in
[issue #5785](https://github.com/NVIDIA/Megatron-LM/issues/5785). A change to
any row above needs a bit-exact test and a deterministic-vs-default performance
comparison (`tests/performance_tests/shell_test_utils/determinism/`).
