<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# TensorParallelMuonHT

`TensorParallelMuonHT` is Megatron Core's tensor-parallel implementation of Muon
Hyperball. Hyperball wraps a base optimizer by fixing the Frobenius norms of each
matrix weight and its optimizer update. See the
[Hyperball paper](https://arxiv.org/abs/2606.16899) and the
[Emerging Optimizers reference implementation](https://docs.nvidia.com/nemo/emerging-optimizers/latest/_modules/emerging_optimizers/orthogonalized_optimizers/muon_hyperball.html).
The related [ELR method](https://hy.tencent.ai/research/elr) rescales the update by
$\lVert W\rVert_F / \lVert u\rVert_F$ without Hyperball's post-update projection;
`TensorParallelMuonHT` specifically implements the fixed-radius Hyperball rule below.

## Update rule

For a matrix weight $W_t$, let $u_t$ be the momentum-processed and
Muon-orthogonalized update, let $\eta_t$ be the learning rate, and let $R$ be
the configured Hyperball radius. Define

$$
\operatorname{Normalize}(X) = \frac{X}{\lVert X \rVert_F}.
$$

The optimizer applies

$$
W_{t+1}
= R\,\operatorname{Normalize}\!\left(
W_t - \eta_t R\,\operatorname{Normalize}(u_t)
\right).
$$

The inner normalization fixes the global update norm to $R$ before the learning
rate is applied. The outer normalization projects the updated weight back to the
sphere with radius $R$. `--muon-ht-radius` is required and asks the
emerging-optimizer factory to initialize every Muon-managed matrix to that fixed
positive radius before it constructs the optimizer or mixed-precision main-parameter
copies. The optimizer validates that direct callers also supply parameters already
initialized at $R$; it does not mutate initialization in its constructor.
`--muon-ht-eps` supplies the numerical-zero threshold. This explicit norm
constraint replaces weight decay, so `TensorParallelMuonHT` requires
`--weight-decay 0`.

Only two-dimensional, non-embedding weights are routed to Muon Hyperball. Megatron's
existing emerging-optimizer grouping sends embeddings, biases, normalization gains,
and other non-matrix parameters to the optimizer selected by
`--muon-scalar-optimizer` (`adam` by default, or `lion`).

## Emerging Optimizers integration

The class hierarchy is:

```text
emerging_optimizers.orthogonalized_optimizers.OrthogonalizedOptimizer
  -> megatron.core.optimizer.emerging_optimizers.TensorParallelMuon
    -> megatron.core.optimizer.emerging_optimizers.TensorParallelMuonHT
```

`TensorParallelMuonHT` reuses `TensorParallelMuon` for momentum, Newton-Schulz
orthogonalization, QKV splitting, and TP/GTP-aware Muon orthogonalization. It uses
the Emerging Optimizers extension points around the final weight update:

1. `pre_weight_update_fn_inplace` computes the global Frobenius norm of the
   orthogonalized update and rescales it to $R$.
2. `OrthogonalizedOptimizer.step` applies `W -= lr * update`.
3. `post_weight_update_fn_inplace` computes the global Frobenius norm of the new
   weight and projects it back to $R$.

## TP, expert TP, and GTP-remat sharding

A local shard norm is not the norm of the logical weight. For a logical tensor
with local shard $X^{(r)}$, the implementation first computes the FP32 local
squared norm

$$
s_r = \sum_{i,j} \left(X^{(r)}_{ij}\right)^2.
$$

It then uses `all_reduce(SUM)` over every axis that contains unique shards and
returns the square root of the resulting sum. TP and GTP-remat groups are
orthogonal axes, so applying their reductions in sequence covers their Cartesian
product without gathering the weight or update.

| Weight layout | Norm communication |
| --- | --- |
| Replicated dense weight | None |
| Tensor-parallel dense weight | `ProcessGroupCollection.tp` |
| Tensor-parallel expert weight | `ProcessGroupCollection.expt_tp` |
| GTP-remat dense weight | `ProcessGroupCollection.gtp_remat` |
| GTP-remat expert weight | `ProcessGroupCollection.expt_gtp_remat` |
| TP plus GTP-remat weight | Sequential reduction over both applicable groups |

The optimizer reads `tensor_model_parallel`, `partition_dim`, `expert_tp`,
`allreduce`, and `is_gtp_weight_remat` from the parameter metadata. It deliberately
does not reduce over replica-only axes, which would multiply the norm by the number
of identical copies. A sharded weight without the required explicit process group
fails fast instead of silently using a local norm.

## Checkpoint behavior

An earlier Hyperball implementation recorded a scalar `hyperball_R` per parameter.
Megatron's `torch_dist` distributed-checkpoint conversion expects ordinary tensor
optimizer state to have the parameter's shape, so that scalar can trigger a shape
mismatch for sharded parameters. The refactored Emerging Optimizers `HyperballHook`
instead takes one fixed radius at construction and returns no pre-update state.

`TensorParallelMuonHT` follows that design: the radius comes directly from
`OptimizerConfig` and is never added to serialized optimizer state, parameter
metadata, or transient per-parameter hook state. The pre- and post-update hooks use
the configured scalar and communicate only the distributed norms.

The configured radius is recreated when training configuration recreates the
optimizer, while the checkpointed weight is already on that sphere. No custom
`load_state_dict` invalidation or metadata copying is necessary. The momentum buffer
remains normal, parameter-shaped optimizer state and follows Megatron's existing
checkpoint path.

Configured fixed-radius initialization uses the same TP, expert-TP, and GTP-remat
reductions as optimizer updates. It runs once in the emerging-optimizer factory,
before Megatron creates FP32 main parameters or parameter views. If this initialization
communication becomes material at very large scale, parameters with the same shard
topology can be batched and measured with `multi_tensor_l2norm` in a follow-up.

## Usage

```bash
--optimizer muon_ht \
--weight-decay 0 \
--muon-ht-radius <positive-float> \
--muon-ht-eps 1e-15
```

The remaining `--muon-*` options configure the inherited `TensorParallelMuon`
behavior.

## Unit-test coverage

The following tests guard the implementation:

- `test_muon_ht_normalizes_update_and_preserves_weight_norm`: verifies both hook
  constraints directly.
- `test_muon_ht_sets_update_below_eps_to_zero`,
  `test_muon_ht_scales_update_equal_to_eps_to_weight_norm`, and
  `test_muon_ht_sets_weight_below_eps_to_zero`: verify the strict numerical-zero
  boundary for updates and weights.
- `test_muon_ht_optimizer_step_preserves_configured_radius_without_state`: verifies
  the real Emerging Optimizers step invokes both hooks, excludes persistent scalar
  radius state and metadata, and resumes with the configured radius.
- `test_muon_ht_fixed_radius_and_validation`: verifies fixed-radius hook behavior,
  strict epsilon validation, and rejection of simultaneous weight decay.
- `TestMuonOptimizerMultiRank.test_get_megatron_optimizer_muon_ht`: verifies public
  factory registration, configured-radius parameter initialization,
  mixed-precision wrapping, and a complete optimizer step.
- `TestMuonOptimizerMultiRankTP.test_muon_ht_uses_global_norm_for_tp_shards`:
  verifies update and weight norms across real TP shards.
- `TestMuonOptimizerMultiRankGTP.test_muon_ht_uses_global_norm_for_gtp_shards`:
  verifies update and weight norms across real GTP-remat shards.
