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
[Hyperball research page](https://hy.tencent.ai/research/elr), the
[paper](https://arxiv.org/abs/2606.16899), and the
[Emerging Optimizers reference implementation](https://docs.nvidia.com/nemo/emerging-optimizers/latest/_modules/emerging_optimizers/orthogonalized_optimizers/muon_hyperball.html).

## Update rule

For a matrix weight $W_t$, let $u_t$ be the momentum-processed and
Muon-orthogonalized update, let $\eta_t$ be the learning rate, and define

$$
R = \lVert W_0 \rVert_F,
\qquad
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
sphere with radius $R$. `--muon-ht-radius` can replace each matrix's initial norm
with one fixed positive radius. `--muon-ht-eps` supplies the denominator floor.
This explicit norm constraint replaces weight decay, so `TensorParallelMuonHT`
requires `--weight-decay 0`.

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

The Emerging Optimizers reference implementation records `hyperball_R` as scalar
per-parameter optimizer state. Megatron's `torch_dist` distributed-checkpoint
conversion expects ordinary tensor optimizer state to have the parameter's shape,
so a scalar radius can trigger a shape mismatch for sharded parameters.

`TensorParallelMuonHT` avoids adding the radius to the serialized optimizer state.
Instead, it treats the constrained checkpointed weight as the source of truth:

- the radius is captured lazily from the FP32 optimizer parameter at the first
  pre-update hook, after model and optimizer parameters have been restored;
- `_muon_ht_radius` is optimizer-only parameter metadata and is copied when
  Megatron creates FP32 main parameters or parameter views;
- `load_state_dict` invalidates any in-memory radius so the next step reconstructs
  it from the restored weight; and
- a configured `--muon-ht-radius` is reconstructed directly from the configuration.

Because every completed step projects the weight to its radius, no extra scalar is
needed to resume the constraint. The momentum buffer remains normal, parameter-shaped
optimizer state and follows Megatron's existing checkpoint path.

## Usage

```bash
--optimizer muon_ht \
--weight-decay 0 \
--muon-ht-eps 1e-8
```

Optionally add `--muon-ht-radius <positive-float>` to use one shared radius for all
Muon-managed matrices. The remaining `--muon-*` options configure the inherited
`TensorParallelMuon` behavior.

## Unit-test coverage

The following tests guard the implementation:

- `test_muon_ht_normalizes_update_and_preserves_weight_norm`: verifies both hook
  constraints directly.
- `test_muon_ht_optimizer_step_preserves_initial_weight_norm`: verifies the real
  Emerging Optimizers step invokes both hooks, excludes a scalar radius from
  optimizer state, and reconstructs the radius after `load_state_dict`.
- `test_muon_ht_fixed_radius_and_validation`: verifies fixed-radius initialization
  and rejects simultaneous weight decay.
- `TestMuonOptimizerMultiRank.test_get_megatron_optimizer_muon_ht`: verifies public
  factory registration, mixed-precision wrapping, and a complete optimizer step.
- `TestMuonOptimizerMultiRankTP.test_muon_ht_uses_global_norm_for_tp_shards`:
  verifies update and weight norms across real TP shards.
- `TestMuonOptimizerMultiRankGTP.test_muon_ht_uses_global_norm_for_gtp_shards`:
  verifies update and weight norms across real GTP-remat shards.
- `test_copy_optimizer_param_metadata_preserves_muon_ht_radius`: verifies radius
  metadata survives creation of optimizer parameter copies and views.
