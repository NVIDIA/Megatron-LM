<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# MoE Paged Stash

*This is an experimental feature and may change.*

**Paged stash** = **sync-free** expert execution + **paged stashing** (packing routed-expert activations for backward into paged buffers).

**Sync-free:** `--moe-flex-dispatcher-backend hybridep`, `--use-transformer-engine-op-fuser`, and `--moe-expert-rank-capacity-factor` pre-size dispatch and fused grouped expert buffers from a user-controlled capacity, avoiding a per-step device query / realloc loop for buffer sizing.

**Paged stashing:** `--moe-paged-stash` stores those activations in paged CUDA buffers (optional pinned host spill). It helps save activation memory; sync-free still works without it, at the cost of higher activation memory use.

Whenever `moe_expert_rank_capacity_factor` is set, a **runner** wraps forward-backward: after each pass it checks **stash overflow** (only with `--moe-paged-stash`) and **token over-budget**. If either hits any rank, the step **reruns once** without capacity padding and without paged stashing.

Native dropless HybridEP leaves both `moe_expert_capacity_factor` and
`moe_expert_rank_capacity_factor` unset. That path dynamically resolves the exact permuted-token
count and incurs a host synchronization. The rank-capacity path instead makes the common attempt
sync-free and graph-capturable; its over-budget eager retry is what preserves dropless step
semantics.

## Prerequisites

HybridEP + TE fused grouped experts are required whenever `moe_expert_rank_capacity_factor` is set. With `moe_paged_stash` enabled: capacity factor must be set; no `cpu_offloading`; `offload_modules` must not include `expert_fc1`, `moe_act`, or `fused_group_mlp`. The runner is active whenever capacity factor is set (even without `--moe-paged-stash`) for over-budget reruns; stash overflow is checked only when paged stashing is on.

## Configuration

```bash
# Sync-free
--moe-token-dispatcher-type flex
--moe-flex-dispatcher-backend hybridep
--use-transformer-engine-op-fuser
--moe-expert-rank-capacity-factor <float>

# Paged stashing (to avoid memory waste due to fragmentation)
--moe-paged-stash
```

## CUDA graph interaction

The static rank budget provides graph-capturable HybridEP dispatch shapes, while paged stash keeps
the corresponding variable live activation count in graph-safe preallocated storage. Together
they allow capture of the complete local model chunk rather than only static attention/router
sub-regions:

```bash
--cuda-graph-impl local
--cuda-graph-granularity chunk
--cuda-graph-dynamic-microbatches
```

`--cuda-graph-dynamic-microbatches` is required for this combination because it enables reusable
physical in-flight slots. For PP greater than one, every slot captures one schedule-independent
operation pair: its forward always stashes its activation and its matching backward reloads that
activation. This deliberately omits the eager scheduler's neighboring-microbatch prefetch/discard
optimization, whose choice depends on the absolute microbatch schedule. For PP=1, each forward is
immediately followed by its backward, so the original activation is retained and no stash/reload
is needed. The same captured slots can therefore replay iterations with different microbatch
counts without recapture. The reload kernel also zero-fills the unused tail of each static-shape
activation in the same launch, matching the profiling path when the real routed-token count changes
between replays without adding a separate memset graph node.

This schedule-independent behavior is enabled only for dynamic local chunk graphs. It does not
change the legacy `cuda_graph_impl=none` paged-stash scheduler, which still expects the packed
microbatch schedule recorded during profiling to remain fixed.

CUDA graph capture remains deferred until paged-stash profiling has completed and its buffers have
been allocated. When the first iteration does not exercise every topology-required physical slot,
profiling continues across subsequent eager iterations and accumulates token maxima until all slots
have appeared. This avoids sizing the stash pool from a small first GA and then overflowing only
because a later iteration has more in-flight activations. The tradeoff is delayed graph capture
until the workload has reached the topology-complete slot pattern. The graph path does not disable
stash-overflow or token-over-budget handling. An overflow/over-budget retry bypasses the static
chunk graph and runs through the existing eager dropless fallback; the next normal step resumes
graph replay.

The graph-safe TE fused grouped-MLP configuration currently used with this path requires SM100+
and MXFP8. A representative precision configuration is:

```bash
--fp8-format e4m3
--fp8-recipe mxfp8
--fp8-param-gather
--reuse-grad-buf-for-mxfp8-param-ag
```

CUDA graph pools retain their captured activation and stash storage. Each physical in-flight slot
within a local model chunk has one runner pair and one pool; simultaneously live slots and
different local model chunks use distinct pools. Size
`--max-seqlen-per-dp-cp-rank`, global batch size, pipeline in-flight depth, expert capacity, and
stash buffer factors together.

## Tuning (paged stashing only)

```bash
# Page size for stashing
--moe-paged-stash-page-size 64
# CUDA stashing buffer scaling factor (default 1.10)
--moe-paged-stash-buffer-size-factor-cuda 1.10
# Host spill (0 = off); same sign rule as CUDA
--moe-paged-stash-buffer-size-factor-cpu 0.0
```

## What `moe_expert_rank_capacity_factor` and `moe_paged_stash_buffer_size_factor_cuda` mean

Both are **multipliers on buffer size relative to the perfectly balanced case**—the space you would need if routed tokens were evenly distributed across expert ranks. A larger factor adds headroom for real-world **skew**.

## Choosing `moe_expert_rank_capacity_factor` and stash buffer scales

Profile how far real routing departs from the **balanced** reference, then pick factors so **skew spikes** rarely exceed your margin (avoid constant reruns).

- **`moe_expert_rank_capacity_factor`:** pick from profiles so **over-budget token drop** is uncommon; set **slightly above** the profiled value so reruns stay rare.
- **`moe_paged_stash_buffer_size_factor_cuda`:** size from the **same stats** (peaks vs averages) so **stash overflow** is uncommon; undersizing triggers reruns like over-budget.
- **`moe_paged_stash_buffer_size_factor_cpu`:** set **> 0** to allow **spill to pinned host** when CUDA pages are full—often **avoids overflow / rerun** at the cost of host memory and more overhead from paged stashing.
