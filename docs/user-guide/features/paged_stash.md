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

**Sync-free:** `--moe-flex-dispatcher-backend hybridep` and `--moe-expert-rank-capacity-factor` pre-size dispatch and grouped expert buffers from a user-controlled capacity, avoiding a per-step device query / realloc loop for buffer sizing. Expert compute can use either `--use-transformer-engine-op-fuser` or the device-initiated Transformer Engine GroupedTensor path (`--moe-grouped-gemm --moe-use-grouped-tensor`).

**Paged stashing:** `--moe-paged-stash` stores those activations in paged CUDA buffers (optional pinned host spill). It helps save activation memory; sync-free still works without it, at the cost of higher activation memory use.

Whenever `moe_expert_rank_capacity_factor` is set, a **runner** wraps forward-backward: after each pass it checks **stash overflow** (only with `--moe-paged-stash`) and **token over-budget**. If either hits any rank, the step **reruns once** without capacity padding and without paged stashing.

For Transformer Engine per-layer CUDA graphs, the whole MoE scope (either
`--cuda-graph-modules moe` or the default whole-layer scope) is supported by the sync-free
HybridEP + paged-stash configuration above. Its captured graph is tied to the static expert-rank
and stash capacities. Before graph capture, and during eager evaluation, capacity overflow can
still use the dynamic rerun. After graph capture, training fails immediately instead of attempting
an invalid rerun; increase the corresponding capacity factor and restart the job.

Transformer Engine warmup and graph capture both replay the previously recorded paged-stash
layer templates, preserving the training layer, microbatch, and virtual-pipeline coordinates. The
runtime paged-stash schedule remains unchanged. This mode requires a fixed runtime microbatch
schedule: `--cuda-graph-dynamic-microbatches` is not supported, the microbatch count may not change
after capture, and at least two CUDA graph warmup steps are required to record the pipeline
schedule. Graph capture temporarily expands Transformer Engine's final `_order` into a
capture-only schedule and restores the recorded runtime schedule afterward. This mode requires
Transformer Engine 2.19.0 or later, which includes
[Transformer Engine #2831](https://github.com/NVIDIA/TransformerEngine/pull/2831) and makes
ordered warmup follow `_order`; older installs fail configuration validation. It does not use
per-callable cursor hooks. The stash/reload kernels remain inside their corresponding CUDA graphs.

Because Transformer Engine captures each MoE layer as an independent graph, the auxiliary
pack/unpack stream must rejoin the graph's capture stream before that graph ends. These joins are
captured graph nodes and therefore execute on every replay. Stash/reload work launched by one MoE
graph consequently cannot overlap eager work or another graph beyond that graph boundary, although
work inside the same graph may still overlap before the join. This correctness constraint can
reduce paged-stash overlap and throughput; benchmark the target PP/VPP configuration.

## Prerequisites

HybridEP and TE grouped experts are required whenever `moe_expert_rank_capacity_factor` is set. The non-op-fuser path requires a Transformer Engine version whose GroupedLinear marks saved GroupedTensor activation buffers for paged stashing. It currently supports only fused SwiGLU or QuickGeGLU (`bias_activation_fusion=True`) without GLU interleaving; restricting the activation contract keeps dynamic-tensor marking at the fused autograd boundaries. With `moe_paged_stash` enabled: capacity factor must be set; no `cpu_offloading`; `offload_modules` must not include `expert_fc1`, `moe_act`, or `fused_group_mlp`. The runner is active whenever capacity factor is set (even without `--moe-paged-stash`) for over-budget reruns; stash overflow is checked only when paged stashing is on.

## Configuration

```bash
# Common static-budget configuration
--moe-token-dispatcher-type flex
--moe-flex-dispatcher-backend hybridep
--moe-expert-rank-capacity-factor <float>

# Paged stashing (to avoid memory waste due to fragmentation)
--moe-paged-stash

# Choose one expert-compute path:

# A. TE operation fuser (used by the full-iteration CUDA graph + CuTe DSL route)
--use-transformer-engine-op-fuser

# B. Device-initiated GroupedLinear, without the operation fuser
--moe-grouped-gemm
--moe-use-grouped-tensor
# Keep the default fused SwiGLU activation; do not pass --no-bias-swiglu-fusion.
```

Path B removes host-device synchronization from grouped GEMM split metadata, but FC1, activation,
and FC2 remain separate launches. Without a full-iteration CUDA graph it can therefore retain
significant CPU launch overhead even though the expert path is host-device sync-free.
The legacy multi-stream cuBLAS GroupedLinear path is not supported because it materializes split
metadata on the host; paged stashing would not make that expert path sync-free.

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
