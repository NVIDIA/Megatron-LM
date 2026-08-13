<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software or related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# CUDA Graph

CUDA Graphs reduce kernel-launch overhead by recording GPU operations once and replaying the recording on subsequent iterations. Megatron-LM provides three CUDA graph implementations controlled by `--cuda-graph-impl`.

For implementation background and design details, see NVIDIA's
[Transformer Engine and Megatron-LM CUDA Graph Support](https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/te-megatron-cuda-graphs.html).
That article is a useful conceptual reference, but some examples there still use older flags such as
`--enable-cuda-graph` or `--cuda-graph-scope full_iteration`; in this repository, prefer
`--cuda-graph-impl local|transformer_engine|full_iteration` as documented below.

## Overview

CUDA graph behavior is set by the following flags:

| Flag | Values | Purpose |
|---|---|---|
| `--cuda-graph-impl` | `none` / `local` / `transformer_engine` / `full_iteration` | Which capture backend or strategy to use |
| `--cuda-graph-granularity` | `layer` / `chunk` | Training capture boundary; `chunk` is supported only by `local` |
| `--cuda-graph-modules` | `attn` / `mlp` / `moe` / `moe_router` / `moe_preprocess` / `mamba` | Per-layer **training** capture coverage; must be empty for `chunk` and `full_iteration` |
| `--cuda-graph-dynamic-microbatches` | flag | Support changing runtime microbatch counts for TE layer and local chunk graphs, including local chunk graphs with paged stash |
| `--cuda-graph-num-microbatch-slots` | positive integer | Optional local-chunk in-flight slot count; requires dynamic microbatches |
| `--inference-cuda-graph-scope` | `none` / `layer` / `block` | Granularity of CUDA graphs during **inference**; only `local` supports non-`none` values |

Supported combinations:

| `--cuda-graph-impl` | Backend | Training capture | Inference capture |
|---|---|---|---|
| `none` | — | off | off |
| `local` | MCore `CudaGraphManager` | per-layer or full local model-chunk capture per in-flight slot | `layer` (default) or `block`, controlled by `--inference-cuda-graph-scope` |
| `transformer_engine` | TE `make_graphed_callables()` | per-layer, controlled by `--cuda-graph-modules` | not supported (`none` only) |
| `full_iteration` | MCore `FullCudaGraphWrapper` | one graph per training iteration; `--cuda-graph-modules` must be empty | not supported (`none` only) |

---

## CUDA Graph — Local Implementation (`--cuda-graph-impl local`)

Uses MCore's built-in `CudaGraphManager`. During training, `--cuda-graph-granularity layer`
(the default) creates per-layer graphs. Leaving `--cuda-graph-modules` unset captures the whole
Transformer layer, while specifying modules restricts capture to selected sub-regions.

`--cuda-graph-granularity chunk` instead captures the complete local `TransformerBlock` for each
model chunk after PP/VPP partitioning. The pipeline schedule, loss calculation, and optimizer stay
outside the graph. Chunk granularity requires an empty `--cuda-graph-modules` list and is not
supported by the Transformer Engine backend.

During inference, `local` can attach graphs at either the layer boundary or the enclosing block
boundary, as controlled by `--inference-cuda-graph-scope`.

Operationally, this path is tightly integrated into MCore training and inference:

- graphable modules create and own their `CudaGraphManager` instances automatically
- the existing training schedules drive warmup/capture/replay automatically
- users select the mode through config flags only; there is no separate helper API to
  wire into a custom training loop or a separate need to handle static input buffers

### Usage

```bash
# Layer granularity (default)
--cuda-graph-impl local

# Full local model-chunk granularity
--cuda-graph-impl local \
--cuda-graph-granularity chunk \
--cuda-graph-dynamic-microbatches
```

### Layer `--cuda-graph-modules` options

| Module | What is captured |
|---|---|
| *(empty / not set)* | Entire Transformer layer (default) |
| `attn` | `TransformerLayer._forward_attention()` |
| `mlp` | `TransformerLayer._forward_mlp()` for dense layers |
| `moe` | `TransformerLayer._forward_mlp()` for MoE layers (drop-and-pad only) |
| `moe_router` | MoE router + shared experts (if not EP-comm-overlapped) |
| `moe_preprocess` | `MoELayer.preprocess()` — must be paired with `moe_router` |
| `mamba` | Mamba SSM layer |

**Example — MoE model, capture attention and router:**
```bash
--cuda-graph-impl local \
# Optionally restrict captured modules (default: capture whole layer, but not working for MoE dynamic shapes)
--cuda-graph-modules attn moe_router moe_preprocess
```

### Chunk granularity

Local chunk graphs derive physical activation-slot assignments from the actual 1F1B or interleaved
schedule and reuse a slot only after the corresponding backward has completed. An omitted
`--cuda-graph-num-microbatch-slots` derives the exact rank-local topology maximum, including every
legal partial VPP tail-group shape. An explicit value must be at least as large as that topology
maximum. Capture remains deferred until real execution has exercised every topology-required slot,
so a later larger microbatch schedule cannot request a graph runner that was never captured. An
explicitly overprovisioned slot count must likewise be exercised before capture and can therefore
delay capture unnecessarily.

Chunk granularity is not compatible with `--overlap-moe-expert-parallel-comm`, whose combined
schedule executes individual layers instead of calling the enclosing `TransformerBlock`, or with
fine-grained activation offloading. Static context parallelism is supported, but dynamic context
parallel groups are not because a captured chunk cannot change its NCCL process group. Use layer
granularity for these unsupported combinations.

Training chunk graphs are bypassed during validation/eval, which continues through the eager
forward-only schedule. Inference CUDA graphs remain controlled separately by
`--inference-cuda-graph-scope`.

THD varlen training also needs static graph input shapes. A typical configuration is:

```bash
--use-varlen-dataset \
--pad-packed-seq-alignment max \
--max-seqlen-per-dp-cp-rank <static-local-token-capacity> \
--cuda-graph-impl local \
--cuda-graph-granularity chunk \
--cuda-graph-dynamic-microbatches
```

`--cuda-graph-dynamic-microbatches` is required when a sequence-packing scheduler is active,
because the number of packed microbatches can change between iterations.

Capture uses static input buffers, but replay copies the current microbatch's actual THD metadata
and padding mask into them. Consequently MoE padding decisions reflect the real token/padding
distribution of each replay, including existing inter-sequence padding; the mask is not replaced
with a permanent all-true or all-false value.

With paged stash, dynamic local chunk graphs make each physical slot independent of the absolute
microbatch schedule. For PP greater than one, a slot always stashes its forward activation and
reloads that same activation in its matching backward; it does not capture schedule-dependent
neighbor prefetch/discard decisions. For PP=1, forward and backward are adjacent, so the graph
keeps the original activation and does not perform an unnecessary stash/reload. The runtime
microbatch count may therefore change without recapturing graphs. Static input shapes, expert
capacity, and stash-buffer limits still apply; an overflow or over-budget step uses the existing
eager retry path.

Paged-stash capacity profiling is also tied to physical-slot coverage. If an early iteration uses
fewer microbatches than the topology maximum, profiling remains active and accumulates capacity
maxima across later eager iterations until every required slot has been exercised. Stash buffers
are allocated only after that point. This prevents a small first iteration from under-sizing the
page pools for a later larger schedule; it can delay graph capture when the workload takes several
iterations to reach its maximum in-flight pattern.

CUDA graph memory pools are assigned per local model chunk and physical in-flight slot. There is
one runner pair per slot rather than one logical runner per absolute microbatch, so memory scales
with the in-flight slot count and static local token capacity rather than total microbatch count.
If capture runs out of memory, first reduce
`--max-seqlen-per-dp-cp-rank`, global batch size, or pipeline in-flight depth; do not set the slot
count below the schedule-required value.

Keep the standard allocator fragmentation setting enabled before process start:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

This is part of the expected H100/GB200 training environment rather than an additional chunk-graph
requirement. It does not reduce live tensor or graph-pool capacity; it allows large allocations to
use otherwise fragmented reserved memory. On pre-SM100 GPUs, follow the existing
`NCCL_GRAPH_REGISTER=0` requirement when combining expandable segments with CUDA graphs.

---

## CUDA Graph — Transformer Engine Implementation (`--cuda-graph-impl transformer_engine`)

Uses Transformer Engine's `make_graphed_callables()` path. In Megatron-LM's CLI, this has the
same training granularity as `local`: leaving `--cuda-graph-modules` unset captures the whole
Transformer layer, while specifying modules restricts capture to selected sub-regions. The main difference from
`local` is the backend implementation and feature compatibility. Unlike `local`, this path does
not support inference CUDA graphs.

Compared to `local`, this path exposes a more general and self-contained API via TE's
`make_graphed_callables()`, giving users greater flexibility and control over how CUDA graphs are
wired into custom training loops. The trade-off is that it requires more manual setup:

- the training loop must instantiate `TECudaGraphHelper`
- the training loop must call helper methods such as `create_cudagraphs()` and
  `cuda_graph_set_manual_hooks()` at the correct points

Megatron-LM's stock training loop already wires these calls in `megatron/training/training.py`,
but custom training scripts must do the same work themselves.

### Usage

```bash
--cuda-graph-impl transformer_engine \
--cuda-graph-modules attn moe_router moe_preprocess
```

The same training `--cuda-graph-modules` options apply as for `local`, and the default is likewise
whole-layer training capture when the flag is omitted.

---

## Full-Iteration Training CUDA Graph (`--cuda-graph-impl full_iteration`)

Captures the entire training iteration (excluding optimizer) as a single CUDA graph. The same
wrapper is also used for training-loop validation/eval in forward-only mode. This provides the
largest training/validation latency reduction.

This implementation does not create inference CUDA graphs. For inference, use
`--cuda-graph-impl local --inference-cuda-graph-scope layer|block`.

### Requirements

- `--no-check-for-nan-in-loss-and-grad` is required: NaN checks involve CPU-GPU synchronization
  which cannot run inside a CUDA graph.
- `--cuda-graph-modules` must be omitted (or left empty): per-module selection has no meaning
  when the entire iteration is captured as a single graph.

### Example

```bash
--cuda-graph-impl full_iteration \
--no-check-for-nan-in-loss-and-grad
```

---

## Common Configuration Examples

### Dense Model Training

All three implementations work for dense models:

```bash
# Per-layer (local)
--cuda-graph-impl local
# equivalent: --cuda-graph-impl local --cuda-graph-modules attn mlp

# Per-layer (TE)
--cuda-graph-impl transformer_engine
# equivalent: --cuda-graph-impl transformer_engine --cuda-graph-modules attn mlp

# Full-iteration
--cuda-graph-impl full_iteration \
--no-check-for-nan-in-loss-and-grad
```

### MoE Model Training

Native dropless MoE dispatch, including HybridEP with both expert capacity factors unset, has
dynamic expert shapes and cannot be captured as a complete chunk. Chunk granularity requires a
static dispatch budget through `--moe-expert-rank-capacity-factor`, optionally with paged stash,
or `--moe-expert-capacity-factor` together with `--moe-pad-expert-input-to-capacity`. The rank
capacity path preserves dropless training semantics by rerunning an over-budget step eagerly with
the rank capacity disabled. Layer granularity uses `--cuda-graph-modules` to capture only static
regions while leaving expert compute in eager mode. Example using `transformer_engine` (`local`
works the same way):

```bash
--cuda-graph-impl transformer_engine \
--cuda-graph-modules attn moe_router moe_preprocess
```

With a static HybridEP rank budget and [paged stash](paged_stash.md), the local backend can capture
the complete model chunk, including the MoE path:

```bash
--cuda-graph-impl local \
--cuda-graph-granularity chunk \
--cuda-graph-dynamic-microbatches \
--moe-token-dispatcher-type flex \
--moe-flex-dispatcher-backend hybridep \
--use-transformer-engine-op-fuser \
--moe-expert-rank-capacity-factor <float> \
--moe-paged-stash
```

The same static-buffer setup also allows full-iteration CUDA graphs to be used on MoE models:

```bash
--cuda-graph-impl full_iteration \
--no-check-for-nan-in-loss-and-grad \
--moe-token-dispatcher-type flex \
--moe-flex-dispatcher-backend hybridep \
--use-transformer-engine-op-fuser \
--moe-expert-rank-capacity-factor <float> \
--moe-paged-stash
```

---

## Additional Notes

- `--cuda-graph-warmup-steps` (default: 3) controls how many warmup steps run before CUDA graph
  capture. Setting it to 0 is not recommended: some operations rely on the first few iterations
  for lazy initialization or autotuning, and capturing too early may produce incorrect or
  suboptimal graphs. Local chunk graphs with paged stash are the exception: one or more real
  schedule iterations profile all topology-required slots, the following iteration allocates the
  stash buffers, and capture occurs after those buffers have been exercised. Synthetic per-runner
  warmups are disabled so they cannot perturb the profiled paged-stash schedule.
- Inference CUDA graphs (serving or RL rollout) currently require
  `--cuda-graph-impl local`. Use `--inference-cuda-graph-scope layer|block` with
  `local`; all other implementations must set `--inference-cuda-graph-scope none`,
  meaning inference runs in eager mode.
- Background reference: [Transformer Engine and Megatron-LM CUDA Graph Support](https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/te-megatron-cuda-graphs.html),
  which also covers PyTorch CUDA Graph best practices and lessons learned.

---

## Migration Guide

Legacy configurations (including `--enable-cuda-graph`, `--external-cuda-graph`, the renamed
`--cuda-graph-scope` flag (now `--cuda-graph-modules`), and deprecated module values such as
`full_iteration` and `full_iteration_inference`) are still accepted and automatically migrated
at runtime, but we encourage updating your configs to the new forms:

| Old command | New command |
|---|---|
| `--enable-cuda-graph` | `--cuda-graph-impl local` |
| `--external-cuda-graph` | `--cuda-graph-impl transformer_engine` |
| `--cuda-graph-scope <modules>` | `--cuda-graph-modules <modules>` |
| `--cuda-graph-impl local --cuda-graph-scope full_iteration` | `--cuda-graph-impl full_iteration` |
| `--cuda-graph-impl local --cuda-graph-scope full_iteration_inference` | `--cuda-graph-impl local --inference-cuda-graph-scope block` |
| `--cuda-graph-impl local --cuda-graph-scope attn moe_router moe_preprocess full_iteration_inference` | `--cuda-graph-impl local --cuda-graph-modules attn moe_router moe_preprocess --inference-cuda-graph-scope block` |
