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

CUDA graph behavior is set by three orthogonal flags:

| Flag | Values | Purpose |
|---|---|---|
| `--cuda-graph-impl` | `none` / `local` / `transformer_engine` / `full_iteration` | Which capture backend or strategy to use |
| `--cuda-graph-modules` | `attn` / `mlp` / `moe` / `moe_router` / `moe_preprocess` / `mamba` | Per-layer **training** capture coverage; multi-valued and only meaningful for `local` and `transformer_engine` |
| `--inference-cuda-graph-scope` | `none` / `layer` / `block` | Granularity of CUDA graphs during **inference**; only `local` supports non-`none` values |

Supported combinations:

| `--cuda-graph-impl` | Backend | Training capture | Inference capture |
|---|---|---|---|
| `none` | — | off | off |
| `local` | MCore `CudaGraphManager` | per-layer, controlled by `--cuda-graph-modules` | `layer` (default) or `block`, controlled by `--inference-cuda-graph-scope` |
| `transformer_engine` | TE `make_graphed_callables()` | per-layer, controlled by `--cuda-graph-modules` | not supported (`none` only) |
| `full_iteration` | MCore `FullCudaGraphWrapper` | one graph per training iteration; `--cuda-graph-modules` must be empty | not supported (`none` only) |

---

## CUDA Graph — Local Implementation (`--cuda-graph-impl local`)

Uses MCore's built-in `CudaGraphManager`. During training, this is a per-layer mode:
leaving `--cuda-graph-modules` unset captures the whole Transformer layer, while specifying
modules restricts capture to selected sub-regions. During inference, `local` can instead attach
graphs at either the layer boundary or the enclosing block boundary, as controlled by
`--inference-cuda-graph-scope`.

Operationally, this path is tightly integrated into MCore training and inference:

- graphable modules create and own their `CudaGraphManager` instances automatically
- the existing training schedules drive warmup/capture/replay automatically
- users select the mode through config flags only; there is no separate helper API to
  wire into a custom training loop or a separate need to handle static input buffers

### Usage

```bash
--cuda-graph-impl local
```

### `--cuda-graph-modules` options

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

### mHC Attention Split

For mHC selective recompute, `--mhc-recompute-attn-cuda-graph-split` keeps mHC aggregation and
BDA eager and captures only the input norm and attention. The producer writes directly into
the graph's single-stream `[s, b, C]` input, including during backward recomputation, instead
of capturing the whole mHC attention range with an `[s, b, n*C]` input.

This supports GPT mHC layers and HybridStack attention-only mHC wrappers. Hybrid MLP/MoE
and Mamba layers remain eager under the required `attn`-only scope. On MLA models,
`mla_up_proj` recompute can be combined with `mhc`; its checkpoint stays inside the
attention graph:

```bash
--enable-hyper-connections \
--recompute-granularity selective \
--recompute-modules mhc mla_up_proj \
--cuda-graph-impl transformer_engine \
--cuda-graph-modules attn \
--mhc-recompute-attn-cuda-graph-split
```

Omit `mla_up_proj` for non-MLA models or when only mHC recompute is needed. Packed (THD)
sequences are supported with the ordinary TE graph packing configuration: a
`sequence_packing_scheduler`, `max_seqlen_per_dp_cp_rank`, `thd_max_packed_sequences`, and
fixed padding via `pad_packed_seq_alignment=max`. The split forwards the packed sequence
tensors and padding mask on every replay; eager MLP/MoE still receives the original metadata.

The split does not support cross-attention or fine-grained offloading of `qkv_linear`,
`core_attn`, and `attn_proj`. Hybrid wrappers must not combine attention and MLP in the same
inner layer. Graph capacities and the attention backend's existing THD/CP constraints still apply.

### Chunk granularity (`--cuda-graph-granularity chunk`)

`--cuda-graph-granularity chunk` changes the callable handed to `make_graphed_callables()` from
each transformer layer to the whole decoder block of every PP/VPP model chunk: one forward graph
and one backward graph per model chunk and microbatch slot. Activation recompute (including
`--recompute-granularity full`), MoE dispatch/combine and the hyper-connection residual streams are
recorded inside the graph, so a training step launches one graph per chunk and pass instead of one
per layer. On the last pipeline stage of packed-sequence (THD) runs the post-process (MTP block,
LM head and loss) is captured as a second callable of that chunk, in the same schedule order, so it
shares the graph memory pool with the decoder graphs. With `--optimizer-cuda-graph` the optimizer
step graph is captured into that same pool.

```bash
--cuda-graph-impl transformer_engine \
--cuda-graph-granularity chunk \
--cuda-graph-warmup-steps 2 \
# packed sequences: static shapes and a capture bound on the packed microbatch count
--pad-packed-seq-alignment max --thd-max-packed-sequences <N> --cuda-graph-dynamic-microbatches \
# optional: optimizer step in the same graph memory pool (optimizer state must stay on the GPU)
--optimizer-cuda-graph
```

Requirements: `--cuda-graph-modules` must be empty; MoE layers need static token shapes (drop-padding
MoE, the HybridEP flex dispatcher with `--moe-expert-rank-capacity-factor`, or a MoE megakernel
backend); `--overlap-moe-expert-parallel-comm` and `--delay-wgrad-compute` are not supported.
`--fine-grained-activation-offloading` runs in a whole-block capture mode: the D2H/H2D copies are captured into each
slot's graphs with static pinned host buffers and no cross-slot prefetch, so any number of live slots can replay. Because a
slot's graphs are self-contained, the copies that eager overlaps with the *next* microbatch (the last groups' D2H at the end
of a forward, the first group's reload at the start of a backward) cannot be hidden the same way, so the copies cost more
step time than in eager. `--fine-grained-offloading-graph-keep-last-group` keeps the last group of every offload module
resident per slot to avoid the synchronous reload at the price of part of the memory saving (DSv4 proxy, five offload
modules, chunk graphs without offload as the reference: 2.35 s/step and -5.3 GB without the flag, 1.55 s/step and -1.7 GB
with it).
`--dsa-cp-balance-indexer` works in its graph-dynamic mode: the per-pack route pair becomes two
static inputs of the block graph (refreshed per replay like `cu_seqlens`). `--moe-paged-stash`
works in a runtime-keyed mode: each MoE layer stashes its own activations after its forward and
reloads them right before its own backward (no cross-layer prefetch; GPU pages only by default),
so the captured graphs do not depend on the recorded pipeline order; it needs TE >= 2.19 and
`--cuda-graph-warmup-steps >= 2`. Full recompute inside the captured block requires
`hidden_dropout=0`, `attention_dropout=0` and no router input jitter: the recompute runs inside the
backward graph, where the RNG state cannot be rewound (forced-load-balancing router logits are
replayed from their recorded seed and are therefore allowed).

**DSA compact-indexer workspace.** With CUDA graphs the fused compact DSA indexer (`--dsa-kernel-backend cudnn`,
`--dsa-indexer-precision mxfp8`) keeps a persistent workspace per static geometry: the MXFP8 q/k quantization
destinations, packed scales and candidate offsets have to be prepared in eager warm-up because their sizing
synchronises with the host. By default all CSA layers built from one model config share one workspace per geometry
and balanced-indexer slot (`dsa_compact_indexer_workspace_sharing`); the layers of a graph run serially and nothing in
the workspace outlives the indexer dispatch that fills it, so sharing removes only duplicates (one workspace per layer
costs ~7 GiB per rank on DSv4 at 16K tokens per rank). `--no-dsa-compact-indexer-workspace-sharing` restores
per-layer workspaces.

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

MoE expert dispatch involves dynamic shapes and cannot be captured. `--cuda-graph-modules` is used
to capture only the static parts (attention, router, preprocess) while leaving expert compute in
eager mode. Example using `transformer_engine` (`local` works the same way):

```bash
--cuda-graph-impl transformer_engine \
--cuda-graph-modules attn moe_router moe_preprocess
```

With [paged stash](paged_stash.md), expert dispatch shapes become static (pre-sized via
`--moe-expert-rank-capacity-factor`), which allows full-iteration CUDA graphs to be used on MoE
models as well:

```bash
--cuda-graph-impl full_iteration \
--no-check-for-nan-in-loss-and-grad \
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
  suboptimal graphs.
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
