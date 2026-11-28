<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software or related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# CUDA Graph

CUDA Graphs reduce kernel-launch overhead by recording GPU operations once and replaying the recording on subsequent iterations. Megatron-LM provides four CUDA graph implementations controlled by `--cuda-graph-impl`.

For implementation background and design details, see NVIDIA's
[Transformer Engine and Megatron-LM CUDA Graph Support](https://docs.nvidia.com/dl-cuda-graph/latest/torch-cuda-graph/te-megatron-cuda-graphs.html).
That article is a useful conceptual reference, but some examples there still use older flags such as
`--enable-cuda-graph` or `--cuda-graph-scope full_iteration`; in this repository, prefer
`--cuda-graph-impl local|transformer_engine|chunk_wise|full_iteration` as documented below.

## Overview

CUDA graph behavior is set by three orthogonal flags:

| Flag | Values | Purpose |
|---|---|---|
| `--cuda-graph-impl` | `none` / `local` / `transformer_engine` / `chunk_wise` / `full_iteration` | Which capture backend or strategy to use |
| `--cuda-graph-modules` | `attn` / `mlp` / `moe` / `moe_router` / `moe_preprocess` / `mamba` | Per-layer **training** capture coverage; multi-valued and only meaningful for `local` and `transformer_engine` |
| `--inference-cuda-graph-scope` | `none` / `layer` / `block` | Granularity of CUDA graphs during **inference**; only `local` supports non-`none` values |

Supported combinations:

| `--cuda-graph-impl` | Backend | Training capture | Inference capture |
|---|---|---|---|
| `none` | — | off | off |
| `local` | MCore `CudaGraphManager` | per-layer, controlled by `--cuda-graph-modules` | `layer` (default) or `block`, controlled by `--inference-cuda-graph-scope` |
| `transformer_engine` | TE `make_graphed_callables()` | per-layer, controlled by `--cuda-graph-modules` | not supported (`none` only) |
| `chunk_wise` | TE `make_graphed_callables()` | one forward/backward graph pair per PP/VPP model chunk and microbatch slot (plus one for the post-process of the last stage with packed sequences); `--cuda-graph-modules` must be empty | not supported (`none` only) |
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

For scheduler-produced THD batches, use `--use-varlen-dataset`,
`--sequence-packing-scheduler dp_balanced`, `--cuda-graph-modules attn`,
`--pad-packed-seq-alignment max`, and an explicit `--cuda-graph-thd-max-packed-sequences`. The metadata capacity includes one
reserved dummy document for token padding. `--max-seqlen-per-dp-cp-rank` sets the
fixed local token capacity; the source global batch size and micro-batch size 1
must stay fixed. PP/VPP automatically keeps graph inputs alive until their
corresponding backward even when packing reduces the number of microbatches.
Source-batch ramp-up is unsupported: `--step-batch-size-schedule`
is rejected for this path, and the deprecated `--rampup-batch-size` is ignored by
the microbatch calculator. Increasing the source batch would require recapturing
the graphs with a new capacity. Dynamic CP, EP communication overlap, delayed
wgrad, and paged stash are outside this path. mHC wrappers and attention graph
splitting require their separate graph integrations; these packing options do
not enable mHC capture by themselves.

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

**DSA compact-indexer workspace.** With CUDA graphs the fused compact DSA indexer (`--dsa-kernel-backend cudnn`,
`--dsa-indexer-precision mxfp8`) keeps a persistent workspace per static geometry: the MXFP8 q/k quantization
destinations, packed scales and candidate offsets have to be prepared in eager warm-up because their sizing
synchronises with the host. By default all CSA layers built from one model config share one workspace per geometry
and balanced-indexer slot (`dsa_compact_indexer_workspace_sharing`); the layers of a graph run serially and nothing in
the workspace outlives the indexer dispatch that fills it, so sharing removes only duplicates (one workspace per layer
costs ~7 GiB per rank on DSv4 at 16K tokens per rank). `--no-dsa-compact-indexer-workspace-sharing` restores
per-layer workspaces.

---

## Chunk-wise CUDA Graph (`--cuda-graph-impl chunk_wise`)

`--cuda-graph-impl chunk_wise` keeps Transformer Engine's `make_graphed_callables()` but hands it the
whole decoder block of every PP/VPP model chunk instead of the individual layers: one forward graph
and one backward graph per decoder block and microbatch slot. Activation recompute (including
`--recompute-granularity full`), MoE dispatch/combine and the hyper-connection residual streams are
recorded inside the graph, so a training step launches one graph per chunk and pass instead of one
per layer. `--cuda-graph-modules` must be empty and inference CUDA graphs are not supported.

With packed sequences (`--sequence-packing-scheduler`) the post-process of the last pipeline stage
(MTP block, LM head and loss) is captured as a second callable of that chunk, in the same schedule
order, so it shares the graph memory pool with the decoder graphs. The post-process stays eager
(it runs after the decoder replay) for fixed-length batches, when the stage owns no decoder layer,
and when Python has to see the logits of every microbatch: forward hooks on the LM head (logits
savers, distillation captures) or, for `GPTModel`, `--config-logger-dir` (its post-process dumps
the inputs and logits of every forward). A call that conditions the post-process on
anything but the captured tensors (no labels, an MTP input mask, precomputed MTP embeddings, a
per-call `runtime_gather_output`, an output processor, extra MTP block arguments, a sequence length
offset, a disabled MTP loss or a context-parallel layout plan) takes the eager post-process as well.
When the post-process of a packed run is not captured, a warning on its pipeline stage names the
reason.

### Usage

```bash
--cuda-graph-impl chunk_wise \
--cuda-graph-warmup-steps 2 \
# packed sequences: static shapes and a capture bound on the packed microbatch count
--micro-batch-size 1 --pad-packed-seq-alignment max --thd-max-packed-sequences <N> \
--cuda-graph-dynamic-microbatches \
# packed sequences with PP > 1: the padded inter-stage shape is static, skip the P2P shape exchange
--pipeline-p2p-fixed-shape \
# optional: optimizer step in the same graph memory pool (optimizer state must stay on the GPU)
--optimizer-cuda-graph
```

With `--optimizer-cuda-graph` the optimizer step graph is captured into the pool of the block graphs
when `cuda_graph_use_single_mempool` is on (the default) and the installed Transformer Engine's
`make_graphed_callables()` takes a `pool` argument; otherwise both keep their own pools. The
optimizer's first step always runs eagerly, so its state exists before the graph is captured.

### Requirements

- MoE layers need static token shapes: drop-padding MoE (`--moe-expert-capacity-factor` with
  `--moe-pad-expert-input-to-capacity`), the HybridEP flex dispatcher with
  `--moe-expert-rank-capacity-factor`, or a MoE megakernel backend. HybridEP's runtime padding
  (`--moe-hybridep-pad-variable-tokens`) is turned off: every rank holds the same static token count.
  MoK's MXFP8 mode records its expert scale re-swizzle inside the captured block, so every replay
  picks up the scales of the latest parameter update.
- Packed sequences need `--micro-batch-size 1`, `--cuda-graph-dynamic-microbatches`,
  `--pad-packed-seq-alignment max` (or the value of `--max-seqlen-per-dp-cp-rank`) and a bound on
  the sequences per pack (`--thd-max-packed-sequences`). Packed data paths without the
  packing scheduler (`--sft`, `--dataloader-inter-document-masking`) are not captured: the training
  CLI rejects them with `chunk_wise`, and in a custom training loop a block graph captured from
  fixed-length batches raises when it receives packed sequence parameters.
- Fixed-length batches that carry a padding mask (for example `--varlen-sbhd-validation`) need
  `--cuda-graph-warmup-steps >= 1` when the block has MoE layers: the capture reserves the mask
  input only if the block received a mask during the eager warm-up, and a mask that reaches a
  graph captured without one raises.
- Not supported with `chunk_wise`: `--overlap-moe-expert-parallel-comm`, `--delay-wgrad-compute`,
  dynamic context parallelism (the captured block records the context-parallel communicator of the
  capture sample), and hybrid stacks that convert between context-parallel layouts across their
  layers.
- Not supported together with packed sequences: `--position-embedding-type mrope` (the packing
  scheduler carries `[1, tokens]` position ids while mRoPE needs the `[3, batch, seq]` layout),
  GDN / KDA / GDP layers (selected by `--experimental-attention-variant`, by the `G` / `K` symbols of
  a hybrid layer pattern, or by a layer spec that puts the gated delta product mixer into its Mamba
  layers: the FLA kernels derive their chunk indices on the host from the `cu_seqlens` values and
  cache them per tensor, so a captured block graph would replay the pack layout of the capture
  sample) and Mamba layers with chunkwise context parallelism (their CP metadata is built on the
  host per pack). mRoPE and the GDN / KDA variants and pattern symbols are rejected when the
  configuration is built, the other two when the block is first captured. These restrictions
  apply to packed sequences only.
- With packed sequences `--defer-embedding-wgrad-compute` is turned off: the LM head is part of the
  captured post-process, so its weight gradient is computed inside the graph.
- A replay runs no Python inside the captured blocks. Forward tensor metrics that observe them
  (residual and router sources; logits sources too with packed sequences) are rejected, and forward
  hooks on modules inside a captured block do not fire at replay.

### Precision and position embeddings

A graph replay runs under the block-wide FP8 autocast the eager block forward opens (the replay does
not run the forward), so delayed scaling keeps reducing its amax and updating its scales, and the FP8
weight cache keeps following `is_first_microbatch`. NVFP4 is captured like FP8.
`--first-last-layers-bf16` layers opt out of the block-wide capture autocast and MTP layers opt in
(`get_layer_fp8_context`, `get_layer_fp4_context`); a post-process without quantized modules (no MTP on
the stage or `--keep-mtp-in-bf16`, and no MXFP8 output projection) is captured and replayed without an
autocast, like its eager counterpart.

`--position-embedding-type rope`, `yarn` and (fixed-length) `mrope`, or MLA's own rotary handling,
can be captured: the block's sample inputs carry a rotary table of the runtime shape, which the
runtime table overwrites at replay.

### Activation recompute

The re-entrant checkpoint sites (full recompute, the MTP block, `mlp` / `moe` / `shared_experts`, the
GDN / GDN2 / KDA core `gdn` and `core_attn`) use Transformer Engine's checkpoint for every precision
whenever `cuda_graph_impl` is
`chunk_wise`, including the eager warm-up iterations: Megatron's `tensor_parallel.checkpoint` runs the
function without a checkpoint node while a graph is warmed up or captured and would silently keep the
activations resident. Recompute built on `CheckpointWithoutOutput` (`layernorm`, `moe_act`,
`mla_up_proj`, `gdn_norm_out`, `mhc`) keeps its own hook mechanism, which registers the recompute
inside the capture as is.

A recompute inside the captured block cannot reproduce a random draw of the forward: it runs
inside the backward graph, where the RNG state cannot be rewound. Full recompute therefore requires
`hidden_dropout=0`, `attention_dropout=0` and no router input jitter; selective recompute requires
`attention_dropout=0` for `core_attn`, `hidden_dropout=0` for `mlp` / `moe` and no router input
jitter for `moe`; hyper-connection (`mhc`) recompute requires both dropouts to be 0.
`--moe-router-force-load-balancing` and `--moe-router-force-biased` are not rejected, but their
random router logits are drawn again when the router is recomputed (full recompute, or selective
recompute of `moe`), so that combination is for throughput measurements only, not for numerical
comparisons (the code warns).
`--distribute-saved-activations` does not shard the input of a block's first checkpoint, which is the
graph's static input buffer.

### Fine-grained activation offloading

`--fine-grained-activation-offloading` runs in a whole-block capture mode: the D2H/H2D copies are
captured into each slot's graphs, with host buffers no other slot reuses and no cross-slot prefetch,
so any number of live slots can replay. The captured callables' static inputs (the block's
`hidden_states`, which the first layer's `attn_norm` commit would otherwise free) are kept out of
the offload release, and the group hooks are no-ops outside a capture-time forward, so a full
activation recompute captured into the backward graph neither offloads nor reloads (a fully
recomputed block saves no activations to offload in the first place, so offloading only pays for
layers that are not recomputed). With pipeline parallelism the combination is limited by the eager
offload schedule, which the warm-up iterations run: it works with two pipeline stages when at least
one layer of every stage is kept out of the recompute (`--recompute-method block` with
`--recompute-num-layers` below the stage's layer count). With every layer of a stage recomputed
(uniform full recompute, or the block method over all layers), or with three or more pipeline
stages, that schedule stops on a chunk-mismatch assertion, with or without CUDA graphs.
Because a slot's graphs are self-contained, the copies that eager
overlaps with the *next* microbatch (the last groups' D2H at the end of a forward, the first group's
reload at the start of a backward) cannot be hidden the same way, so the copies cost more step time
than in eager. `--fine-grained-offloading-graph-keep-last-group` keeps the last group of every
offload module resident per slot to avoid the synchronous reload at the price of part of the memory
saving (DSv4 proxy, five offload modules, chunk graphs without offload as the reference: 2.35 s/step
and -5.3 GB without the flag, 1.55 s/step and -1.7 GB with it). Without packed sequences the
post-process (MTP, LM head, loss) runs eagerly after the decoder replay and keeps its activations on
the GPU: the captured graphs own every offload of the step. Every slot inherits the offload policy
recorded during the eager warm-up (which groups stay resident, including the groups kept for
`delta_offload_bytes_across_pp_ranks`), so that budget applies per live slot rather than once per
schedule. It needs `--cuda-graph-warmup-steps >= 1`; deleting the graphs returns the offload to the
eager policy and releases the pinned host buffers of the deleted slots (a fixed pinned buffer,
`--fine-grained-offloading-buffer-size-gib`, keeps the range the capture reserved for its
lifetime, so eager offloads after a deletion use the remaining tail).

### Paged stash and DSA

`--moe-paged-stash` works in a runtime-keyed mode: each MoE layer stashes its own activations after
its forward and reloads them right before its own backward (no cross-layer prefetch; GPU pages only
by default), at any pipeline size, so the captured graphs do not depend on the recorded pipeline
order; it needs TE >= 2.19 and `--cuda-graph-warmup-steps >= 2`. A MoE layer that is recomputed in
the backward (full recompute, or selective recompute of `moe`) is stashed only when another MoE layer
of the same recompute segment follows it; the last one of a segment is consumed by that backward at
once and stays in place, as in the eager schedule. Only stashed layers count towards the size of the
page buffers. With one MoE layer per segment and every MoE layer recomputed, the flag therefore has
no effect: no page buffers are allocated and a warning says so. As at larger pipeline sizes, a stash
that overflows its page buffers after the capture stops the run (raise
`--moe-paged-stash-buffer-size-factor-cuda`), and so does a rank whose routed tokens exceed
`--moe-expert-rank-capacity-factor`: a captured graph cannot repeat the step without the budget as
the eager schedule does. Without pipeline parallelism the stash saves what the capacity factor pads:
in a 16-layer MXFP8 MoE proxy (8 experts, top-4, 4096 tokens per rank, PP=1) it lowered the peak
allocation by 9 % at a capacity factor of 2.0 and by 0.3 % at 1.25, at the same step time.

`--dsa-cp-balance-indexer` with `--sequence-packing-scheduler dp_balanced` works in its graph-dynamic
mode: the per-pack route pair becomes two static inputs of the block graph (refreshed per replay
like `cu_seqlens`).

---

## Full-Iteration Training CUDA Graph (`--cuda-graph-impl full_iteration`)

Captures the entire training iteration (excluding optimizer) as a single CUDA graph. The same
wrapper is also used for training-loop validation/eval in forward-only mode. This provides the
largest training/validation latency reduction.

With an eager optimizer, full-iteration training captures model-to-optimizer gradient conversion
in the forward/backward graph's memory pool. The distributed optimizer and the layer-wise
mixed-precision optimizer can reuse completed forward/backward scratch for FP32 gradients.
Replay order must remain forward/backward, gradient conversion, then optimizer update.
The converted values are valid until the next forward/backward replay; gradient addresses must
remain stable. Resetting the training graph also invalidates the conversion graph.

This conversion graph is disabled with `--optimizer-cuda-graph`, which captures the optimizer
step itself. CPU-offloaded and Megatron-FSDP optimizer paths are unchanged, as is the distributed
precision-aware optimizer path that already aliases model gradients.

This implementation does not create inference CUDA graphs. For inference, use
`--cuda-graph-impl local --inference-cuda-graph-scope layer|block`.

### Requirements

- Loss and gradient checks remain enabled. Values passed to the rerun state machine during
  capture are snapshotted at their validation sites. Host-side checks run after each replay,
  before the training loop decides whether to rerun or update the optimizer.
- `--cuda-graph-modules` must be omitted (or left empty): per-module selection has no meaning
  when the entire iteration is captured as a single graph.

### Example

```bash
--cuda-graph-impl full_iteration
```

---

## Common Configuration Examples

### Dense Model Training

All four implementations work for dense models:

```bash
# Per-layer (local)
--cuda-graph-impl local
# equivalent: --cuda-graph-impl local --cuda-graph-modules attn mlp

# Per-layer (TE)
--cuda-graph-impl transformer_engine
# equivalent: --cuda-graph-impl transformer_engine --cuda-graph-modules attn mlp

# Whole decoder block per pipeline stage (TE)
--cuda-graph-impl chunk_wise

# Full-iteration
--cuda-graph-impl full_iteration
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
- With `--optimizer-cuda-graph` the optimizer's first step always runs eagerly (it allocates the
  optimizer state): the step graph is captured at iteration `max(--cuda-graph-warmup-steps, 1)`.
- Inference CUDA graphs (serving or RL rollout) currently require
  `--cuda-graph-impl local`. Use `--inference-cuda-graph-scope layer|block` with
  `local`; all other implementations must set `--inference-cuda-graph-scope none`,
  meaning inference runs in eager mode.
- Background reference: [Transformer Engine and Megatron-LM CUDA Graph Support](https://docs.nvidia.com/dl-cuda-graph/latest/torch-cuda-graph/te-megatron-cuda-graphs.html),
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
