{/*
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
*/}

# Context Parallel Overview

![Diagram of a transformer layer with tensor parallelism 2 and context parallelism 2, showing CP and TP communication patterns around attention and other blocks.](../../images/context_parallel/CP_overview.png)

Figure 1: A transformer layer running with TP2CP2. Communications next to Attention are for CP, others are for TP. (AG/RS: all-gather in forward and reduce-scatter in backward, RS/AG: reduce-scatter in forward and all-gather in backward, /AG: no-op in forward and all-gather in backward).

Context Parallelism (CP) is a parallelization scheme on the sequence-length dimension. Unlike prior SP (sequence parallelism), which only splits the sequence of Dropout and LayerNorm activations, CP partitions the network inputs and all activations along the sequence dimension. With CP, all modules except attention (for example, Linear and LayerNorm) can work as usual without any changes, because they do not have inter-token operations. For attention, the Q (query) of each token must combine with the KV (key and value) of all tokens in the same sequence. CP therefore requires an additional all-gather across GPUs to collect the full sequence of KV. Correspondingly, reduce-scatter is applied to the activation gradients of KV in backward propagation. To reduce activation memory footprint, each GPU stores only the KV of a sequence chunk in forward and gathers KV again in backward. KV communication happens between a GPU and its counterparts in other TP groups. The all-gather and reduce-scatter are implemented as point-to-point communications in a ring topology. Exchanging KV can also leverage MQA or GQA to reduce communication volume, because those variants use one or a few attention heads for KV.

For example, in Figure 1, if the sequence length is 8K, each GPU processes 4K tokens. GPU0 and GPU2 form a CP group and exchange KV with each other; the same pattern applies between GPU1 and GPU3. CP is similar to [Ring Attention](https://arxiv.org/abs/2310.01889) but targets higher performance by (1) using current open-source and cuDNN flash attention kernels, and (2) avoiding extra work from lower-triangle causal masking while keeping load balanced across GPUs.

## Context Parallelism Benefits

![Chart of speedup for 175B GPT with different tensor parallelism and context parallelism combinations compared with full activation recomputation.](../../images/context_parallel/CP_results.png)

Figure 2: Speedup of 175B GPT with various TP+CP combinations compared to full recomputation (that is, TP8CP1).

An LLM can hit an out-of-memory (OOM) error on long contexts (long sequence lengths) because activation memory grows about linearly with sequence length. Recomputing activations in backward can avoid OOM but adds significant overhead (about 30 percent with full recomputation). Increasing TP (tensor model parallelism) can also fix OOM, but it can make compute in layers such as Linear too short to hide communication latency. Scaling to more GPUs with larger TP can hit that overlap limit even when OOM is not the driver.

CP addresses these tradeoffs. With CP, each GPU computes on part of the sequence, which scales down both compute and communication by the CP degree. Overlap between them is less of a concern. The activation memory footprint per GPU is also smaller by the CP degree, which reduces OOM risk. As Figure 2 shows, TP and CP together can outperform full recomputation by removing most recompute overhead and balancing compute against communication.

## Enabling Context Parallelism

CP support is included on the GPT code path. Other models that share that path, such as LLaMA, can use CP as well. CP works with TP (tensor model parallelism), PP (pipeline model parallelism), and DP (data parallelism). The total GPU count is TP × CP × PP × DP. CP also works with different attention variants, including MHA, MQA, and GQA, with unidirectional or bidirectional masking.

Enable CP by setting `context_parallel_size=<CP_SIZE>` on the command line. The default `context_parallel_size` is 1, which disables CP. Running with CP requires Megatron Core (>=0.5.0) and Transformer Engine (>=1.1).

## Dynamic context parallelism

Dynamic CP assigns a CP group to each packed microbatch according to its sequence
lengths. Short sequences can use CP1 while longer sequences share the same DP×CP
domain at CP2, CP4, and other enabled sizes. The model's build-time group remains
unchanged. Enable it with `--dynamic-context-parallel --use-varlen-dataset
--sequence-packing-scheduler default_dynamic_cp --calculate-per-token-loss`, and
set `--max-seqlen-per-dp-cp-rank` to the token capacity of one rank.
`--min-dynamic-context-parallel-size` sets the smallest group the scheduler may
select (default 1). The largest selectable group spans the DP×CP domain; there is
currently no independent maximum-size option.

GPT and Hybrid models propagate the selected groups through packed microbatch
metadata. Runtime CP is supported by standard attention, MLA and absorbed MLA,
DSA with the MLA path, Mamba, GDN/GDN2, GDP, hybrid layout conversion, and MTP.
Sequence parallelism also carries the matching TP×CP group. CSA and DSA-GQA do
not support dynamic CP; DeepSeek-V4 is not supported.

The following constraints apply:

- Use per-token loss and variable-length packed training data. Dynamic CP does
  not support deterministic mode, CUDA graphs, or router z-loss.
- Headwise Mamba, GDP, and GDN/GDN2 require their TP-local head/group geometry
  to divide **every** CP size the scheduler can select. Construction validates
  this against the caller's DP×CP group, before a long sequence reaches forward.
- Mamba does not support chunkwise CP. Chunkwise GDP keeps all TP-local heads,
  avoids the head-sharding divisibility restriction, and is recommended when
  dynamic CP would select more ranks than the head geometry supports.
- Inference uses CP1. These runtime training groups do not enable CP inference.

MTP with `--calculate-per-token-loss` sums the main and MTP valid-token counts
over the effective CP group before scaling gradients. This also corrects static
CP>1 runs whose ranks have different valid-token counts; it adds a two-element
all-reduce per MTP depth per microbatch. Logged `mtp_N loss` is a global
token-weighted mean, including under static CP, so packed or masked runs may log
different values from older mean-of-microbatch-means metrics.

## Layout conversion contract

`megatron.core.context_parallel.convert_cp_partition_mode` accepts concrete
`source_partition_mode` and `target_partition_mode` values (`zigzag` or
`contiguous`). It calls the same implementation as `convert_cp_layout`;
there is no separate partition-mode communication backend. `seq_dim` selects
the tensor's sequence axis. With sequence parallelism, callers must supply the
matching TP and TP×CP groups.

For packed tensors, prepare a `ContextParallelBatch` through
`get_batches_on_this_cp_rank`, request both physical layouts, and pass its
`thd_plan` to conversion. The plan supports different physical token counts on
the two sides and combines TP×CP redistribution in one all-to-all. The consumer
must use `batch.get_packed_seq_params(target_layout)` as well as the converted
tensor. Changing only `PackedSeqParams.cp_partition_mode` is incorrect when
zigzag attention adds padding. Logical boundaries and physical padded
boundaries remain distinct; empty documents are allowed within a nonempty pack.

Batch views annotate metadata with its concrete layout. Wrapping metadata
whose layout is unspecified makes a shallow metadata copy, without rebuilding
its precomputed tensors or modifying the caller's object. Contradictory layout
annotations are rejected. The annotation is MCore-only and is not passed to
Transformer Engine.

The low-level `get_batch_on_this_cp_rank` dispatcher accepts
`cp_partition_mode` as a spelling of its existing contiguous/zigzag selection.
It validates conflicts with `use_contiguous_cp`; it does not merely annotate
the returned tokens. `auto` is a conversion policy, not a physical ordering,
and is rejected by these tensor and batch APIs.

### Model policies and module boundaries

`TransformerConfig.cp_partition_mode` (also `--cp-partition-mode`) selects the
residual-stream policy. The default `None` retains the existing main behavior;
`auto` explicitly selects that policy. Hybrid stacks coalesce runs of equal
consumer layouts. `zigzag` and `contiguous` instead keep a fixed residual layout:
only the self-attention or mixer branch converts to its consumer layout, then
converts back **before** BDA, residual writes, and MLP/MoE execution. These policies
share the same canonical conversion implementation.

`linear_cp_layout` and `attention_cp_layout` still describe consumer requirements,
not the residual policy. For example, chunkwise linear attention still requires
`linear_cp_layout="contiguous"`. The module adapter receives the actual consumer
layout selected by the Hybrid layer config, rather than inferring every layer's
requirements from one model-wide attention-variant flag. The lower-level manager's
`boundary_mode="fixed"` remains a layer-boundary primitive and is not used to
implement this attention-only policy.

Both pretraining entry points and the Hybrid packing scheduler prepare the
required batch views. Fixed-layout producers call
`ContextParallelBatch.prepare_module_layouts()` once after metadata finalization.
This attaches an acyclic `CPLayoutViews` context: its leaf metadata has no back
reference, tensors and the plan are reused, and module calls do not copy metadata
or build routes. Legacy/auto metadata keeps `cp_layout_views=None`. RoPE preparation
uses the largest physical sequence length among the prepared views.

Bridge callers that already have packed metadata can call
`finalize_packed_seq_params` from `megatron.core.context_parallel.module` at batch
preparation and use its returned metadata. Both physical padding and logical
boundaries must describe the actual input. Already-sharded zigzag data requiring
additional padding is rejected: prepare it from the raw batch instead. The old
`megatron.core.context_parallel_layout` import path provides thin exports, not a
second route builder. Its old equal-length `ThdCpRoute` constructor is not retained;
callers migrate to `THDCPLayoutPlan` and the returned target metadata.

Metadata-only Hybrid callers reuse the contiguous leaf when preparing packed
chunkwise recurrence metadata, even if the residual input is zigzag with different
physical padding. MTP resolves the source layout from the actual batch or tagged
metadata, falling back to the model boundary policy. When MTP needs a different
layout, callers must supply the full `ContextParallelBatch` with token, label and
mask views; a metadata-only adapter does not provide these tensors.

Two-stage execution carries its reverse converter with each intermediate result,
so interleaved calls cannot overwrite one another's layout state. Runtime CP1
does not convert and requires no route. TP×CP groups are resolved from the same
runtime metadata as the CP group. Hybrid MTP keeps the existing explicit conversion
to its rolling layout, including token/mask views and the matching plan. GPT MTP
currently requires a fixed residual layout matching `attention_cp_layout`; use
HybridModel for MTP across different layouts.

GDN1's FLA backend supports chunkwise CP with contiguous compute inputs, retaining
all TP-local heads and passing the same FLA CP context to convolution and recurrence.
Static SBHD contexts are cached by runtime group and shape; THD recurrence contexts
are shared through the producer-owned metadata within a microbatch. Chunkwise CP
currently requires THD or SBHD micro-batch size one, unfused pre-GDR, no independent
local convolution padding, and eager execution. Torch/TE backends retain headwise CP;
GDN2 also retains headwise-only CP. This does not claim deterministic FLA replay.

### Explicit boundaries

- A changed module layout currently requires implicit attention masking, no dense
  attention bias, and no cross-attention. CP module conversion is training-only.
- New explicit policies that change layout reject mHC/wide cross-layer residual
  state and custom layers without a matching adapter. Existing unset-policy paths
  are retained. Fixed layouts requiring no conversion are unaffected.
- Module conversion with THD views is not CUDA-graph replayable. SBHD conversion
  also rejects graphs containing attention/mixer or the full iteration; MLP-only
  graph scopes leave the SBHD converter outside capture.
- Auto layout conversion can remain outside layer-local graphs. A THD plan inside
  a full-iteration graph is rejected because native all-to-all fixes split sizes
  during capture. Accepting a metadata dataclass does not establish replay support.
