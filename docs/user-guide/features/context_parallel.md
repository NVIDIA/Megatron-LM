<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Context Parallel Overview

```{figure} ../../images/context_parallel/CP_overview.png
:alt: Diagram of a transformer layer with tensor parallelism 2 and context parallelism 2, showing CP and TP communication patterns around attention and other blocks.
:align: center

Figure 1: A transformer layer running with TP2CP2. Communications next to Attention are for CP, others are for TP. (AG/RS: all-gather in forward and reduce-scatter in backward, RS/AG: reduce-scatter in forward and all-gather in backward, /AG: no-op in forward and all-gather in backward).
```

Context Parallelism (CP) is a parallelization scheme on the sequence-length dimension. Unlike prior SP (sequence parallelism), which only splits the sequence of Dropout and LayerNorm activations, CP partitions the network inputs and all activations along the sequence dimension. With CP, all modules except attention (for example, Linear and LayerNorm) can work as usual without any changes, because they do not have inter-token operations. For attention, the Q (query) of each token must combine with the KV (key and value) of all tokens in the same sequence. CP therefore requires an additional all-gather across GPUs to collect the full sequence of KV. Correspondingly, reduce-scatter is applied to the activation gradients of KV in backward propagation. To reduce activation memory footprint, each GPU stores only the KV of a sequence chunk in forward and gathers KV again in backward. KV communication happens between a GPU and its counterparts in other TP groups. The all-gather and reduce-scatter are implemented as point-to-point communications in a ring topology. Exchanging KV can also leverage MQA or GQA to reduce communication volume, because those variants use one or a few attention heads for KV.

For example, in Figure 1, if the sequence length is 8K, each GPU processes 4K tokens. GPU0 and GPU2 form a CP group and exchange KV with each other; the same pattern applies between GPU1 and GPU3. CP is similar to [Ring Attention](https://arxiv.org/abs/2310.01889) but targets higher performance by (1) using current open-source and cuDNN flash attention kernels, and (2) avoiding extra work from lower-triangle causal masking while keeping load balanced across GPUs.

## Context Parallelism Benefits

```{figure} ../../images/context_parallel/CP_results.png
:alt: Chart of speedup for 175B GPT with different tensor parallelism and context parallelism combinations compared with full activation recomputation.
:align: center

Figure 2: Speedup of 175B GPT with various TP+CP combinations compared to full recomputation (that is, TP8CP1).
```

An LLM can hit an out-of-memory (OOM) error on long contexts (long sequence lengths) because activation memory grows about linearly with sequence length. Recomputing activations in backward can avoid OOM but adds significant overhead (about 30 percent with full recomputation). Increasing TP (tensor model parallelism) can also fix OOM, but it can make compute in layers such as Linear too short to hide communication latency. Scaling to more GPUs with larger TP can hit that overlap limit even when OOM is not the driver.

CP addresses these tradeoffs. With CP, each GPU computes on part of the sequence, which scales down both compute and communication by the CP degree. Overlap between them is less of a concern. The activation memory footprint per GPU is also smaller by the CP degree, which reduces OOM risk. As Figure 2 shows, TP and CP together can outperform full recomputation by removing most recompute overhead and balancing compute against communication.

## Enabling Context Parallelism

CP support is included on the GPT code path. Other models that share that path, such as LLaMA, can use CP as well. CP works with TP (tensor model parallelism), PP (pipeline model parallelism), and DP (data parallelism). The total GPU count is TP × CP × PP × DP. CP also works with different attention variants, including MHA, MQA, and GQA, with unidirectional or bidirectional masking.

Enable CP by setting `context_parallel_size=<CP_SIZE>` on the command line. The default `context_parallel_size` is 1, which disables CP. Running with CP requires Megatron Core (>=0.5.0) and Transformer Engine (>=1.1).


## Physical layouts and conversion ownership

`megatron.core.context_parallel` owns layout conversion, route preparation and batch
partitioning. `megatron.core.context_parallel_layout` retains compatibility exports
for its public APIs; new code should use `context_parallel`.

The `zigzag` layout assigns two chunks of each sequence to a CP rank to balance
causal attention. The `contiguous` layout assigns a contiguous interval of the
packed token stream. `cp_partition_mode` describes the module-boundary layout;
`linear_cp_layout` and `attention_cp_layout` describe Hybrid layer preferences.
These settings have different roles and are not aliases.

Both Hybrid's cross-layer layout manager and module-boundary adapters use the
same SBHD redistribution primitive. With sequence parallelism it communicates
directly over TP×CP; without sequence parallelism it uses the CP group. Module
adapters also handle alternate sequence dimensions and restore the caller's layout.

THD plans are prepared with the batch that owns their physical token ordering:

- Managed Hybrid batches prepare layout-specific metadata and a `THDCPLayoutPlan`
  in `get_batches_on_this_cp_rank`. The plan may connect different local token counts
  when zigzag attention requires extra padding. Forward reuses that plan without
  also building equal-length module routes for those metadata views.
- Sequence-packing scheduler batches have one physical layout. Hybrid batch
  preparation finalizes their `PackedSeqParams.cp_partition_route` for module-local
  conversion before entering the model. GPT prepares a module route only when its
  boundary layout requires conversion to zigzag attention. Default GPT inter-document
  masking partitions whole samples; it does not require per-document routes or document
  lengths divisible by `2 * CP`. Contiguous CP with this sample-level masking path is
  currently rejected, even when document lengths happen to be aligned.

The two THD contracts remain distinct: module routes require packed sequence
lengths divisible by `2 * CP` and preserve local token count, while managed plans
can insert zigzag padding. Module THD conversion with sequence parallelism retains
its TP gather, CP all-to-all, TP scatter path. Batch metadata and route geometry
must match the active process group and the actual physical layout.

Scheduler `[T]` tensors and ordinary `[B, T]` tensors share CP row selection with an
explicit sequence dimension. Their adapters retain padding policy and mask handling;
in particular, dense attention-mask query rows remain zigzag. Per-document zigzag
selection uses a shared Transformer Engine wrapper after the adapter has prepared
aligned sequence boundaries.

For split attention execution, the pre/core stage returns its tensor directly when
no layout conversion is needed. Otherwise it pairs that tensor with the return-layout
converter for that call. Pass this result unchanged into `forward_post_core_attn`:
the output projection, offload, and recomputation bookkeeping run before the CP
conversion restores the input layout. No converter is stored on the module, so
interleaved calls keep independent layout state. Ordinary `forward` still returns
`(output, bias)`.

Runtime CP metadata follows the Dynamic-CP group contract: when `local_cp_size` is
set, `cp_group` must have that size, including a singleton group for CP=1. An explicit
`cp_group` also takes precedence when `local_cp_size` is omitted. TE temporarily binds
the resolved group for each forward and restores its previous group on exit; singleton
metadata disables CP communication in TE without changing the shared model groups.

## MTP and mixer layout contracts

MTP keeps scheduler and GPT tensors in their boundary layout. Its rolling operations
use that physical layout for tokens, positions, labels, loss masks and precomputed
embeddings. Contiguous rolling exchanges a single boundary element with adjacent CP
ranks, then clears document ends and padding. Its backward exchanges gradients in the
opposite direction. Sequence-parallel precomputed embeddings gather before rolling and
scatter afterward; local HSM rolling instead clears the local shard seam. Managed Hybrid
MTP still uses its prepared attention-layout batch for both inputs and loss computation.

When a packing scheduler disables Hybrid's cross-layer layout manager, Mamba and GDP
convert inputs to `linear_cp_layout` and restore the caller's layout after their output
projection. Both ordinary and split execution follow this contract; split calls keep
independent return converters and packed metadata. MLA and AbsorbedMLA require zigzag
at their actual module input for both SBHD and THD. They raise before RoPE or attention
if an unsupported contiguous shard reaches them; a manager-converted zigzag input is valid.

GDN and GDN2 currently convert to zigzag and execute headwise CP. GDP's chunkwise CP is
a separate implementation. This main-branch port does not provide a GDN chunkwise kernel
that directly consumes contiguous shards, and does not claim a two-thirds reduction in
GDN layout conversions or a measured Qwen throughput improvement.
