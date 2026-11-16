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
