<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# pipeline_parallel package

This package contains implementations for two different pipeline parallelism
schedules (one without interleaving and one with interleaving, see [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)
for details), and a default no-pipelining schedule. It also contains methods
for the point-to-point communication that is needed between pipeline stages.


## Multi-stream residuals (mHC)

Both the ordinary and interleaved pipeline schedules support
`TransformerConfig.enable_mhc_connections`. Embeddings expand the residual at
`pre_process`; intermediate physical and virtual stages carry all
`mhc_num_residual_streams` streams. The stage containing the final output
contracts the residual before the loss. GPT retains its mean contraction and
HybridModel retains its learned contraction; checkpoint parameter names do not
change.

The common pipeline-width calculation sizes fixed P2P buffers as
`hidden_size * mhc_num_residual_streams` when mHC and PP are enabled. This applies
to **every communicating edge**, including the last physical rank sending to the
first rank's next virtual chunk. Suppressed sends/receives at the true model
boundaries need no single-stream buffer. Sequence parallelism and context
parallelism continue to divide the sequence axis, while variable sequence lengths
continue to exchange actual shapes through the existing P2P protocol.
At PP2 the previous and next ranks coincide; batched communication orders forward
messages before backward messages so that VPP's simultaneous activation and
gradient exchanges remain distinct.

Empty intermediate Hybrid stages return an independent, graph-connected output.
This allows the schedule to pseudo-deallocate a sent output without resizing its
input or losing the gradient path. Standalone GPT embedding/loss stages and
explicit Hybrid pipeline patterns, including empty segments, retain their normal
boundary behavior. GPT's embedding expansion returns a viewless tensor so that
an embedding-only stage can safely release its sent output too.

Eager selective mHC recomputation can be used with PP/VPP. Hybrid MTP must be on
the final `post_process` chunk, where both the multi-stream decoder result and
its learned single-stream contraction are available. Standalone mHC MTP stages
and `overlap_moe_expert_parallel_comm` with mHC pipeline parallelism are rejected.
CUDA graph, offload, and full-recompute compatibility are separate from this
pipeline support and retain their existing restrictions.

`tests/unit_tests/pipeline_parallel/test_pp_mhc_compatibility.py` compares actual
forward/backward schedules, losses, and all finalized parameter gradients with
an identically initialized PP1 model. It covers PP2, PP2/VPP2, standalone GPT
embedding/loss stages, empty Hybrid stages with output deallocation, Hybrid MTP
with TP/SP, and CP with changing sequence lengths. The test selects the existing
mHC GPT layer through a custom spec; enabling mHC does not add a new default GPT
factory path. BF16 CP cases use native FP32 gradient accumulation and the local
loss-sum/token-count contract for CP-correct normalization.
