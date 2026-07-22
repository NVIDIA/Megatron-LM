<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# DSv4 CP communication overlap

This note covers the ratio-4 indexer path in packed-THD context parallelism.
Boundary-token P2P exchange is separate from the two collectives described here.

## Forward

The two local compressor outputs are gathered in a fixed order on the CP group:

```text
compute: indexer-K compressor | attention-KV compressor + local Q/weight projections
comm:                          | AG(indexer K)

compute: score-buffer init | indexer score + top-k | sparse attention
comm:                       | AG(attention KV)
```

`AG(indexer K)` is launched before the attention-KV compressor and waited on
before top-k. In the fused training path, the local indexer Q and weight
projections are also placed between this launch and wait. `AG(attention KV)` is
launched after the indexer score buffer is initialized and waited on only after
top-k. This avoids overlapping NCCL with the score-buffer fill and makes the
gather eligible to overlap score computation and top-k. All ranks enqueue the
collectives in the same order.

## Backward

The fused indexer-loss forward saves the indexer Q/K/weight gradients. At
backward entry, the global indexer-K gradient is therefore ready before sparse
attention backward produces the global compressed-KV gradient.

```text
compute:        sparse-attention backward | local Q/weight projection backward
comm:    RS(indexer K)                     | RS(attention KV) -----------------> wait
                    wait -> indexer-K compressor backward       KV compressor backward
```

The fused autograd function owns both CP gradient reductions. It launches
`RS(indexer K)` first and runs sparse-attention backward before waiting. Once
the global attention-KV gradient is available, it launches `RS(attention KV)`
and returns its output through a deferred branch-local wait. That edge is
created before the independent local Q and weight projection nodes, so PyTorch
schedules those newer backward nodes before the wait. The KV compressor consumes
the reduced gradient only after the wait completes. The gathered forward
tensors are detached from the generic gather backward so each gradient is
reduced exactly once.

Both reduce-scatters use the same CP communicator and are enqueued in the same
order on every rank. The unfused and no-indexer-loss paths keep the standard
synchronous autograd mappings.

Actual kernel concurrency depends on the communication size and host dispatch
latency. If a collective finishes before the independent compute kernel is
enqueued, its latency is hidden by host work rather than GPU compute.

## Profiling ranges

The forward path emits `dsv4_cp_*_all_gather_*`, compressor, and top-k NVTX
ranges. The backward path emits these dedicated ranges:

- `dsv4_cp_indexer_k_reduce_scatter_launch`
- `dsv4_cp_sparse_attention_backward`
- `dsv4_cp_attention_kv_reduce_scatter_launch`
- `dsv4_cp_local_indexer_grads`
- `dsv4_cp_attention_kv_reduce_scatter_consumer_wait`

The ranges are enabled by Megatron's `--nvtx-ranges` profiler option.
