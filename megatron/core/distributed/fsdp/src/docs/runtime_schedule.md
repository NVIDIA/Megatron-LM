<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# MFSDP Runtime Schedule

The sub-design proposes the MFSDP runtime schedule for overlapping, double buffering, and
prefetching.

# Proposed Schedule

## Forward

![MFSDP forward runtime schedule][mfsdp-forward-schedule]

## Backward

![MFSDP backward runtime schedule][mfsdp-backward-schedule]

## Legend

- AO/DO: allocation/deallocation of allgather output, typically a NCCL user buffer
- AI/DI: allocation/deallocation of reduce scatter input, typically a NCCL user buffer
- CI/CO: copy into/out of communication input
- Arrows: inter-stream dependency
- GA: gradient accumulation
- \<action\>\_i: \<action\> for the i-th layer. Note backprop goes in reverse direction, so
  i+1 happens before i.

## Prefetching

During forward propagation, AG\_{i+1} needs to be prefetched before DO\_i. Otherwise,
AG\_{i+1} and Fwd\_i wouldn’t overlap. Without prefetching, AG\_{i+1} (triggered by layer
i+1) would be enqueued after DO\_i (triggered by layer i), which happens after Fwd\_i.

Similarly, during backprop, AG\_{i-1} needs to be prefetched before DO\_i and RS\_i.
Otherwise, AG\_{i-1} and Bwd\_i wouldn’t overlap. When AG and RS share a NCCL communicator,
the communicator enforces the host-side launch order even if the operations use independent
CUDA streams[^1]. Without prefetching, AG\_{i-1} would therefore happen after RS\_i, which
happens after Bwd\_i. An independent communicator for AG removes this ordering constraint,
but prefetching is still needed to initiate AG early enough to overlap with Bwd\_i.

[PR #5719](https://github.com/NVIDIA/Megatron-LM/pull/5719) prefetches the next
`FsdpModule` from the static `nn.Module.modules()` traversal order. This naturally implements
double buffering and can be extended to prefetch more aggressively.

This is simplest to implement but has several limitations that can be partially addressed
with a record-and-replay mechanism.

1. Static order may differ from runtime order. For example, `parent_module.forward()` may
   call its submodules in a different order from `parent_module.__init__`. It might also run
   a submodule multiple times.
2. Even runtime order might not be deterministic run-to-run. This will break CUDA graph
   capture as well.
3. Different input sizes may require different runtime orders for efficiency. We could
   capture or compute multiple orders for the runtime to choose from.

# Alternatives considered

## FSDP1

FSDP1 uses `record_stream` to avoid deallocating allgather output too early. This has caused
[non-determinism](https://dev-discuss.pytorch.org/t/fsdp-cudacachingallocator-an-outsider-newb-perspective/1486)
in memory usage. Therefore, FSDP2
[has avoided using `record_stream`](https://github.com/pytorch/pytorch/issues/114299).
Instead, it adds extra stream synchronization so allocation and deallocation of a buffer
happen on the same stream.

## FSDP2

Due to per-parameter sharding, FSDP2 issues more data copy than MFSDP. This simplifies the
implementation: for example, we don’t have to prefetch AG i+1 (or delay DO i) to overlap
AG i+1 and Fwd i.

In addition, backprop and copy-in run on different streams, disabling their fusion.

### Forward

![PyTorch FSDP2 forward runtime schedule][pytorch-fsdp2-forward-schedule]

FSDP2 enqueues copy-in kernels to a separate stream from allgather. We omitted that in the
figure.

### Backward

![PyTorch FSDP2 backward runtime schedule][pytorch-fsdp2-backward-schedule]

## Single Communication Stream

PyTorch’s CUDA caching allocator maintains memory pools on a per-stream basis. Consolidating
communication onto a single stream, rather than using separate streams for AllGather and
ReduceScatter, can reduce allocator fragmentation by allowing allocations to be reused from
the same stream-local pool.

Using a single communication stream may also improve determinism by imposing a consistent
ordering on communication operations that could otherwise make independent progress.

A potential drawback is the introduction of artificial dependencies between AllGather (AG)
and ReduceScatter (RS) operations. When AG and RS use separate NCCL communicators and CUDA
streams, they can make independent progress and potentially overlap. Anecdotal measurements
on some workloads have shown performance gains from this overlap, so consolidating
communication onto one stream may trade away performance for lower allocator fragmentation.
The trade-off is workload- and system-dependent and should be measured.

## Prefetching via Delayed Execution

[PR #5124](https://github.com/NVIDIA/Megatron-LM/pull/5124) implements prefetching via
delayed execution. During forward propagation, `DO_i` is delayed until after `AG_{i+1}`,
equivalently before `AO_{i+2}`. During backpropagation, `DO_i` and `RS_i` are delayed until
after `AG_{i-1}`. The delayed actions are placed into queues.

The main advantage of this approach is that it does not require predicting the next
`FsdpModule`, making it applicable even when the module execution order is dynamic or
difficult to determine ahead of time. The downside is that the implementation is considerably
more complex. In particular, the post-backward logic is spread across multiple locations
because delayed operations must be executed through callbacks, making the control flow harder
to follow and maintain. For now, we have decided not to pursue this design.

### Double buffering

By default, each all-gather (`AG`) drains its queue to a target length of one, effectively
providing double buffering. The remaining delayed operations are flushed during the
post-forward and post-backward hooks of the root `FsdpModule`.

If needed, the target queue length—or alternatively, a maximum memory budget for delayed
operations—can be configured on a per-`FsdpModule` basis. This allows prefetching to be tuned
more or less aggressively, trading off memory consumption against communication/computation
overlap.

[^1]: The [`async_op=True`](scripts/nccl_same_pg_two_streams_async.py) and
    [`async_op=False`](scripts/nccl_same_pg_two_streams_sync.py) profiling
    scripts demonstrate this. In both cases, the kernels run in sequence.

[mfsdp-forward-schedule]: images/runtime_schedule/mfsdp_forward_schedule.png

[mfsdp-backward-schedule]: images/runtime_schedule/mfsdp_backward_schedule.png

[pytorch-fsdp2-forward-schedule]: images/runtime_schedule/pytorch_fsdp2_forward_schedule.png

[pytorch-fsdp2-backward-schedule]: images/runtime_schedule/pytorch_fsdp2_backward_schedule.png
