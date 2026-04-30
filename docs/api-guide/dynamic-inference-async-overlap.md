<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Dynamic Inference Async Overlap

Dynamic inference uses a queue-depth-limited pipeline to overlap GPU forward and
sampling work with CPU-owned bookkeeping and ordered output retirement.
`async_overlap_queue_depth=2` is the default validated mode. Queue depth one uses
the same architecture and remains the debug and correctness fallback.

## Invariants

- Every dynamic step has a monotonic step id, a step journal entry, and an owned
  snapshot slot before the forward launches.
- The scheduler updates CPU-resident request state first, then transfers one
  coalesced metadata snapshot to a fixed-address GPU snapshot slot.
- Output retirement is ordered by step id, not by launch completion order.
- Lookahead may be disabled for a step, but the engine does not switch to a
  second serial implementation.
- Distributed ranks must agree on prepare metadata before launching a forward.
  Repeated divergence downgrades the live pipeline to queue depth one and records
  the reason in metrics.

## Observability

Inference metrics include the active queue depth, queue-depth distribution,
maximum in-flight launches observed, snapshot-slot waits, output-event waits,
lookahead-token discards, rollback counts, graph-cache hit/miss counts by
snapshot slot, host-observed inter-step gap, and CPU time retired while another
launch remains in flight.

## Tuning

Depth-two mode allocates one spare snapshot slot beyond the launch queue depth so
retiring snapshots and CUDA graph capture reuse do not force avoidable host-side
stalls. The pinned output-copy pool also keeps one spare slot beyond the queue
depth, allowing public output retirement to lag the next launch without reusing a
buffer that still has an outstanding D2H event.
