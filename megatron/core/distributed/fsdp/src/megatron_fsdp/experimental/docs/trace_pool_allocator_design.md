# M-FSDP v2 trace-pool allocator

## Motivation

The experimental M-FSDP v2 path normally releases full-parameter and
reduce-scatter input storage after each use. Repeated `torch.empty` and Storage
resize operations are logically memory-efficient, but they exercise the CUDA
caching allocator from the all-gather and reduce-scatter streams throughout a
step. Large, differently sized buffers can leave many cached segments behind,
so `memory_reserved()` may stay tens of GiB above `memory_allocated()`.

`--fsdp-trace-pool` replaces that steady-state allocation pattern with fixed
physical slots. It is an opt-in experiment; the normal allocator remains the
default.

## Requirements

The pool must preserve three properties of the existing implementation:

1. Autograd may save a transposed or otherwise non-leaf view of an unsharded
   weight during forward. Releasing and restoring the weight must keep its
   `Storage` object alive so that the saved view sees the backward re-gather.
2. All-gather and reduce-scatter run asynchronously on different CUDA streams.
   Host-side lifetime non-overlap alone is not sufficient to share memory
   across those streams.
3. VPP + combined 1F1B uses `FsdpExecutionRunner` to trace occurrence order,
   prefetch the next unit, and skip redundant reshards. Storage planning must
   not replace or bypass that execution path.

## Lifecycle

### Trace: schedule-learning batch(s)

Each logical buffer has a stable key derived from its parameter group and role:

- BF16 full model weight;
- MXFP8 row-wise full payload;
- MXFP8 column-wise full payload;
- partial-gradient reduce-scatter input.

During the trace phase, a key owns one tensor `Storage`. `free()` resizes that
Storage to zero, and the next `allocate()` restores the same Storage object.
This deliberately does not share physical storage between keys during the
trace phase: doing so before future lifetimes are known could invalidate a
weight view saved by GEMM for backward.

Allocation and free events are recorded in the actual VPP/1F1B occurrence
order. `FsdpExecutionRunner` continues to record its independent prefetch trace
at the same time.

Without execution replay, storage planning can follow the first global batch.
With `FsdpExecutionRunner`, the first batch learns execution order with
prefetch disabled. Prefetch becomes active only during replay and intentionally
extends some allocation lifetimes. Storage tracing therefore spans both the
initial execution-trace batch and one complete prefetch-enabled replay batch.
Planning from only the first batch would under-estimate overlap and cause a
safe slot-collision failure when replay begins.

### Plan: boundary after the observed replay

After the optimizer step, `complete_fsdp_trace()` reaches the shared
`FsdpContext`. The context first completes execution-runner tracing. It defers
storage planning while the runner is tracing, and ignores duplicate VPP chunk
notifications with no intervening execution events. Planning occurs once at
the boundary after the first complete replay and requires no live logical
allocations.

The allocator:

1. waits once for outstanding work on the local CUDA device;
2. converts the trace into one or more live intervals per logical key;
3. builds an interval-conflict graph;
4. greedily colors that graph, largest buffers first with best-fit slot reuse;
5. reuses trace `Storage` objects as the fixed physical slots; and
6. calls `torch.cuda.empty_cache()` once, after the live pool has claimed its
   storage, to discard surplus trace fragmentation.

The coloring is a memory-oriented heuristic, not a proof of the minimum
weighted coloring.

### Optimized: later global batches

Every key maps to one fixed tensor view. `allocate()` and `free()` only update
logical ownership and do not call the CUDA allocator or resize Storage. A slot
collision raises an error, making a schedule that diverges from the traced
lifetimes visible instead of silently corrupting memory.

An unobserved late key receives a dedicated slot and emits a warning. It is not
allowed to share with traced keys.

## Stream arenas

Slots are partitioned by `(dtype, device, arena)`. The current arenas are:

- `allgather` for BF16/MXFP8 unshard buffers;
- `reduce_scatter` for partial-gradient buffers.

Keys may share only within one arena. Operations that reuse a slot are then
ordered by the same CUDA stream. Cross-stream sharing is intentionally
forbidden even when Python allocation intervals do not overlap.

## DBuffer integration

`DBuffer.bind_local_buffer()` lets an existing distributed layout adopt an
allocator-owned flat tensor without reconstructing the layout. Cached tensor
views are invalidated after rebinding.

For regular parameters, persistent `nn.Parameter` objects are rebound to fresh
DBuffer views after a pool allocation. In the trace phase, forward and backward
allocations for a key share the key's Storage object. After planning, all later
allocations for that key share its fixed slot. MXFP8 payload setters similarly
refresh row-wise and column-wise tensor views after their DBuffers are rebound.

Partial-gradient buffers use `DBuffer.from_local()` so the pooled tensor is the
collective input directly; no extra `torch.empty` is created.

## Scope and limitations

- The feature pools M-FSDP unshard and partial-gradient communication buffers.
  Persistent sharded weights, main gradients, optimizer state, activations, and
  MXFP8 quantization temporaries remain under their existing allocators.
- NCCL user buffers and PyTorch symmetric-memory pools require separately
  registered storage, so they cannot be combined with `--fsdp-trace-pool`.
- The traced execution must be repeatable. Runtime slot-collision checks cover
  changed overlap; late-key warnings cover newly observed allocations.
- Planning introduces one synchronization and cache trim after the observed
  trace/replay phase. They are excluded from steady-state benchmark statistics.

## Validation

Correctness tests cover Storage identity across trace release/reallocation,
slot sharing and collisions, arena isolation, DBuffer rebinding, and multi-step
loss parity across the trace-to-optimized transition. Performance validation
should compare identical 24-GPU PP3/VPP2/EP8 jobs, discard at least the initial
execution trace, first replay, and first five steps, and report allocated,
reserved, and device-used memory on one rank from every pipeline stage.
