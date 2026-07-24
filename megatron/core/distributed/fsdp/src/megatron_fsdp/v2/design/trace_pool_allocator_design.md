# TracePoolAllocator design

`TracePoolAllocator` is the CUDA-graph-friendly bucket allocator used by
Megatron FSDP v2 for temporary parameter and gradient communication buffers.
It lives in [`allocator.py`](../allocator.py).

The allocator has one job: after a short trace, each logical allocation key
must resolve to a stable tensor address while still allowing non-overlapping
keys to share memory.

## Why Megatron FSDP v2 needs it

FSDP repeatedly allocates temporary buffers for operations such as:

```text
forward pre-hook:   all-gather parameter shard -> full parameter buffer
post-forward:       reshard / release full parameter buffer
backward hook:      all-gather parameter shard for backward
post-backward:      reduce / reduce-scatter gradient buffer
```

The default storage-freeing allocator is fine for eager training, but CUDA
graphs require stable addresses. If a graph captures kernels that read a
parameter buffer at address `A`, replay must use address `A` again. Reallocating
or resizing the backing tensor after capture is not allowed.

`TracePoolAllocator` solves this by tracing one micro-batch, building a static
key-to-slot plan, and returning the same precomputed view for each key in
later micro-batches.

`fully_shard()` selects `TracePoolAllocator` when either `enable_trace_pool` or
`enable_cuda_graph` is enabled. It also selects trace-pool allocation for a root
module that contains child FSDP modules with CUDA graph enabled.

## Terms

| Term | Meaning |
| --- | --- |
| Key | Caller-provided allocation identifier, typically `(param_group_id, role)`. |
| Interval | Lifetime of one key between a traced `allocate()` and matching `free()`. |
| Slot | Physical backing tensor that can be shared by keys whose intervals never overlap. |
| View | `slot.tensor[:size_for_key]`, cached in `_key_to_view[key]` for O(1) optimized allocation. |

There is no monolithic pool tensor and no slot offset. Each slot is its own
`torch.empty()` tensor. This lets the CUDA caching allocator place slots
independently instead of requiring one large contiguous block.

## Lifecycle

### 1. Trace phase

The allocator starts in phase `"trace"`.

`_trace_allocate()` records an alloc event with a monotonic sequence number,
stores the first observed `(size, dtype, device)` metadata for the key, and
returns a normal temporary bucket. If the key is already active, allocation is
idempotent and returns the existing bucket.

`_trace_free()` records a free event, releases the bucket storage, and removes
the key from `_active_keys`. Double-free is treated as a no-op.

The trace phase records lifetime information only. It does not require later
micro-batches to replay the same sequence positions.

### 2. Plan phase

`plan()` consumes the trace and switches the allocator to phase `"optimized"`.

It performs these steps:

1. Pair traced alloc/free events into per-key intervals.
2. Assign a sentinel end sequence to allocations that were still live when
   tracing ended.
3. Group keys by `(dtype, device)`.
4. For each group, build a conflict graph:
   - keys are graph nodes;
   - an edge means at least one interval for key A overlaps at least one
     interval for key B;
   - keys without an edge are allowed to share a slot.
5. Greedily color the graph:
   - visit keys largest-size-first;
   - choose the existing non-conflicting slot with the smallest waste;
   - otherwise create a new slot.
6. Allocate one `torch.empty(slot_size, dtype, device)` tensor per slot.
7. Populate `_key_to_slot[key]` and `_key_to_view[key]`.

The coloring is heuristic, not a proof of globally optimal memory usage. It is
simple, deterministic enough for the traced graph on each rank, and captures
the property FSDP needs: overlapping lifetimes do not share the same backing
tensor.

Important implementation detail: intervals are not merged into one wide
`first_alloc -> last_free` range. `_intervals_overlap()` compares all intervals
for the two keys, so two keys can still share a slot when their lifetimes have
multiple non-overlapping windows.

### 3. Optimized phase

In phase `"optimized"`, allocation is a dictionary lookup plus safety checks:

```python
if key not in _key_to_slot:
    _add_optimized_key(key, size, dtype, device)

slot = _slots[_key_to_slot[key]]
if slot.in_use and key not in _active_keys:
    raise RuntimeError("slot collision")
assert size <= slot.size

slot.in_use = True
_active_keys.add(key)
return Bucket(data=_key_to_view[key])
```

The slot-collision check catches cases where the optimized lifetime pattern no
longer matches the traced pattern: a different key tries to use a slot that is
still active.

If the same key is allocated again while active, the call is idempotent and
returns the same cached view.

`free(key)` clears `slot.in_use` and removes the key from `_active_keys`.
Freeing a key that is not active is a no-op.

### 4. Late keys

The trace is expected to observe normal FSDP temporary keys. Some paths are
control-flow dependent, for example last-microbatch-only HSDP gradient sync.
For these cases, `_optimized_allocate()` supports a key first seen after
`plan()` by creating a dedicated slot through `_add_optimized_key()`.

This preserves allocator correctness for subsequent uses and for
`release()`/`resume()`, but it is not a substitute for retracing an allocation
pattern that must be part of an already captured CUDA graph. If graph-captured
work starts using a new key, the graph owner must drop and recapture the graph.

## Release, resume, and reset

`reset()` is a full teardown:

- phase becomes `"trace"`;
- trace events and metadata are cleared;
- active buckets are cleared;
- slots, key-to-slot mapping, and key-to-view mapping are cleared.

`release()` is a memory-pressure path:

- in phase `"trace"`, it behaves like `reset()`;
- in phase `"optimized"`, it frees each slot tensor's storage but preserves
  slot metadata and key mappings;
- phase becomes `"released"`;
- the next `allocate()` or `free()` automatically calls `_auto_resume()`,
  reallocates slot tensors, rebuilds key views, and returns to `"optimized"`.

Because resumed tensors may have different addresses, CUDA graphs captured
against the old slot tensors must not be reused. `FSDPModule.release_memory_pool()`
handles this by releasing installed CUDA graphs and clearing graph sentinels
before calling `TracePoolAllocator.release()`.

## CUDA graph interaction

`TracePoolAllocator` does not capture CUDA graphs itself. It only guarantees
stable tensor views for the FSDP buffers that graph capture may observe.

The relevant flow is:

1. First micro-batch runs in trace phase.
2. The post-backward final callback calls `bucket_allocator.plan()`.
3. The next optimized forward records sample inputs/outputs for CUDA graph
   capture when `enable_cuda_graph=True`.
4. The post-backward final callback calls `CudaGraphRunner.capture_and_install()`.
5. `CudaGraphRunner` uses capture-time hooks to perform FSDP unshard/reshard
   around capture while installing graphed module forwards.

The allocator invariant required by this flow is:

```text
same key + same planned slot + same cached view => stable address
```

If `release_memory_pool()` is called, the installed graphs are cleared so the
next forward can recapture against the newly allocated slot tensors.

## Debugging

`dump_trace()` prints:

- current phase;
- trace events with key metadata;
- slot count and slot metadata in `"optimized"` or `"released"` phase;
- slot address when memory is live, or `<released>` after `release()`;
- keys assigned to each slot;
- total live pool bytes.

Typical issues:

| Symptom | Likely cause |
| --- | --- |
| `TracePoolAllocator slot collision` | Optimized allocation lifetimes differ from traced lifetimes; two keys planned to share a slot are live at the same time. |
| `requested size > slot capacity` | A key requested a larger buffer after planning than it requested during trace. |
| CUDA graph replay uses stale data or invalid addresses after memory release | Graphs were not dropped before allocator `release()` / auto-resume. Use `FSDPModule.release_memory_pool()` rather than calling allocator `release()` directly. |

## Design properties

| Property | Mechanism |
| --- | --- |
| Stable address per planned key | `plan()` allocates one backing tensor per slot and caches `_key_to_view[key]`. |
| Memory reuse | Conflict-graph coloring lets non-overlapping keys share a slot. |
| Runtime simplicity | Optimized allocation is a key lookup, collision check, size check, and cached-view return. |
| Fragmentation control | Per-slot tensors avoid one large contiguous pool allocation. |
| Recovery from memory pressure | `release()` preserves the plan while dropping storage; auto-resume rebuilds slot tensors and views. |

## Non-goals

- No cross-dtype or cross-device slot sharing.
- No attempt to prove globally optimal graph coloring.
- No allocator-side CUDA graph ownership; graph invalidation and recapture are
  handled by FSDP module/runtime code.
- No automatic retracing when an optimized allocation pattern changes. Slot
  collisions and size-capacity assertions fail fast instead.
