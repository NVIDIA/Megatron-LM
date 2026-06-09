# TracePoolAllocator — v3 design (static key→slot plan)

## Background: Megatron FSDP v2 buffer allocation

Megatron FSDP v2 shards model parameters across GPUs. Before a layer computes
forward/backward, it must **all-gather** (unshard) the parameters — collecting
shards from all ranks into a temporary full-sized buffer. After compute, it
**reshards** — freeing that temporary buffer. The same pattern applies to
gradient reduction.

These temporary buffers are managed by a `BucketAllocator` that provides
`allocate(key, size, dtype, device) → Bucket` and `free(key)`. Each FSDP
parameter group gets its own key — e.g. `(pg_id, "model_weight")` for
parameter all-gather and `(pg_id, "main_grad")` for gradient reduction.

```
Forward:   allocate(model_weight)  →  [compute]  →  free(model_weight)
Backward:  allocate(main_grad)     →  [reduce]   →  free(main_grad)
```

## Why CUDA graph needs stable addresses

CUDA graphs capture a sequence of GPU kernel launches into a single replayable
object. During capture, PyTorch records every kernel launch and the exact GPU
memory addresses those kernels read/write. During replay, the graph replays
those exact operations — it cannot handle the addresses changing.

So if a layer's forward reads from address `0x7f00`, that same layer's replay
must also read from `0x7f00`. Every buffer that a CUDA graph touches must have
a **fixed, deterministic memory address** across all micro-batches.

## Why the current TracePoolAllocator breaks

The current allocator uses a **seq-driven replay schedule**:

1. **Trace** (micro-batch 0): Record every alloc/free call as `(seq, op, key)`.
2. **Plan** (`plan()`): Build intervals, color them, produce a `seq → (op, key, slot)` schedule (`_seq_ops`).
3. **Replay** (subsequent micro-batches): Walk `_seq_ops` linearly — each
   `allocate`/`free` call must arrive at the exact seq position in the exact
   order the trace produced.

**The failure**: When CUDA graph is introduced, the seq-driven replay cannot
adapt. Hook calls during capture warmup, capture, and replay may arrive in
different orders or at different relative positions from the trace. The
`_seq` counter drifts from `_seq_ops`, and future `allocate`/`free` calls
hit `RuntimeError` because the seq-driven schedule expects a different call
at the current position.

**The root cause**: The allocator assumes a fixed, repeatable alloc/free call
order and wraps it into a position-indexed schedule (`_seq_ops`). CUDA graph
capture and replay break this assumption — calls can be reordered, omitted,
or duplicated relative to the trace.

## Design goal

> Keep `TracePoolAllocator` as a single class. Replace the seq-driven replay
> schedule (`_seq_ops`) with a **static key→slot plan** (`_key_to_slot`).
> Each key maps to exactly one slot, giving one fixed memory address —
> derived once from the trace.  Per-key intervals are merged into a
> single time range during ``plan()`` so every key gets exactly one slot.

## Approach: `plan()` builds a static key→slot mapping

Instead of using the trace as a replay script, use it as **input data for
memory planning**. The trace tells us which allocations overlap in time.
Per-key intervals are merged into a single time range, then colored with
greedy left-edge. Runtime is just a dict lookup:

```python
def allocate(key, ...):
    slot = _key_to_slot[key]                     # dict lookup, O(1)
    return pool[slot.offset : slot.offset + size]  # always same address
```

No `_seq` counter. No `_seq_ops` schedule. No dependency on call order.
Per-key cursors reset to 0 between micro-batches, guaranteeing forward-pass
address stability.

## What stays, what changes, what's removed

| Stays unchanged | Changes | Removed |
|---|---|---|
| Trace phase (`_trace_allocate`, `_trace_free`, `_TraceEvent`) | `plan()` builds `_key_to_slot` instead of `_seq_ops` | `_seq`, `_seq_ops` |
| Interval construction from alloc/free pairs | `allocate()` / `free()` dispatch to key→slot lookup instead of seq walk | `_pool_allocate`, `_pool_free` |
| Per-(dtype, device) pool tensors | `enable_flexible_mode` / `disable_flexible_mode` not needed | `_flexible`, `_flex_key_to_slot` |
| `Bucket` dataclass, `BucketAllocator` interface | `_phase` transitions: `"trace"` → `"optimized"` (no intermediate `"plan"` phase held) | `snapshot_slots`, `restore_slots` |
| `_key_to_slot: Dict[alloc_key, slot_idx]` | | `reset_cursor()` |

### Why flexible mode is removed

In the current design, `enable_flexible_mode` / `disable_flexible_mode` provide
key→slot lookup for auxiliary allocations between micro-batches (e.g. weight
quantisation). In v3, **every** `allocate()` is already a key→slot lookup — the
flexible-mode path is the **default** path. A separate toggle is unnecessary.

Between micro-batches, the allocator is idle. An auxiliary `allocate(quant_key)` will
find the slot free → works exactly as before.

## `plan()` — the core method

```python
def plan(self) -> int:
    """Build a static key→slot plan from the recorded trace.

    1. Replay trace events to pair alloc↔free into intervals.
    2. Group intervals by (dtype, device), color each group with
       greedy left-edge algorithm.
    3. Allocate one flat pool tensor per group.
    4. Build _key_to_slot: every alloc_key → exactly one slot_idx.
    """
```

### The coloring algorithm

Per-key intervals are merged into a single time range before coloring,
so each key appears exactly once — plain left-edge, no same-key
enforcement needed:

```python
def _color_group(intervals, dtype, device) -> int:
    sorted_intervals = sorted(intervals, key=lambda iv: iv.alloc_seq)

    free_slots = []           # (local_slot_idx, free_seq)
    group_slots = []          # Slot objects for this group
    local_to_global = {}      # local → global slot index

    for iv in sorted_intervals:
        # ── left-edge: reuse an existing free slot, or create new ──
        assigned_local = None
        for local_idx, slot_free_seq in free_slots:
            if slot_free_seq < iv.alloc_seq:
                slot = group_slots[local_idx]
                if iv.size > slot.size:
                    slot.size = iv.size
                free_slots[...] = (local_idx, iv.free_seq)
                assigned_local = local_idx
                break

        if assigned_local is None:
            assigned_local = len(group_slots)
            global_idx = len(self._slots)
            local_to_global[assigned_local] = global_idx
            slot = Slot(offset=0, size=iv.size, dtype=dtype, device=device)
            group_slots.append(slot)
            self._slots.append(slot)
            free_slots.append((assigned_local, iv.free_seq))
        else:
            global_idx = local_to_global[assigned_local]

        self._key_to_slot[iv.key] = global_idx

    # Lay out slots contiguously with alignment
    offset = 0
    alignment = _get_alignment(device, dtype)
    for slot in group_slots:
        offset = (offset + alignment - 1) // alignment * alignment
        slot.offset = offset
        offset += slot.size

    if offset > 0:
        self._pools[(dtype, device)] = torch.empty(offset, dtype=dtype, device=device)

    return offset
```

### Memory alignment

Each slot's offset is aligned to a device- and dtype-aware minimum. This is
critical for NVFP4 (sub-byte) types and CUDA kernel alignment requirements
(e.g. 256-byte base alignment from `cudaMalloc`):

```python
def _get_alignment(device, dtype):
    """Return the minimum alignment (in elements) for the given device/dtype."""
    element_bytes = torch.empty(0, dtype=dtype, device=device).element_size()
    # At minimum, align to the element size itself
    align_bytes = max(
        element_bytes,
        torch.cuda.get_device_properties(device).texture_alignment
        if device.type == "cuda" else 1,
    )
    return align_bytes // element_bytes
```

### Coloring algorithm — known limitations and future improvements

The current greedy left-edge algorithm with per-key interval merging
meets the basic requirements but has room for improvement:

1. **Per-group isolation**: Coloring runs independently per `(dtype, device)`
   group. Groups on the same device could share a single pool with a unified
   coloring pass, potentially reducing total memory. This requires handling
   different element sizes in the same pool (offset arithmetic must account
   for dtype-specific strides).

2. **Merged range over-estimates active time**: Merging all of a key's
   intervals into a single range (first alloc → last free) may prevent
   other keys from reusing a slot during gaps between the key's intervals.
   For FSDP forward/backward patterns, these gaps are typically small
   (consecutive passes), so the overhead is minimal in practice.

3. **Alternatives worth exploring**:
   - **Global (cross-group) coloring**: Run one coloring pass across all
     ``(dtype, device)`` groups that share a device, reducing total pool
     bytes.
   - **Slot merging pass**: After coloring, adjacent slots with compatible
     dtypes could be merged if their intervals never overlap and no alignment
     constraints are violated.
   - **Size-class bucketing**: Group intervals by size class to reduce
     internal fragmentation when a small interval forces a large slot.

4. **Evaluation criteria**:
   - Total pool bytes vs. the per-key `torch.empty` baseline.
   - Slot count (fewer slots = less metadata overhead).
   - Internal fragmentation (unused bytes within each slot, `slot.size - max_iv_size`).
   - Alignment waste (padding bytes between slots).

**Plan**: Ship with the current algorithm and instrument it with pool-size
and fragmentation metrics. Collect traces from real workloads (large models,
varied parallelism configs). Use the data to guide which improvements are
worth the complexity.

## Runtime: allocate / free / reset

State is driven entirely by ``_optimized_allocate`` and ``_optimized_free``.
No batched reset is needed — if every ``allocate`` is matched by a ``free``
within the micro-batch, ``in_use`` flags and ``_active_keys`` are already
correct by the end.

```python
def allocate(key, size, dtype, device):
    slot_idx = _key_to_slot[key]             # raises KeyError if key never traced
    slot = _slots[slot_idx]
    if slot.in_use and key not in _active_keys:
        raise RuntimeError("Slot collision")
    assert size <= slot.size
    if key in _active_keys:
        # Re-entrant: key already allocated this micro-batch — idempotent
        return Bucket(data=pool[slot.offset : slot.offset + size])
    slot.in_use = True
    _active_keys.add(key)
    return Bucket(data=pool[slot.offset : slot.offset + size])

def free(key):
    if key not in _active_keys:
        return  # double-free or never-allocated
    _slots[_key_to_slot[key]].in_use = False
    _active_keys.discard(key)

def reset():
    """Full teardown: discard pool, plan, and trace."""
    self._phase = "trace"
    self._seq = 0
    self._trace.clear()
    self._trace_meta.clear()
    self._buckets.clear()
    self._active_keys.clear()
    self._pools.clear()
    self._key_to_slot.clear()
    self._slots.clear()
```

### Error handling for unknown keys

If `allocate(key)` is called with a key never seen during trace, `_key_to_slot`
raises `KeyError`. The caller must re-trace (call `reset()`, re-run micro-batch
0, then `plan()`) to pick up new allocation patterns. This guarantees that all
keys used during CUDA graph replay were planned, keeping addresses stable.

No on-the-fly slot allocation is supported — it would change pool tensor sizes
and invalidate all captured CUDA graphs.

## CUDA graph integration lifecycle

The full lifecycle spanning trace, plan, capture, and replay:

```
                    ┌─────────────────────────────────────────┐
                    │ Micro-batch 0 (trace)                    │
                    │ Phase: "trace"                           │
                    │                                          │
                    │ forward_pre_hook (per-module):           │
                    │   if is_root: forward_phase=True         │
                    │   unshard → _trace_allocate              │
                    │ Backward:     alloc/free → _trace_*      │
                    │               records                    │
                    │ Post-backward: plan() builds             │
                    │                _key_to_slot, allocates   │
                    │                pool tensors              │
                    │                Phase → "optimized"       │
                    └──────────────────┬──────────────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
    ┌─────────▼──────────┐   ┌────────▼───────────┐   ┌───────▼───────────┐
    │ Micro-batch 1      │   │ Micro-batch 2       │   │ Micro-batch N     │
    │ (capture)          │   │ (replay)            │   │ (replay)          │
    │ Phase: "optimized" │   │ Phase: "optimized"  │   │ Phase: "optimized"│
    │                    │   │                     │   │                   │
    │ forward_pre_hook   │   │ forward_pre_hook    │   │ forward_pre_hook  │
    │ (per-module):      │   │ (per-module):       │   │ (per-module):     │
    │   assert not       │   │   assert not        │   │   assert not      │
    │   cuda_graph_active│   │   cuda_graph_active │   │   cuda_graph_active│
    │   unshard→allocate │   │   unshard→allocate  │   │   unshard→allocate│
    │     → key→slot     │   │     → key→slot      │   │     → key→slot    │
    │                    │   │                     │   │                   │
    │   if not captured: │   │   graphed() call    │   │   graphed() call  │
    │     capture_fwd(): │   │   (replay capture)  │   │   (replay capture)│
    │     • pop hooks    │   │                     │   │                   │
    │       (recursive)  │   │   pool tensors      │   │   pool tensors    │
    │     • set cuda_    │   │   at SAME addr      │   │   at SAME addr    │
    │       graph_active │   │   as MB 1           │   │   as MB 1         │
    │     • warmup       │   │                     │   │                   │
    │     • capture fwd  │   │                     │   │                   │
    │     • clear flag   │   │                     │   │                   │
    │     • restore hooks│   │                     │   │                   │
    │     • install()    │   │                     │   │                   │
    │                    │   │                     │   │                   │
    │ Post-bwd hooks     │   │ Post-bwd hooks      │   │ Post-bwd hooks    │
    │   fire (eager)     │   │   fire              │   │   fire            │
    │                    │   │                     │   │                   │
    │ Post-backward:     │   │ Post-backward:      │   │ Post-backward:    │
    │   (hooks clean up) │   │   (hooks clean up)  │   │   (hooks clean up)│
    └────────────────────┘   └─────────────────────┘   └───────────────────┘
```

### Per-module CUDA graph capture detail (MB 1)

FSDP captures one CUDA graph per compatible leaf module (not the whole model).
The capture is triggered inside the unified per-module forward pre-hook
(``forward_pre_hook``, which also handles root-level phase init):

1. **Pre-forward hook allocates param**: The hook calls
   `allocate(pg_id, "model_weight")` → key→slot lookup → pool view at fixed
   address. FSDP all-gathers shards into that view. The param buffer is now
   resident at a deterministic address for the rest of the pool's lifetime.

2. **main_grad is manually allocated**: Before capture, the gradient buffer
   is pre-allocated via `allocate(pg_id, "main_grad")` to ensure both forward
   and backward passes see fixed addresses.

3. **Hooks popped recursively**: `_pop_hooks_recursive` removes all FSDP
   hooks on the target module and every submodule. ``cuda_graph_active`` is
   set to ``True`` so any stray hook that fires during this window asserts.

4. **Warmup + capture**: `make_graphed_callables` runs 3 warmup iterations
   (no hooks → raw forward only) + the actual graph capture. The graph
   records GPU kernels reading/writing the pool views at their fixed addresses.

5. **Cleanup**: ``cuda_graph_active`` is cleared, hooks are restored
   recursively, the patched forward is installed.

6. **Replay**: The patched forward calls `graphed(*flat)`. Since the graph
   captured the fixed pool addresses, replay reads/writes the **same**
   addresses every time — no re-allocation, no address change.

### Key safety property

During capture, ``cuda_graph_active`` is ``True`` and all hooks are popped
(recursively).  The CUDA graph records kernel operations against the pool
tensor addresses.  Because the pool tensors are allocated **once** in
``plan()`` and never resized, the same ``key`` will always resolve to the
same address — regardless of call ordering variations across micro-batches.

### TE wgrad fusion under CUDA graph

When Transformer Engine's gradient-accumulation fusion writes weight
gradients directly into ``param.main_grad``, it sets the Python-side
``grad_added_to_main_grad = True`` to tell FSDP not to overwrite
``main_grad``.  Under CUDA graph replay, TE's GPU kernel still runs but the
``setattr`` is not part of the graph.

**Fix**: Recording and consume live together in ``reduce_grad()``.  On the
first call (trace micro-batch, eager backward), if TE set the flag, we
persist it as ``param._mfsdp_recorded_te_wgrad = True``.  On all subsequent
calls, ``reduce_grad`` checks both the live flag (eager path) and the
recorded flag (CUDA graph replay path).  ``pre_backward_hook`` restores
the recorded value before each backward pass.

```
reduce_grad():
  if grad_added_to_main_grad or _mfsdp_recorded_te_wgrad:
      discard .grad                           # TE already populated main_grad
      if grad_added and enable_cuda_graph:
          _mfsdp_recorded_te_wgrad = True     # persist for future replays
```

## Visual

```
                      ┌───────────────────────────────┐
                      │ Micro-batch 0 (trace)          │
                      │ Phase: "trace"                 │
                      │ Forward + backward runs        │
                      │ All alloc/free calls logged    │
                      └──────────────┬────────────────┘
                                     │
                            ┌────────▼──────────┐
                            │ plan()              │
                            │ Phase → "optimized" │
                            │                     │
                            │ 1. Build intervals  │
                            │    from trace       │
                             │ 2. Merge per-key     │
                             │    intervals         │
                             │ 3. Interval coloring │
                             │    → slots           │
                            │ 3. Align & allocate │
                            │    pool tensors     │
                            │ 4. Build static     │
                            │    key → slot map   │
                            └─────────┬───────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
    ┌─────────▼───────┐     ┌────────▼────────┐     ┌───────▼─────────┐
    │ Micro-batch 1   │     │ Micro-batch 2    │     │ Micro-batch N   │
    │ (capture)       │     │ (replay)         │     │ (replay)        │
    │ Phase:          │     │ Phase:           │     │ Phase:          │
    │ "optimized"     │     │ "optimized"      │     │ "optimized"     │
    │                 │     │                  │     │                 │
    │ alloc(key) →    │     │ alloc(key) →     │     │ alloc(key) →    │
    │   same address  │     │   same address   │     │   same address  │
    └─────────────────┘     └──────────────────┘     └─────────────────┘
```

## Hooks integration (updated for v3)

`_register_forward_pre_hook` is a single hook that serves all FSDP modules
(root and non-root).  Root-only logic (phase flags, CUDA graph stream init)
is gated by ``is_root``.

```
Micro-batch 0 (trace)
┌───────────────────────────────────────────────────────────────────────────┐
│  forward_pre_hook (per-module)                                             │
│    if is_root: forward_phase = True                                        │
│    assert not cuda_graph_active                                            │
│    unshard params → _trace_allocate(); forward (trace)                     │
│                                                                           │
│  pre_backward_hook (per-module, reverse order)                             │
│    if is_root: switch to backward phase, enqueue final callback            │
│    assert not cuda_graph_active                                            │
│    unshard (bwd) → _trace_allocate()                                       │
│    reset: post_backward_issued=False, grad_added_to_main_grad=False        │
│    TE wgrad groups: overwrite_main_grad=True                               │
│    backward (trace, eager) → TE sets grad_added_to_main_grad               │
│                                                                           │
│  post_backward (per-module): reshard, reduce_grad, post_backward_issued   │
│                                                                           │
│  _post_backward_final_callback                                             │
│    handle modules with post_backward_issued=False (activation ckpt)        │
│    drain async reduce-grad events                                          │
│    plan() → "optimized"                                                    │
│                                                                           │
│    reduce_grad (per-module, inside post_backward):                         │
│      if grad_added_to_main_grad (TE set it eagerly):                       │
│        discard .grad; param._mfsdp_recorded_te_wgrad ← True               │
│        (recorded for CUDA graph replays where setattr doesn't fire)        │
└───────────────────────────────────────────────────────────────────────────┘

Micro-batch 1+ (optimized)
┌───────────────────────────────────────────────────────────────────────────┐
│  forward_pre_hook (per-module)                                             │
│    if is_root: forward_phase = True                                        │
│    assert not cuda_graph_active                                            │
│    unshard params → allocate() → key→slot                                  │
│    → if enable_cuda_graph and not yet captured and not backward:           │
│        FSDPCudaGraphRunner.capture_forward()                               │
│        (pops hooks recursively, sets cuda_graph_active,                    │
│         runs warmup+capture, restores hooks, clears flag)                  │
│    forward (optimized, graph replays if captured)                          │
│                                                                           │
│  pre_backward_hook (per-module, reverse order)                             │
│    if is_root: switch to backward phase, enqueue final callback            │
│    assert not cuda_graph_active                                            │
│    unshard (bwd) → allocate() → key→slot                                   │
│    reset: post_backward_issued=False                                       │
│    grad_added_to_main_grad ← _mfsdp_recorded_te_wgrad  (CUDA graph)       │
│    TE wgrad groups: overwrite_main_grad=True                               │
│    backward (optimized, graph replays if captured)                         │
│                                                                           │
│  post_backward (per-module): reshard, reduce_grad, post_backward_issued   │
│                                                                           │
│  _post_backward_final_callback                                             │
│    handle modules with post_backward_issued=False                          │
│    drain async reduce-grad events                                          │
│    reset context state for next micro-batch                                │
└───────────────────────────────────────────────────────────────────────────┘
```

## Debug: `dump_trace()`

Updated for v3 to show the full static key→slot mapping (not just active keys):

```python
def dump_trace(self) -> str:
    lines = [f"=== TracePoolAllocator (phase={self._phase}) ==="]
    # ... trace events (unchanged) ...

    if self._phase == "optimized":
        lines.append(f"\nslots: {len(self._slots)}")
        for i, slot in enumerate(self._slots):
            lines.append(
                f"  slot[{i}]: offset={slot.offset} size={slot.size} "
                f"dtype={slot.dtype} device={slot.device} "
                f"{'in_use' if slot.in_use else 'free'}"
            )
        total_bytes = sum(
            s.size * torch.empty(0, dtype=s.dtype).element_size()
            for s in self._slots
        )
        lines.append(f"\ntotal pool: {len(self._slots)} slots, {total_bytes} bytes")
        lines.append(f"\nkey_to_slot ({len(self._key_to_slot)} entries):")
        for key, slot_idx in sorted(self._key_to_slot.items(), key=lambda x: str(x[0])):
            slot = self._slots[slot_idx]
            lines.append(
                f"  {key} -> slot[{slot_idx}] "
                f"(offset={slot.offset}, size={slot.size}, "
                f"address=0x{pool[slot.offset].data_ptr():x})"
            )
        lines.append(f"\nactive_keys ({len(self._active_keys)}):")
        for key in self._active_keys:
            lines.append(f"  {key}")
    return "\n".join(lines)
```

## Key properties

| Property | How it's achieved |
|---|---|
| Fixed address per key | Pool tensors allocated once in `plan()`, never resized. Each key maps to one slot at a fixed, aligned offset. `pool[offset:offset+size]` returns identical view every time. |
| Memory efficiency | Left-edge interval coloring reuses slots for non-overlapping allocations. Per-key intervals are merged into a single time range before coloring, so each key gets exactly one slot — guaranteed stable address with minimal slot count. |
| CUDA graph compatible | One key → one slot → one address. Per-key intervals merged into a single time range during ``plan()`` ensures each key gets exactly one dedicated slot at a fixed offset. |
| No fragmentation within pool | Slots are laid out contiguously with alignment padding. No gaps between slots (only alignment-padding gaps). No dynamic allocation/deallocation — pool is a single `torch.empty`. |
| Simple implementation | `allocate()` is a dict lookup + guard. `free()` is a flag clear. State is self-driving — no batch reset, no cursors, no seq walking. |
| Same trace mechanism | Phase 1 (trace) is identical. Only the plan output and runtime dispatch change. |
| Backward compatible for non-CUDA-graph users | The key→slot runtime also works for regular eager execution — it's strictly simpler than the seq-driven approach. |
