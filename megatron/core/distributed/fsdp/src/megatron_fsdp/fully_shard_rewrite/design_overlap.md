# Design: Unshard Prefetch & Reduce-Grad Overlap in `fully_shard_rewrite`

---

## File Map

| File | Role in overlap |
|---|---|
| `fully_shard.py` | `FSDPModule`, `_FSDPRootContext`, `_FSDPState`, all hooks, `unshard()`, `reshard()`, `reduce_grad()`, final callback |
| `param_group.py` | `ParameterGroup.unshard(async_op)`, `reduce_grad()`, `release_grad_buffer()` |
| `dp_buffer.py` | `DataParallelBuffer.unshard(async_op)` (all-gather + `p.data` rebind), `reduce_grad()` (reduce-scatter + shard accumulation) |
| `allocator.py` | `TemporaryBucketAllocator` — pooled memory for unsharded parameter and gradient buffers |

No changes to `utils.py` are needed for either overlap feature.

---

## `_FSDPRootContext` — Shared Coordination Object

One instance is created by the root `FSDPModule` at `_init_fsdp_state()` time and stored as
`_fsdp_root_context` on **every** FSDP module (root and all children).

```python
@dataclass
class _FSDPRootContext:
    # --- CUDA streams ---
    ag_stream: torch.cuda.Stream   # all-gather (unshard) side stream
    rs_stream: torch.cuda.Stream   # reduce-scatter side stream
    # When the corresponding feature flag is False, these are set to
    # torch.cuda.current_stream() so stream-context switches become no-ops.

    # --- Static execution order (set at init, never mutated) ---
    forward_order: List[FSDPModule]
    # Populated as: [m for m in root.modules() if isinstance(m, FSDPModule)]

    # --- Unshard prefetch tracking ---
    unshard_done_events: Dict[int, Optional[torch.cuda.Event]]
    # module_id -> Event: signals when that module's all-gather is complete.
    # None means "not yet launched" or "already consumed by a wait".

    # --- Reduce-grad overlap tracking ---
    reduce_grad_buckets: Dict[int, List[Tuple[torch.cuda.Event, ParameterGroup]]]
    # module_id -> [(event, param_group), ...]
    # Each entry: event signals RS complete; param_group holds the grad buffer.

    # --- Feature flags ---
    enable_unshard_prefetch: bool
    enable_async_reduce_grad: bool
```

### Initialization in `_init_fsdp_state()`

```python
forward_order = [child for child in self.modules() if isinstance(child, FSDPModule)]
root_context = _FSDPRootContext(
    ag_stream=torch.cuda.Stream() if enable_unshard_prefetch else torch.cuda.current_stream(),
    rs_stream=torch.cuda.Stream() if enable_async_reduce_grad else torch.cuda.current_stream(),
    forward_order=forward_order,
    reduce_grad_buckets={id(m): [] for m in forward_order},
    unshard_done_events={id(m): None for m in forward_order},
    enable_unshard_prefetch=enable_unshard_prefetch,
    enable_async_reduce_grad=enable_async_reduce_grad,
)
# Root gets its own context; each child is overwritten with the same object:
for child in self.modules():
    if child is not self and isinstance(child, FSDPModule):
        child._init_fsdp_state(...)     # creates a child-local context first
        child._fsdp_state._is_root = False
        setattr(child, "_fsdp_root_context", root_context)   # then replaced by root's
```

`forward_order` is **static** (module tree topology, computed once). There is no first-pass
dynamic recording phase.

---

## Feature 1: Unshard Prefetch

### Hook entry points

```python
# _register_forward_pre_hook:
module.unshard(async_op=ctx.enable_unshard_prefetch, bwd_pass=False)

# _register_backward_pre_hook (called inside register_multi_grad_hook):
module.unshard(async_op=ctx.enable_unshard_prefetch, bwd_pass=True)
```

### `FSDPModule.unshard(async_op, bwd_pass)`

```
stream = ctx.ag_stream if async_op else current_stream

# Build the work list: self + (optionally) next module to prefetch
if async_op:
    prefetch = _get_prefetch_next_modules(bwd_pass)   # returns [next_module] or []
else:
    prefetch = []

for module in [self] + prefetch:
    if ctx.unshard_done_events[id(module)] is not None:
        continue          # AG already launched for this module — skip

    with torch.cuda.stream(stream):
        for _, param_group in module._named_param_groups:
            param_group.unshard(async_op=async_op)
            # → DataParallelBuffer.unshard(async_op):
            #     allocate unsharded bucket, launch all_gather_into_tensor,
            #     rebind p.data → unsharded buffer slice (even before AG done!)

    if async_op:
        event = stream.record_event()
        ctx.unshard_done_events[id(module)] = event   # store completion signal

# Synchronize self: block main stream until this module's AG is done
if ctx.unshard_done_events[id(self)] is not None:
    ctx.unshard_done_events[id(self)].wait()          # main stream waits on ag_stream event
    ctx.unshard_done_events[id(self)] = None          # consume and reset

# Install full parameter tensors into the nn.Module (safe after event.wait)
for param_names, param_group in self._named_param_groups:
    for name, param in zip(param_names, param_group.params):
        _replace_module_parameter(self, name, param)  # swaps nn.Parameter object
```

**Important: `p.data` rebind race.**
`DataParallelBuffer.unshard()` rebinds `p.data` to the unsharded buffer slice
**inside the `with torch.cuda.stream(stream)` block**, before the all-gather completes (when
`async_op=True`). The memory is already allocated and the slice indices are correct; only the
NCCL fill is in-flight. The outer `unshard()` guards correctness by calling `event.wait()`
before calling `_replace_module_parameter`, so the module's parameters are safe to read by
the time the forward kernel uses them.

Prefetched modules' data also becomes valid when their own pre-hook later calls `event.wait()`
for them. If a module's pre-hook arrives and its event is already set (prefetch was launched
by the previous module), it just waits on the event and skips re-launching the AG.

### `_get_prefetch_next_modules(bwd_pass)`

```python
order = list(reversed(ctx.forward_order)) if bwd_pass else ctx.forward_order
i = order.index(self)
return [order[i + 1]] if i + 1 < len(order) else []
```

Exactly one module is prefetched per step. Multi-module lookahead is a future extension.

### `FSDPModule.reshard()`

```python
for param_names, param_group in self._named_param_groups:
    param_group.reshard()                           # → DataParallelBuffer.reshard()
                                                    #   frees TemporaryBucketAllocator bucket
                                                    #   sets _unsharded_buffer = None
    for name, dist_param in zip(param_names, param_group.dist_params):
        _replace_module_parameter(self, name, dist_param)   # reinstall sharded DTensor
ctx.unshard_done_events[id(self)] = None    # reset so next iteration can prefetch again
```

---

## Feature 2: Reduce-Grad Overlap

### Hook entry point

Inside the `post_backward` closure registered by `_register_backward_hook`:

```python
module.reshard()
module.reduce_grad(async_op=ctx.enable_async_reduce_grad)
module.post_backward_issued = True
```

### `FSDPModule.reduce_grad(async_op)`

```
stream = ctx.rs_stream if async_op else current_stream

# --- Step 1: Sliding drain — free grad buffers 2 positions back in backward order ---
if async_op:
    backward_order = list(reversed(ctx.forward_order))
    for i, module in enumerate(backward_order):
        if i - 2 >= 0:
            for event, param_group in drain(ctx.reduce_grad_buckets[id(backward_order[i-2])]):
                event.wait()
                param_group.release_grad_buffer()
                #   → deletes param.main_grad views (prevents TE grad-accum-fusion leak)
                #   → DataParallelBuffer.reshard() (frees unsharded grad bucket)
        if module is self: break

# --- Step 2: Copy .grad → main_grad_buffer (on main stream, fast memcpy) ---
for param_names, param_group in self._named_param_groups:
    if not param_group.requires_grad: continue

    for name, param in zip(param_names, param_group.params):
        main_grad = param.get_main_grad()
        if param.grad is None:
            if not getattr(param, 'grad_added_to_main_grad', False):
                main_grad.zero_()       # no TE fusion: zero the slot
        else:
            main_grad.copy_(param.grad.detach())   # normal backward: copy .grad
            del param.grad

    # --- Step 3: Reduce-scatter on rs_stream ---
    if async_op:
        stream.wait_stream(current_stream)    # ensure .grad copy is visible to rs_stream
        with torch.cuda.stream(stream):
            param_group.reduce_grad()
            #   → DataParallelBuffer.reduce_grad() (synchronous within this stream):
            #       fetch_unsharded_buffer() allocates full grad buffer
            #       reduce_scatter_tensor(output=grad_shard, input=full_grad)
            #       self.data[local_idx:...] += grad_shard
            event = torch.cuda.Event()
            event.record()
        ctx.reduce_grad_buckets[id(self)].append((event, param_group))
        # param_group.release_grad_buffer() is NOT called here; deferred until drain/final CB
    else:
        param_group.reduce_grad()
        param_group.release_grad_buffer()

    # --- Step 4: Install dist_grad on dist_param (runs in stream context) ---
    for name, param, dist_param, dist_grad in zip(
        param_names, param_group.params, param_group.dist_params, param_group.dist_grads
    ):
        if param.requires_grad and dist_grad is not None:
            with torch.cuda.stream(stream):
                dist_grad = dist_grad.to(dist_param.dtype)  # dtype cast on rs_stream
            setattr(dist_param, "grad", dist_grad)          # Python ref, no GPU dependency
```

**Key design point — `DataParallelBuffer.reduce_grad()` has no `async_op` parameter.**
The operation is inherently synchronous *within whatever stream is current* when called. The
"async" behavior is achieved entirely by the caller dispatching into `rs_stream` via
`with torch.cuda.stream(stream)`. This avoids any API changes to `DataParallelBuffer`.

**`grad_added_to_main_grad` flag:**
When TransformerEngine's `gradient_accumulation_fusion` is active, the backward kernel writes
directly into `param.main_grad` (bypassing `.grad`). The `pre_backward_hook` resets this flag
to `False` before each backward pass; the kernel sets it to `True` after writing. In
`reduce_grad`, the `zero_()` call is skipped when the flag is `True` to preserve the
fused-gradient value.

### Sliding Drain: The `i-2` Rule

The drain loop ensures at most **2 modules' gradient buffers** are live at any time:

```
Backward processing order (reversed forward):
  layer[N]   ← current (i=0): i-2=-2  → no drain
  layer[N-1] ← current (i=1): i-2=-1  → no drain
  layer[N-2] ← current (i=2): i-2=0   → drain layer[N]    (i-2=0)
  layer[N-3] ← current (i=3): i-2=1   → drain layer[N-1]  (i-2=1)
  ...
```

By the time RS for `layer[N-2]` starts, `layer[N]`'s RS event is expected to be done
(two backward steps of compute have elapsed). `event.wait()` makes this explicit and safe
even if the timing estimate is wrong.

### `_post_backward_final_callback`

Registered on the root by `_register_post_backward_final_callback()` via
`Variable._execution_engine.queue_callback`. Fires after all autograd ops complete.

```python
def _post_backward_final_callback(root_state, root_module):
    ctx = root_module._fsdp_root_context
    stream = ctx.rs_stream

    # Handle modules whose post_backward hook was never triggered
    # (e.g. modules with no grad-requiring inputs on this micro-batch)
    for module in reversed(ctx.forward_order):
        if module.post_backward_issued:
            continue
        module.reshard()
        module.reduce_grad(async_op=ctx.enable_async_reduce_grad)

    # Drain ALL remaining buckets (anything not drained by the sliding rule above)
    for buckets in ctx.reduce_grad_buckets.values():
        while buckets:
            event, param_group = buckets.pop()
            event.wait()
            param_group.release_grad_buffer()

    # Ensure main stream sees all rs_stream work before optimizer step
    torch.cuda.current_stream().wait_stream(stream)

    root_state._post_backward_callback_queued = False
```

---

## Complete Timeline

```
FORWARD PASS (enable_unshard_prefetch=True)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
main stream:  │← compute L[0] →│← compute L[1] →│← compute L[2] →│
ag_stream:    │AG(L[0])  AG(L[1])│        AG(L[2])│                │

pre-hook L[0]: async unshard L[0] + prefetch L[1] on ag_stream
               event[L[0]].wait() → main stream unblocks
               _replace_module_parameter(L[0])

pre-hook L[1]: event[L[1]] already set → wait (likely done)
               _replace_module_parameter(L[1])
               async prefetch L[2] on ag_stream

pre-hook L[2]: event[L[2]].wait() → main stream unblocks
               _replace_module_parameter(L[2])

BACKWARD PASS (enable_async_reduce_grad=True)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
main stream:  │bwd L[2]│copy grad[2]│bwd L[1]│copy grad[1]│bwd L[0]│copy grad[0]│
ag_stream:    │AG(L[1]) prefetch    │AG(L[0]) prefetch     │                      │
rs_stream:    │                RS(L[2]) ──────│     RS(L[1]) ──────│   RS(L[0])───│

post-bwd L[2]: reshard, copy grad[2]→main_grad, rs_stream.wait(main), RS(L[2]), event[2]
post-bwd L[1]: drain event[2-2]? (i=1, no drain), copy grad[1], RS(L[1]), event[1]
post-bwd L[0]: drain event[L[2]] (i=2, drain backward_order[0]=L[2]), RS(L[0]), event[0]

final_callback:
  drain event[L[1]], event[L[0]]
  main_stream.wait_stream(rs_stream)
  ← optimizer step safe →
```

---

## `DataParallelBuffer.reduce_grad()` — Implementation Note

No `async_op` parameter is needed. The method is purely synchronous within the calling stream:

```python
def reduce_grad(self):
    if not self.is_distributed:
        torch.distributed.all_reduce(self.data, group=self.dp_group)
        return
    full_grad = self.fetch_unsharded_buffer()  # allocates bucket if not present
    sm = self.buffer_index.shard_meta
    grad_shard = full_grad[sm.bucket_data_index : sm.bucket_data_index + sm.size]
    torch.distributed.reduce_scatter_tensor(
        output=grad_shard, input=full_grad, group=self.dp_group
    )
    # Accumulate into persistent shard — supports multi-micro-batch grad accumulation
    self.data[sm.local_data_index : sm.local_data_index + sm.size] += grad_shard
```

The caller (`FSDPModule.reduce_grad`) provides the stream context; `DataParallelBuffer`
just does the collective. This clean separation means `DataParallelBuffer` requires no
modifications for the overlap feature.

---

## Configuration

```python
fully_shard(
    module,
    mesh=mesh,
    enable_unshard_prefetch=True,   # pipeline AG on ag_stream while current module computes
    enable_async_reduce_grad=True,  # pipeline RS on rs_stream while later modules compute bwd
)
```

Setting either flag to `False` assigns `torch.cuda.current_stream()` to the corresponding
stream variable, making all `with torch.cuda.stream(stream)` blocks no-ops — zero overhead,
identical to baseline.

---

## Known Gaps / Recommended Follow-ups

1. **Single-module prefetch only.** `_get_prefetch_next_modules` returns at most one module.
   For networks with many small modules, a size-aware multi-step lookahead
   (analogous to `suggested_AG_prefetch_size` in the old `AllGatherPipeline`) would
   yield better overlap.

2. **`p.data` rebind before AG completion.** `DataParallelBuffer.unshard()` rebinds
   `p.data` to the unsharded buffer slice immediately inside the stream context, before
   the NCCL all-gather fills the data. Correctness is guarded by the outer `event.wait()`,
   but the internal state of `DataParallelBuffer` is briefly inconsistent. A cleaner design
   is to return the buffer without rebinding and let `FSDPModule.unshard()` do the rebind
   after the wait.

3. **Child `_init_fsdp_state` call is redundant.** Each child creates its own
   `_FSDPRootContext` in `_init_fsdp_state()`, only to have it immediately overwritten by
   the root's context (`setattr(child, "_fsdp_root_context", root_context)`). The child-local
   stream allocation is wasted. A small refactor would pass `root_context` directly to avoid
   the redundant allocation.

4. **`torch.current_stream()` vs `torch.cuda.current_stream()`.** The final callback uses
   `torch.current_stream().wait_stream(stream)` (no `.cuda`). This is equivalent only when
   the default device is CUDA; it should be `torch.cuda.current_stream()` for consistency
   with the rest of the file.

5. **Outer-DP / HSDP.** `_FSDPRootContext` does not carry an outer-DP stream for the second
   all-gather needed in hybrid-sharding (outer-DP × inner-FSDP) setups. This mirrors the
   `outer_fsdp_group_param_gather_stream` in the old `AllGatherPipeline`.
