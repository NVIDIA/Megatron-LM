# Design: Megatron-FSDP2 Implementation

---

## File Map

| File | Role in overlap |
|---|---|
| `fully_shard.py` | Public `fully_shard()` API and allocator selection |
| `fsdp_module.py` | `FSDPModule`, `_FSDPRootContext`, `_FSDPState`, `unshard()`, `reshard()`, `reduce_grad()` |
| `hooks.py` | Forward/backward hook registration and final callback |
| `param_group.py` | `ParameterGroup.unshard(async_op)`, `reduce_grad()`, `release_grad_buffer()`, `_init_buffers()` (memory optimization) |
| `dp_buffer.py` | `DataParallelBuffer.unshard(async_op)` (all-gather + `p.data` rebind), `reduce_grad()` (reduce-scatter + shard accumulation) |
| `allocator.py` | `BucketAllocator` hierarchy: `TemporaryBucketAllocator`, `StorageFreeingBucketAllocator`, `TracePoolAllocator` — pooled memory for unsharded parameter and gradient buffers |
| `mcore_fsdp_adapter.py` | `FullyShardedDataParallel.stop_communication()` — synchronizes ag_stream and rs_stream into main stream |

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

    # --- Temporary bucket allocator ---
    bucket_allocator: BucketAllocator
    # One allocator handles all temporary buckets. Allocation keys include
    # both parameter-group identity and buffer role.

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

    # --- Activation recompute support ---
    backward_phase: bool = False
    # True from the root backward pre-hook until the final callback.

    backward_module: Optional[int] = None
    # ``id(module)`` of the FSDP module whose backward is pending next.
    # Derived from ``_reversed_order`` and ``backward_done_modules`` — NOT
    # set by any hook directly.  Updated by ``_advance_backward_module()``.

    backward_done_modules: set = field(default_factory=set)
    # Set of ``id(module)`` for FSDP modules whose backward has completed.
    # Populated in ``post_backward``, cleared in the root backward pre-hook.

    _reversed_order: List[FSDPModule] = field(default_factory=list)
    # ``list(reversed(forward_order))`` — precomputed backward processing order.

    def _advance_backward_module(self) -> None:
        """Set ``backward_module`` to the first module in ``_reversed_order``
        that is NOT in ``backward_done_modules``."""
        for m in self._reversed_order:
            if id(m) not in self.backward_done_modules:
                self.backward_module = id(m)
                return
        self.backward_module = None
```

### Initialization in `_init_fsdp_state()`

```python
bucket_allocator = TracePoolAllocator() if enable_trace_pool else StorageFreeingBucketAllocator()
module._init_named_param_groups(..., bucket_allocator=bucket_allocator)

forward_order = [child for child in self.modules() if isinstance(child, FSDPModule)]
root_context = _FSDPRootContext(
    ag_stream=torch.cuda.Stream() if enable_unshard_prefetch else torch.cuda.current_stream(),
    rs_stream=torch.cuda.Stream() if enable_async_reduce_grad else torch.cuda.current_stream(),
    bucket_allocator=bucket_allocator,
    forward_order=forward_order,
    reduce_grad_buckets={id(m): [] for m in forward_order},
    unshard_done_events={id(m): None for m in forward_order},
    enable_unshard_prefetch=enable_unshard_prefetch,
    enable_async_reduce_grad=enable_async_reduce_grad,
)
# Root and children share one context and one bucket allocator:
for module in forward_order:
    for param_group in module._fsdp_param_groups:
        param_group.set_allocator(root_context.bucket_allocator)
for child in self.modules():
    if child is not self and isinstance(child, FSDPModule):
        child._fsdp_state._is_root = False
        setattr(child, "_fsdp_root_context", root_context)
```

`forward_order` is **static** (module tree topology, computed once). There is no first-pass
dynamic recording phase.

**Safety constraint.** `_init_fsdp_state()` must be called **before** any forward/backward pass
runs.  The method includes a runtime guard that rejects re-initialization if any child
FSDPModule is still unsharded (`unshard_done_events` live) or has pending reduce-scatter
operations (`reduce_grad_buckets` non-empty).  Violating this constraint would overwrite a
running module's `_fsdp_root_context` while its hooks are still firing, causing undefined
behavior.

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

```python
stream = ctx.ag_stream if async_op else torch.cuda.current_stream()

# *** Critical: synchronize ag_stream with current_stream before launching AG ***
# This ensures main-stream writes to parameter data (e.g. reshard after forward,
# or tensor-parallel slice writes) are visible before the all-gather reads them.
# Without this barrier, stale or partially-written parameter shards may be
# gathered, causing convergence divergence.
if async_op:
    stream.wait_stream(torch.cuda.current_stream())

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
            param_group.unshard()
            # → DataParallelBuffer.unshard():
            #     allocate unsharded bucket, launch all_gather_into_tensor,
            #     rebind p.data → unsharded buffer slice (even before AG done!)
            # NOTE: async_op is NOT passed; the stream context handles dispatch.

    if async_op:
        event = stream.record_event()
        ctx.unshard_done_events[id(module)] = event   # store completion signal

# Synchronize self: block main stream until this module's AG is done.
# The event is NOT cleared here — it persists as a "currently unsharded" flag
# and is only cleared by reshard(). This prevents redundant all-gathers during
# activation recompute and prefetch re-entry (see Feature 3 below).
if ctx.unshard_done_events[id(self)] is not None:
    ctx.unshard_done_events[id(self)].wait()          # main stream waits on ag_stream event

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

**Stream ordering barrier.** When `async_op=True`, `unshard()` inserts a
`stream.wait_stream(torch.cuda.current_stream())` on `ag_stream` before launching any
all-gather. This ensures that writes performed on the main stream (e.g., reshard after a
previous forward, or tensor-parallel slice updates) are fully visible to the all-gather
kernel. Without this barrier, stale or partially-written parameter shards may be read by
the NCCL collective, causing convergence divergence.

**NVTX profiling.** `unshard()`, `reshard()`, and `reduce_grad()` each push/pop a
`torch.cuda.nvtx` range (`"MFSDP unshard"`, `"MFSDP reshard"`, `"MFSDP reduce_grad"`)
for profiling visibility in tools like Nsight Systems.

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

```python
stream = ctx.rs_stream if async_op else torch.cuda.current_stream()

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
        stream.wait_stream(torch.cuda.current_stream())    # ensure .grad copy is visible to rs_stream
        with torch.cuda.stream(stream):
            param_group.reduce_grad()
            #   → DataParallelBuffer.reduce_grad() (synchronous within this stream):
            #       fetch_unsharded_buffer() allocates full grad buffer
            #       reduce_scatter_tensor(output=grad_shard, input=full_grad)
            #       self.data[local_idx:...] += grad_shard
        event = stream.record_event()
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

**`grad_added_to_main_grad` and `overwrite_main_grad` flags:**
When TransformerEngine's `gradient_accumulation_fusion` is active, the backward kernel writes
directly into `param.main_grad` (bypassing `.grad`). Two flags coordinate this:

- **`grad_added_to_main_grad`**: Set to `False` in `pre_backward_hook` before each backward
  pass; the kernel sets it to `True` after writing. In `reduce_grad`, the `zero_()` call is
  skipped when `True` to preserve the fused-gradient value.

- **`overwrite_main_grad`**: Set to `True` in `pre_backward_hook` for sharded parameters
  (`optim_grads_params` / `optim_grads`). By default TE **accumulates** (adds) into
  `main_grad` — useful for micro-batch gradient accumulation in non-FSDP settings. With FSDP
  the gradient buffer is re-used across micro-batches; accumulation would silently **double**
  gradients and produce NaN after the first step. This flag tells TE to **overwrite** instead
  of accumulate.

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

## Feature 3: Activation Recomputation (Gradient Checkpointing)

### Problem

When activation checkpointing re-runs a forward pass during backward, the FSDP
forward hooks fire again. Without mitigation this causes two problems:

1. **Redundant all-gather**: `forward_pre_hook` → `unshard()` launches a second
   all-gather even though parameters are already unsharded.
2. **Premature reshard**: `forward_hook` → `reshard()` releases the unsharded
   parameter buffer before backward gradient computation has consumed it.

The baseline Megatron-FSDP addresses this by setting `TrainingState.PRE_BACKWARD`
on all submodules before backprop (`megatron_fsdp.py:900-938`).

### Solution Overview

Two mechanisms:

| Mechanism | Effect |
|---|---|
| **Derived `backward_module`** | `_advance_backward_module()` scans `_reversed_order` for the first module **not** in `backward_done_modules`. This identifies the pending module even when activation recompute fires **before** any layer's `pre_backward_hook` (which is always the case — the checkpoint wrapper triggers recompute, then backward flows through the recomputed graph). |
| **Persistent `unshard_done_events`** | Event is only cleared by `reshard()`, never by `unshard()`. Prevents redundant all-gathers. |

The `backward_phase` flag gates the forward post-hook check; `backward_done_modules`
drives both the derived pointer and the prefetch guard.

### Hook Entry Points

```python
# _register_forward_hook → reshard_param_groups:
if ctx.backward_phase and id(module) == ctx.backward_module:
    return                              # skip reshard — this is the pending module

# _register_backward_pre_hook → pre_backward_hook (root only):
ctx.backward_done_modules.clear()
ctx.backward_phase = True
ctx._advance_backward_module()          # picks first non-done in _reversed_order

# _register_backward_hook → post_backward:
ctx.backward_done_modules.add(id(module))
ctx._advance_backward_module()          # advances to next pending module
module.reshard()

# _register_post_backward_final_callback:
ctx.backward_phase = False
ctx.backward_module = None
ctx.backward_done_modules.clear()
```

### Prefetch Constraint

During backward, `unshard(bwd_pass=True)` prefetches the next module in
`_reversed_order`.  An extra guard skips modules whose backward is already done:

```python
# fsdp_module.py — unshard()
if bwd_pass and id(module) in ctx.backward_done_modules:
    continue        # backward already done — skip prefetch
```

### Timeline

Consider two FSDP-wrapped layers L1, L2 checkpointed together.
`forward_order = [root, L1, L2]`, `_reversed_order = [L2, L1, root]`.

```
----- FORWARD (normal) ----------------------------------
L1: pre → unshard(L1) → forward → reshard(L1)
L2: pre → unshard(L2) → forward → reshard(L2)
      (checkpoint drops intermediates)

----- BACKWARD (root enters phase) ----------------------
root pre_backward:
  clear done_modules, backward_phase = True
  _advance → backward_module = L2    (first not done)
  unshard(root)

----- ACTIVATION RECOMPUTE (L1→L2, inside checkpoint backward) --
L1 pre → unshard(L1) → forward
L1 post: L1 ≠ backward_module(L2) → reshard(L1)
L2 pre → unshard(L2)                (event[L2] set, persistent)
L2 post: L2 == backward_module → skip reshard

----- L2 BACKWARD ----------------------------------------
L2 pre_backward → unshard(L2)       (event set → skip)
L2 backward compute
L2 post_backward:
  done_modules.add(L2), _advance → L1, reshard

----- L1 BACKWARD ----------------------------------------
L1 pre_backward → unshard(L1)       (re-allocates, all-gathers)
L1 backward (gradients already computed → copies .grad)
L1 post_backward:
  done_modules.add(L1), _advance → root, reshard

----- FINAL CALLBACK --------------------------------------
backward_phase = False
backward_module = None
done_modules.clear()
```

### Key Design Decisions

1. **`backward_module` is derived, not set by hooks.**  Activation recompute
   always fires before any layer's `pre_backward_hook`.  Deriving from the done
   set + `_reversed_order` correctly identifies the pending module regardless
   of timing.

2. **`_advance_backward_module()` is called at exactly two points:** root
   `pre_backward_hook` (after clearing the done set) and `post_backward`
   (after adding a done module).  These are the only mutations to `backward_done_modules`.

3. **`backward_done_modules` serves dual purpose:** drives the derived pointer
   AND gates the prefetch guard in `unshard()`.

4. **Event persists between `unshard()` and `reshard()`.**  `unshard()` no
   longer clears its own event.  Prevents redundant all-gathers.

### Edge Cases

- **Sync mode (`enable_unshard_prefetch=False`):** No event is recorded,
  so the persistent-event mechanism does not apply.  `backward_module` still
  prevents premature resharding.
- **Module not reached by backward:** The final callback runs `reshard()`
  for untouched modules.
- **Multiple micro-batches:** All state is reset in the final callback.

---

## Complete Timeline

```
FORWARD PASS (enable_unshard_prefetch=True)
---------------------------------------------------------
main stream:  |← compute L[0] →|← compute L[1] →|← compute L[2] →|
ag_stream:    |AG(L[0])  AG(L[1])|        AG(L[2])|                |

pre-hook L[0]: async unshard L[0] + prefetch L[1] on ag_stream
               event[L[0]].wait() → main stream unblocks
               _replace_module_parameter(L[0])

pre-hook L[1]: event[L[1]] already set → wait (likely done)
               _replace_module_parameter(L[1])
               async prefetch L[2] on ag_stream

pre-hook L[2]: event[L[2]].wait() → main stream unblocks
               _replace_module_parameter(L[2])

BACKWARD PASS (enable_async_reduce_grad=True)
---------------------------------------------------------
main stream:  |bwd L[2]|copy grad[2]|bwd L[1]|copy grad[1]|bwd L[0]|copy grad[0]|
ag_stream:    |AG(L[1]) prefetch    |AG(L[0]) prefetch     |                      |
rs_stream:    |                RS(L[2]) ------|     RS(L[1]) ------|   RS(L[0])---|

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

## Pitfall: Zero-Numel Gradient Shards and Fused Optimizers

**Problem.** When a parameter is sharded across DP ranks, its local shard on a given rank
may contain **zero elements** (e.g., a small bias or embedding table on a high-DP-count setup).
Materializing a `DTensor` gradient for such a shard creates a tensor with `numel() == 0`.

Fused multi-tensor optimizers (e.g., TransformerEngine `FusedAdam`) operate on **all**
gradients in a parameter group in a single fused kernel launch. Passing a zero-numel
tensor into these fused ops can silently corrupt the weight updates for **neighboring
non-empty parameters** in the same group. The optimizer does not crash or raise an error
— it produces numerically incorrect steps that manifest only as **convergence divergence**,
making this extremely difficult to attribute and debug.

**Symptoms (hard to diagnose):**
- Training loss diverges or fails to converge despite correct hyperparameters
- No NaN or Inf in gradients — the corruption is a numerical perturbation
- Occurs only at certain DP-world-size / model-size combinations where sharding produces empty local slices
- Bisecting the codebase is unhelpful because the optimizer runs without error

**Fix in `param_group.py`:**
```python
# DO NOT REMOVE THIS CHECK:
if p.requires_grad and grad_data.numel() > 0:
    self.dist_grads.append(make_uneven_dtensor(...))
else:
    self.dist_grads.append(None)  # zero-numel shard → no DTensor grad
```

By recording `None` for zero-numel shards instead of a DTensor with an empty local tensor,
the fused optimizer never receives the empty tensor and cannot corrupt neighboring updates.
The optimizer already handles `None` grads correctly (parameters without a grad are
simply skipped during the fused update).

**Additional safeguard in `_scale_gradients`:**
```python
for dist_grad in param_group.dist_grads:
    if dist_grad is None:
        continue   # skip zero-numel shards
    dist_grad._local_tensor.mul_(scaling_factor)
```

---

## Pitfall: Attribute Propagation from Original Params to DTensor Dist Params

**Problem.**  `_init_buffers()` in `ParameterGroup` creates DTensor views (`dist_params`) into
sharded buffers and `_replace_module_parameter` registers these DTensors on the module.
However, critical metadata set on the **original** `nn.Parameter` objects by upstream layers
(e.g. TE linear layers from `transformer_engine.py`) is **not** automatically transferred to
the new DTensor wrappers.

The adapter (`mcore_fsdp_adapter.py:310-330`) copies a fixed list of attributes from original
params to dist_params.  If an attribute is missing from this list, downstream consumers that
inspect the registered module parameters will see the wrong metadata.

**Affected attributes and their consumers:**

| Attribute | Set by | Consumer | Failure mode |
|-----------|--------|----------|-------------|
| `allreduce` | `transformer_engine.py:841` — set to `False` on expert MLP weights | `_get_param_groups` (`optimizer/__init__.py:348`) — classifies `is_expert_parallel` | Expert params misclassified as non-expert, causing wrong gradient scaling, clipping group assignment, and optimizer partition placement |
| `is_embedding_parameter` | Various embedding layers | `_get_param_groups` — controls weight decay exclusion | Embeddings incorrectly decayed → convergence divergence |
| `is_embedding_or_output_parameter` | Embedding / output layers | Same as above | Same |
| `sequence_parallel` | TE layers | `parallel_state` / loss computation | Incorrect SP semantics |
| `tensor_model_parallel`, `partition_dim`, `partition_stride` | TE layers | Distributed checkpointing / state dict | Incorrect checkpoint sharding |
| `requires_grad` | All layers | Optimizer | Frozen params may receive updates |

**Fix.**  When adding a new metadata attribute to TE layers or custom modules that are
consumed by downstream code (optimizer, checkpointing, mixed precision), add its name to
the `attr_name` list in `mcore_fsdp_adapter.py` to ensure it propagates to the DTensor
dist_params.

**Debugging.**  Misattributed params can be detected by dumping
`model._log_parameter_groups()` output and verifying that expert params appear in the
`is_expert_parallel` group.  NaN after a single step with `gradient_accumulation_fusion`
is a strong indicator of missing `allreduce` propagation.

---

## Memory Optimization: Freeing Original Parameter Storage

After `ParameterGroup._init_buffers()` copies parameter data into the internal weight buffers
(`model_weight_buffer` and optionally `main_weight_buffer`), the original full parameter tensors
are freed via `_free_storage(p.data)`. The module holds DTensor shard views and `unshard()`
rebinds `.data` to the all-gathered buffer, so the original storage is dead and freeing it
reduces peak memory during model construction.

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

## `stop_communication()` — Main Stream Synchronization

The `FullyShardedDataParallel.stop_communication()` method (in `mcore_fsdp_adapter.py`) ensures
all pending FSDP communication is complete and visible to the main CUDA stream. This is called
before the optimizer step to guarantee that gradient reductions and parameter updates are
synchronized.

For the `fully_shard` path, the implementation was previously `NotImplementedError`. It now
calls:

```python
torch.cuda.current_stream().wait_stream(ctx.ag_stream)  # finish all-gather work
torch.cuda.current_stream().wait_stream(ctx.rs_stream)   # finish reduce-scatter work
```

This brings both communication streams into the main stream, ensuring the optimizer sees
fully-synchronized parameters and gradients.

---

## Known Gaps / Recommended Follow-ups

1. **Single-module prefetch only.** `_get_prefetch_next_modules` returns at most one module.
   For networks with many small modules, a size-aware multi-step lookahead
   (analogous to `suggested_AG_prefetch_size` in the old `AllGatherPipeline`) would
   yield better overlap.

2. **Outer-DP / HSDP.** `_FSDPRootContext` does not carry an outer-DP stream for the second
   all-gather needed in hybrid-sharding (outer-DP × inner-FSDP) setups. This mirrors the
   `outer_fsdp_group_param_gather_stream` in the old `AllGatherPipeline`.

---

## Bucket Allocator Hierarchy

`allocator.py` provides a polymorphic allocator family via the `BucketAllocator`
interface, letting callers swap allocation strategies without changing
`DataParallelBuffer` or `ParameterGroup`.

```
BucketAllocator  (interface)
|-- TemporaryBucketAllocator        — legacy: allocates per key, frees + deletes
|-- StorageFreeingBucketAllocator   — allocates per key, frees storage but keeps bucket
|                                     (same tensor object reused on next allocation)
\-- TracePoolAllocator             — two-phase: trace → plan → static pool
```

### `TracePoolAllocator`

**Purpose.**  During parameter unshard and gradient reduction the FSDP
framework allocates and frees temporary flat buffers (all-gather input/output,
gradient accumulation) in a deterministic, repeatable order.  `TracePoolAllocator`
replaces per-call `torch.empty` + `_free_storage` with a one-time planned pool
that eliminates allocation overhead and fragmentation.

**Design — three phases.**

| Phase | Behaviour |
|---|---|
| **Trace** (``plan()`` not yet called) | Records every ``allocate`` / ``free`` call as a ``(seq, op, key)`` event.  Also stores ``(size, dtype, device)`` metadata per key.  Buckets are allocated via ``torch.empty`` as usual.  Duplicate allocs (without an intervening free) do not generate new trace events. |
| **Plan** (``plan()``) | Replays the trace to build intervals ``(alloc_seq, free_seq, size)`` for each matched alloc/free pair, groups them by ``(dtype, device)``, and runs a greedy left-edge interval-coloring algorithm per group.  Each color is a **slot** in a contiguous flat pool tensor.  Because the same key may appear in multiple intervals, ``_slot_map[key]`` is a **list** of slot indices in alloc order. |
| **Optimized** (after ``plan()``) | ``allocate`` returns a ``Bucket`` with a slice-view into the pool, advancing a per-key cursor through the slot list.  ``free`` marks the most recently allocated slot as unused (idempotent).  ``reset_cursor()`` rewinds all cursors between micro-batches so the same sequence replays. |

**Slot lists and cursors.**  A single allocation key can appear in multiple
intervals — e.g., forward unshard → free → backward unshard → free.  The plan
may assign these intervals to *different* slots (if they overlap) or *reuse*
the same slot (if they don't).  ``_slot_map`` therefore maps each key to a
**list** of slot indices in the exact alloc order.  During optimized-phase
runtime a per-key **cursor** tracks which list entry to consume next; between
micro-batches ``reset_cursor()`` rewinds all cursors to 0.

**Greedy left-edge coloring.**  For each ``(dtype, device)`` group, intervals are
sorted by ``alloc_seq``.  For each interval the algorithm tries to reuse a slot
whose previous occupant has already freed (``slot_free_seq < alloc_seq``).  If no
slot is free a new one is allocated.  The slot is sized to the maximum bucket
assigned to it.  After coloring, slots are laid out contiguously and a single
``torch.empty`` is issued per group.

**Properties.**

- **Optimal slot count:** left-edge produces the minimum number of slots for
  interval graphs — it is impossible to use fewer without causing a conflict.
- **Repeatable trace required:** the same allocate/free call sequence must
  repeat across micro-batches.  Call ``reset_cursor()`` between micro-batches;
  call ``reset()`` to re-profile if the pattern changes.
- **Double-free safe:** ``free`` is idempotent (silently returns if the slot
  is already free), matching ``TemporaryBucketAllocator``'s behavior.

**API.**

```python
allocator = TracePoolAllocator()
# … run one iteration (trace phase) …
pool_elems = allocator.plan()          # returns total element count
# … subsequent micro-batches use the pool …
allocator.reset_cursor()                # between micro-batches
print(allocator.total_pool_bytes)       # bytes across all groups
allocator.reset()                       # back to trace phase
```

**Lifecycle diagram for one allocation key across two micro-batches.**

```
Trace phase                            Optimized phase
-----------                            ---------------
allocate(key) → torch.empty  --.         allocate(key) → pool slot 0   (cursor 0→1)
free(key)     → _free_storage  | plan    free(key)     → slot free
allocate(key) → torch.empty  --'         allocate(key) → pool slot 1   (cursor 1→2)
free(key)     → _free_storage           free(key)     → slot free
                                        -- reset_cursor() --
                                        allocate(key) → pool slot 0   (cursor 0→1)
                                        free(key)     → slot free
                                        ...
```

No ``torch.empty`` or storage resizing occurs in the optimized phase — the pool
owns all memory, and buckets are lightweight views.
