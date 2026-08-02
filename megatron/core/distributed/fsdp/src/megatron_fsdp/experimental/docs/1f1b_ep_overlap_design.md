# 1F1B EP Overlap — Experimental FSDP Integration Design

This document describes the FSDP-side contract required by the 1F1B EP-overlap
schedule (`combined_1f1b`) and how the experimental Megatron FSDP
(`megatron_fsdp.experimental`) must fulfil it.  The v1 implementation
(`megatron_fsdp`) is used as a reference for the contract.

---

## 0. Background — Why the Overlap Schedule Needs Special FSDP Handling

### Normal FSDP flow

Hooks fire on `TransformerLayer`:

```
TransformerLayer.forward()
  → pre_forward hook:  unshard_parameters()
  → actual compute
  → post_forward hook: reshard_parameters()
backward()
  → pre_backward hook:  unshard_parameters()
  → compute grads
  → post_backward hook: reshard_parameters() + reduce_grad()
```

FSDP hooks are registered on the `TransformerLayer` and `register_multi_grad_hook`
captures its output tensors.

### EP overlap flow

The combined 1F1B schedule calls **sub-modules directly**, bypassing
`TransformerLayer.forward()`:

```
combined_forward_backward_step()
  → f_layer.attn.forward()     ← no TransformerLayer hook fires
  → b_layer.mlp.backward()     ← no TransformerLayer hook fires
  → f_layer.moe_dispatch.forward()
  → ...
```

Because the schedule invokes sub-modules (`attn`, `mlp`, `moe_dispatch`,
`moe_combine`) directly, FSDP hooks registered on `TransformerLayer` are
**never triggered**.  FSDP must expose manual management APIs that the schedule
calls explicitly at the right moments.

### Key flags

| Flag | What it does |
|---|---|
| `--overlap-moe-expert-parallel-comm` | Enables the combined 1F1B schedule. |
| `--delay-wgrad-compute` | TE layers postpone weight gradient computation until `backward_dw()`. FSDP must defer `reduce_grad()` until after `backward_dw()`. |

### Why `delay_wgrad_compute` matters for FSDP

When `delay_wgrad_compute=True`, the normal per-param `post_accumulate_grad_hook`
fires **before** `backward_dw()` writes weight gradients.  This corrupts `.grad`
if gradient reduction happens at the wrong time.  FSDP must skip the autograd
backward callback and let the schedule's explicit `post_backward_release_module()`
call handle reduction **after** `backward_dw()` completes.

---

## 1. Schedule Integration Contract

The schedule in `combined_1f1b.py` has five injection points where it calls into
FSDP.  The experimental API must satisfy all five.

### 1.1 Discovery — `find_megatron_fsdp(model)`

**File:** `megatron/core/pipeline_parallel/combined_1f1b.py:68`

```python
fsdp_wrapper = find_megatron_fsdp(model)
```

**Contract:** Return the FSDP root object (v1 `MegatronFSDP` instance) or `None`.
Currently only finds v1 instances (walks `model` looking for `MegatronFSDP`).

**Experimental API requirement:** Extend `find_megatron_fsdp()` (in
`megatron_fsdp/utils.py`) to detect `FsdpModule` instances:

```python
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
if isinstance(model, FsdpModule):
    return model
if hasattr(model, 'module') and isinstance(model.module, FsdpModule):
    return model.module
return None
```

When using the `FullyShardedDataParallelV2` adapter, the schedule passes the
raw model tree (not the adapter), so the root `FsdpModule` is directly
discoverable via `isinstance(model, FsdpModule)`.

### 1.2 Pre-schedule setup — `_replace_param_with_raw_if_needed()`

**File:** `combined_1f1b.py:74`

```python
if fsdp_wrapper is not None:
    fsdp_wrapper._replace_param_with_raw_if_needed()
```

**Contract:** Swap distributed (optimizer-managed) DTensor parameters back to
raw `nn.Parameter` tensors so the schedule can access sub-module parameters
directly.  v1 requires this because it stores optimizer-facing DTensor params
separate from module params.

**Experimental API requirement:** The experimental API always stores raw tensors
backed by `DBuffer` on the module, so no parameter swap is needed. Use this
schedule-entry callback to initialize the root context before any child FSDP
unit is called directly. Otherwise a child on a middle pipeline stage can run
first and incorrectly initialize itself as the root:

```python
# On FsdpModule (module.py)
def _replace_param_with_raw_if_needed(self) -> None:
    self._lazy_init_context()
```

### 1.3 Root backward-phase setup — `pre_backward()`

**File:** `combined_1f1b.py:349`

```python
if fsdp_wrapper is not None and b_model is not None:
    fsdp_wrapper.pre_backward()
```

**Contract:** Called once before each overlapped forward+backward run. In v1,
sets training state to `PRE_BACKWARD` on all sub-modules (so forward
reconstruction during activation recompute doesn't reshard), marks AG buckets
as releasable, and tracks params requiring gradient handling.  Does **not**
auto-enqueue the post-backward final callback — the schedule calls it manually.

**Experimental API requirement:** `FsdpModule` already has a private `pre_backward()`
(method in `module.py`).  Make it public and accept the v1-compatible signature:

```python
# On FsdpModule (module.py)
def pre_backward(self) -> None:
    """Prepare all FSDP parameter groups for backward compute.

    - Unshards parameters (all-gather).
    - If root: registers post_backward_final_callback.
    - Resets post_backward_issued flags.
    """
```

Called by the schedule as `fsdp_wrapper.pre_backward()`.

### 1.4 Per-layer parameter release — `set_fsdp_reshard_hooks()`

**File:** `combined_1f1b.py:405-408`

```python
layer_plan.set_fsdp_reshard_hooks(
    forward_fsdp_wrapper.post_forward_release_module,
    forward_fsdp_wrapper.post_backward_release_module,
)
```

**Contract:** The schedule attaches two callables to each `TransformerLayerSchedulePlan`:

- **`post_forward_release_module(module)`** — called after the last forward node
  of a layer.  Releases all-gathered parameter storage (reshard to DTensor).
- **`post_backward_release_module(module)`** — called after the last backward node
  (`pre_dispatch_computation`).  Reshards parameters **and** reduces gradients
  (copy `.grad` → reduce-scatter → install DTensor `.grad`).

**Experimental API requirement:** Expose two `partial`-wrapped callables on the
root `FsdpModule`:

```python
def _post_forward_release(module: FsdpModule) -> None:
    """Release all-gathered parameters after forward compute."""
    module.reshard_parameters()

def _post_backward_release(module: FsdpModule) -> None:
    """Release parameters and reduce gradients after backward compute."""
    module.reshard_parameters()
    module.reduce_grad()
```

Attach as instance attributes:
```python
root_module.post_forward_release_module = partial(_post_forward_release)
root_module.post_backward_release_module = partial(_post_backward_release)
```

### 1.5 Root backward finalization — `post_backward()`

**File:** `combined_1f1b.py:510`

```python
if fsdp_wrapper is not None and b_model is not None:
    fsdp_wrapper.post_backward()
```

**Contract:** Called once after each overlapped run completes. Handles any
modules whose per-module post-backward was skipped, drains pending async
reduce-grad events, resets root state (`backward_phase`, `backward_done_modules`),
and transitions the bucket allocator from trace → optimized plan.

**Experimental API requirement:** `FsdpModule` already has a private `post_backward()`
(method in `module.py`).  Make it public:

```python
# On FsdpModule (module.py)
def post_backward(self) -> None:
    """Finalise the backward pass.

    - Handles any parameter groups with pending gradients.
    - Drains async reduce-scatter events.
    - Resets root phase state.
    - Transitions allocator trace → optimized (first microbatch only).
    """
```

### 1.6 Gradient sync suppression — `no_sync_func()`

**File:** `combined_1f1b.py:97`

```python
with no_sync_func():
    for i in range(num_microbatches - 1):
        combined_forward_backward_step(...)
```

**Contract:** A context manager that suppresses gradient finalization for
non-final microbatches (so gradients accumulate across inner microbatches).

**Experimental API requirement:** The experimental API already has
`FsdpContext.is_last_microbatch` controlled by the `microbatch()` context
manager.  Add a `no_sync()` method to `FullyShardedDataParallelV2` that
toggles this flag on all root `FsdpContext` instances:

```python
# On FullyShardedDataParallelV2 (mcore_fsdp_adapter.py)
@contextmanager
def no_sync(self):
    roots = [m for m in self.module.modules()
             if isinstance(m, FsdpModule) and m.is_root()]
    for root in roots:
        root.context.is_last_microbatch = False
    try:
        yield
    finally:
        for root in roots:
            root.context.is_last_microbatch = True
```

This automatically wires into `config.no_sync_func` via the existing training
loop contract.

---

## 2. Fine-Grained Hook Registration

When `overlap_moe_expert_parallel_comm=True`, hooks must be registered on
**every sub-module** of each FSDP unit, not just on the FSDP unit itself.

### 2.1 Pre-forward hooks on sub-modules

Register a pre-forward hook on every sub-module of each `FsdpModule`.  When the
schedule calls `f_layer.attn.forward()`, the hook fires → resolves the parent
`FsdpModule` → calls `unshard_parameters()`.

```python
# In fully_shard.py or module.py
def _register_fine_grained_forward_hooks(module: FsdpModule) -> None:
    for submodule in module.modules():
        if _find_fsdp_target(submodule) is not module:
            continue
        submodule.register_forward_pre_hook(_fine_grained_pre_forward,
                                            prepend=True, with_kwargs=True)

def _fine_grained_pre_forward(hook_module, args, kwargs):
    target = _find_fsdp_target(hook_module)
    if target is None:
        return
    target.unshard_parameters()
```

### 2.2 Pre-backward hooks on sub-modules

Register via `register_multi_grad_hook` on each sub-module's output tensors.
When backward reaches a sub-module, the hook fires → `unshard_parameters()`.

```python
def _register_fine_grained_backward_hooks(module: FsdpModule) -> None:
    for submodule in module.modules():
        if _find_fsdp_target(submodule) is not module:
            continue
        _create_custom_backward_hook(submodule,
            lambda m, g: _fine_grained_pre_backward(m))

def _fine_grained_pre_backward(hook_module):
    target = _find_fsdp_target(hook_module)
    if target is None or target._fsdp_pre_backward_done:
        return
    target._fsdp_pre_backward_done = True
    target.unshard_parameters()
```

### 2.3 Wiring from `fully_shard()`

Add a `fine_grained: bool = False` parameter:

```python
def fully_shard(module, mesh, placements, *, fine_grained=False, ...):
    ...
    if fine_grained:
        _register_fine_grained_forward_hooks(module)
        _register_fine_grained_backward_hooks(module)
```

### 2.4 Wiring from the adapter

```python
# In FullyShardedDataParallelV2.__init__()
fully_shard(
    submodule, mesh=mesh, placements=placements,
    mixed_precision_policy=self.mp_policy,
    fine_grained=config.overlap_moe_expert_parallel_comm,
)
```

---

## 3. `delay_wgrad_compute` — Skipping the Autograd Backward Callback

### Problem

The normal backward path uses per-param `post_accumulate_grad_hook` (registered
in `FsdpModule._register_hooks()`) to detect when all parameter gradients are
ready, then calls `post_backward()` which does `reshard_parameters()` +
`reduce_grad()`.  With `delay_wgrad_compute=True`, this hook fires during
`backward()` (activation gradients only), but weight gradients are written
later in `backward_dw()`.  This causes `reduce_grad()` to reduce only partial
gradients.

### Solution

When `skip_backward_callback=True`, do **not** register per-param
`post_accumulate_grad_hook`.  Instead, rely entirely on the schedule's explicit
`post_backward_release_module()` call which fires at the correct time (after
`backward_dw()`).

```python
# In module.py, FsdpModule._register_hooks()
def _register_hooks(self, skip_backward_callback: bool = False) -> None:
    ...
    if not skip_backward_callback:
        for param in parameters:
            param.register_post_accumulate_grad_hook(...)
```

Wired from the adapter:

```python
fully_shard(
    submodule, ...,
    skip_backward_callback=config.delay_wgrad_compute,
)
```

---

## 4. Public API Surface — Summary

### New public methods on `FsdpModule`

| Method | Signature | Description |
|---|---|---|
| `unshard_parameters()` | `() -> None` | All-gather full parameters for compute. Idempotent. |
| `reshard_parameters()` | `() -> None` | Release all-gathered storage, install DTensor params. |
| `reduce_grad()` | `() -> None` | Pack gradients → reduce-scatter → install DTensor `.grad`. |
| `pre_backward()` | `() -> None` | Root backward-phase setup (unshard for backward). |
| `post_backward()` | `() -> None` | Finalise backward (drain pending reductions, reset state). |
| `_replace_param_with_raw_if_needed()` | `() -> None` | Initialize the root context; no parameter swap is needed. |

### New public attributes on root `FsdpModule`

| Attribute | Type | Description |
|---|---|---|
| `post_forward_release_module` | `Callable[[FsdpModule], None]` | Release forward-pass params (reshard only). |
| `post_backward_release_module` | `Callable[[FsdpModule], None]` | Release backward-pass params (reshard + reduce). |

### New parameters on `fully_shard()`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `fine_grained` | `bool` | `False` | Register hooks on every sub-module for EP overlap. |
| `skip_backward_callback` | `bool` | `False` | Skip per-param `post_accumulate_grad_hook` for `delay_wgrad_compute`. |

### New method on `FullyShardedDataParallelV2`

| Method | Signature | Description |
|---|---|---|
| `no_sync()` | `() -> contextmanager` | Suppress gradient finalization for non-final microbatches. |

### Changes to `find_megatron_fsdp()` (`megatron_fsdp/utils.py`)

Extend to detect `FsdpModule` instances (currently only finds v1 `MegatronFSDP`).

---

## 5. Implementation Phases

### Phase 1 — Public APIs on `FsdpModule`
**Files:** `experimental/module.py`, `experimental/fully_shard.py`, `experimental/__init__.py`

- Rename `_unshard_parameter_groups()` → `unshard_parameters()` (public)
- Rename `_reshard_parameter_groups()` → `reshard_parameters()` (public)
- Rename `_reduce_gradient_groups()` → `reduce_grad()` (public)
- Make `pre_backward()` public
- Make `post_backward()` public
- Add `_replace_param_with_raw_if_needed()` as the root-context schedule entry point
- Add `post_forward_release_module` / `post_backward_release_module` attributes

### Phase 2 — Extend `find_megatron_fsdp()`
**File:** `megatron_fsdp/utils.py`

- Add `FsdpModule` detection alongside existing v1 detection

### Phase 3 — Fine-grained hook registration
**Files:** `experimental/fully_shard.py`, `experimental/module.py`, `mcore_fsdp_adapter.py`

- Add `fine_grained` parameter to `fully_shard()`
- Implement sub-module pre-forward hook registration
- Implement sub-module pre-backward hook registration
- Wire `fine_grained` flag from adapter config

### Phase 4 — `no_sync()` context manager
**File:** `mcore_fsdp_adapter.py`

- Add `no_sync()` to `FullyShardedDataParallelV2`
- Remove validation gate blocking EP overlap with v2

### Phase 5 — `delay_wgrad_compute` support
**Files:** `experimental/module.py`, `mcore_fsdp_adapter.py`

- Add `skip_backward_callback` parameter to `fully_shard()` / `_register_hooks()`
- Wire from adapter when `config.delay_wgrad_compute`

### Phase 6 — Activation recomputation guard
**File:** `experimental/module.py`

- Add `_training_state` flag to skip `reshard_parameters()` during recompute

---

## 6. Files Touched

| File | Phase | Change |
|---|---|---|
| `experimental/module.py` | 1, 3, 5, 6 | Public APIs, fine-grained hooks, skip_backward_callback, recompute guard |
| `experimental/fully_shard.py` | 1, 3, 5 | New params, hook registration, release callables |
| `experimental/__init__.py` | 1 | New public exports |
| `mcore_fsdp_adapter.py` | 3, 4, 5 | Wire fine_grained/skip_backward, add no_sync, remove validation gate |
| `megatron_fsdp/utils.py` | 2 | Extend `find_megatron_fsdp()` |
| `combined_1f1b.py` | — | No changes needed |
| `model_chunk_schedule_plan.py` | — | No changes needed |

---

## 7. Edge Cases

### 7.1 Activation recomputation with overlap

When activation recomputation overlaps with backward, the forward pass
reconstructs activations that are still needed by the ongoing backward.
`post_forward()` must NOT reshard parameters during recompute (they're still
needed).  Add a training-state flag to `FsdpContext`:

```python
ctx.is_recompute = False  # set by the schedule or autograd
```

In `post_forward()`:
```python
if ctx.is_recompute:
    return  # skip reshard
module.reshard_parameters()
```

### 7.2 Interleaved pipeline parallelism + EP overlap

Not supported with FSDP in the current schedule.  The schedule explicitly
rejects multi-chunk models with EP overlap (`combined_1f1b.py:317-321`).
No additional validation needed in the experimental API.

### 7.3 `fsdp_double_buffer` incompatibility

Double buffering is incompatible with per-sub-module parameter management.
The experimental API does not implement double buffering, so no additional
validation needed.

---

## 8. Reference — v1 Contract

For context, here is the exact v1 API contract that the schedule calls:

| Call Site (combined_1f1b.py) | v1 Attribute/Method | Type |
|---|---|---|
| Line 68 | `find_megatron_fsdp(model)` | Discovers `MegatronFSDP` instance |
| Line 74 | `fsdp_wrapper._replace_param_with_raw_if_needed()` | Method (root-context initialization for experimental) |
| Line 97 | `no_sync_func()` → `model.no_sync()` | Context manager on adapter |
| Line 349 | `fsdp_wrapper.pre_backward()` | `partial(_root_pre_backward, module=None, skip_backward_hook=True)` |
| Line 405-407 | `forward_fsdp_wrapper.post_forward_release_module` | `partial(_post_forward, input=None, output=None)` |
| Line 405-407 | `forward_fsdp_wrapper.post_backward_release_module` | `_post_backward_release_module` |
| Line 510 | `fsdp_wrapper.post_backward()` | `_root_post_backward` |
