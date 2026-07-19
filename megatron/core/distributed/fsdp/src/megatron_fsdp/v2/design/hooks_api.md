# Megatron FSDP v2 hooks API

This document describes the callable contract for
`megatron.core.distributed.fsdp.src.megatron_fsdp.v2.hooks`.

For hook ordering, overlap behavior, and state-machine details, see
[`design.md`](design.md). This document is intentionally limited to API
semantics for callers that need to invoke the same hook logic outside the
default registration path, such as the 1F1B EP-overlap schedule.

These functions are integration APIs for Megatron FSDP v2 internals, not a
general user extension mechanism.

## Target resolution

### `_find_fsdp_target(hook_module) -> Optional[FSDPModule]`

Returns the FSDP module that owns `hook_module`.

| `hook_module` state | Result |
|---|---|
| It is already an `FSDPModule` | Returns `hook_module` |
| It has `_fsdp_parent_module` | Calls the stored weak reference and returns the parent `FSDPModule` |
| It has no FSDP owner | Returns `None` |

`_fsdp_parent_module` is installed during `FSDPModule._init_fsdp_state()` so
fine-grained hooks registered on non-FSDP submodules can delegate to the nearest
owning FSDP module.

## API contracts

| Function | Accepted module argument | Main responsibility |
|---|---|---|
| `mfsdp_forward_pre_hook(hook_module, args, kwargs)` | `FSDPModule` or owned submodule | Resolve the FSDP target, install a pending optimizer-to-model weight refresh at the next normal root forward, enter forward phase when appropriate, unshard parameters, release stale grad storage, and record CUDA-graph sample inputs for direct FSDP-module calls. |
| `mfsdp_post_forward_hook(module, *unused)` | `FSDPModule` only | Record CUDA-graph sample outputs when applicable, then reshard after forward unless the module is being used by activation recomputation. |
| `mfsdp_pre_backward_setup(hook_module, grads=None, skip_final_callback=False)` | `FSDPModule` or owned submodule | Resolve the FSDP target, enter backward phase on the root, optionally enqueue the final callback, unshard for backward, and reset per-module backward bookkeeping. |
| `mfsdp_post_backward_hook(module)` | `FSDPModule` only | For the module and its recursive FSDP submodules whose post-backward work has not issued, reshard, reduce gradients, and mark them done. |
| `mfsdp_post_backward_final_callback(root_module)` | Root `FSDPModule` only | Finish skipped post-backward work, drain async reduce-grad events, arm a lazy optimizer-to-model weight refresh at an optimizer-step boundary, reset root backward state, clear fine-grained pre-backward flags, finalize trace-pool planning, and trigger CUDA-graph capture when eligible. |

All hook APIs assert that they are not called while `ctx.cuda_graph_active` is
true. CUDA graph replay must not run Python hooks.

## Function details

### `mfsdp_forward_pre_hook(hook_module, args, kwargs)`

This is the shared pre-forward implementation for both default FSDP-module hooks
and fine-grained submodule hooks.

- No-ops if `_find_fsdp_target(hook_module)` returns `None`.
- On the next normal root forward after an `is_last_backward=True` callback,
  consumes `ctx.model_weight_refresh_pending` by calling
  `_copy_main_weights_to_model_weights()` before parameter unshard or prefetch.
  Activation-recompute forwards never consume this flag.
- Before any normal root parameter unshard, performs a root-wide gradient
  liveness sweep. If a plain PyTorch optimizer has cleared every
  optimizer-facing gradient, the sweep resets stale parameter-group
  accumulation flags and releases distributed-gradient backing storage. This
  sweep is skipped entirely for full-iteration CUDA graph mode, which owns
  stable gradient storage and in-place zeroing across graph replays.
- Sets root forward/backward phase state for a normal forward pass.
- Unshards the target's parameters for forward; during activation recomputation,
  also ensures backward-pass buffers are available.
- Calls `ParameterGroup._release_grad_storage_if_unused()` for each parameter
  group. This path must stay idempotent because fine-grained hooks may call the
  function more than once for the same FSDP target.
- Records CUDA-graph sample inputs only when the hook was invoked on the
  `FSDPModule` itself. Submodule calls still perform FSDP lifecycle work, but do
  not define a standalone graph boundary.

### `mfsdp_post_forward_hook(module, *unused)`

This post-forward hook is only valid for direct `FSDPModule` calls. Passing any
other module type raises `TypeError`.

The hook records CUDA-graph sample outputs when graph capture is enabled and the
module is compatible. It then calls `module.reshard()` unless the module is the
current backward module during activation recomputation, where resharding would
drop parameters that backward still needs.

### `mfsdp_pre_backward_setup(hook_module, grads=None, skip_final_callback=False)`

This is the shared pre-backward implementation for default and fine-grained
backward hooks.

- No-ops if `_find_fsdp_target(hook_module)` returns `None`.
- Uses `_fsdp_pre_backward_done` to deduplicate repeated fine-grained hook calls
  for the same FSDP target.
- On the root module, switches the root context into backward phase, advances
  the expected backward module, and enqueues `mfsdp_post_backward_final_callback`
  unless `skip_final_callback=True`.
- Unshards target parameters for backward.
- Resets `post_backward_issued`.
- Resets Transformer Engine gradient-fusion state on each parameter.
- Initializes distributed-gradient storage needed by CUDA graph modes.

If `skip_final_callback=True`, the caller is responsible for calling
`mfsdp_post_backward_final_callback(root_module)` exactly once after the
microbatch backward pass has completed.

### `mfsdp_post_backward_hook(module)`

This post-backward hook is only valid for direct `FSDPModule` calls. Passing any
other module type raises `TypeError`.

The hook walks `module._get_fsdp_modules(recursive=True)`. For each FSDP module
whose `post_backward_issued` flag is still false, it:

1. adds the module id to `ctx.backward_done_modules`,
2. reshards parameters,
3. reduces gradients using `ctx.enable_async_reduce_grad`, and
4. sets `post_backward_issued=True`.

After that loop it advances `ctx.backward_module`.

### `mfsdp_post_backward_final_callback(root_module)`

This callback is only valid for the root `FSDPModule`. Passing a non-FSDP module
raises `TypeError`; passing a non-root FSDP module raises `RuntimeError`.

The callback is the final cleanup point for a microbatch backward pass:

- It handles any FSDP module in `ctx.forward_order` whose post-backward hook did
  not issue.
- It waits on and releases pending async reduce-grad buckets.
- When `ctx.is_last_backward` is true, it sets
  `ctx.model_weight_refresh_pending=True`. An integrated optimizer may consume
  the flag immediately by calling `_copy_main_weights_to_model_weights()`;
  otherwise the next normal root pre-forward consumes it lazily.
- It resets root backward state so the next microbatch can start cleanly.
- It clears `_fsdp_pre_backward_done` on all FSDP modules in `ctx.forward_order`.
- It transitions `TracePoolAllocator` from trace phase to optimized phase on the
  first microbatch.
- It triggers CUDA graph capture and installation when the graph runner is ready.

During graph construction, the capture-time backward post-hook reshards
parameters and releases the temporary full `main_grad` allocation after its
address has been recorded. Gradient reduction remains disabled for capture,
but the allocator lifetime must match eager backward so non-overlapping module
keys can safely share trace-pool slots.
The graph runtime carries the pre-capture static-gradient binding into its
returned-gradient clone decision. It does not refetch `main_grad` after this
release, which would reactivate a trace-pool key outside its traced lifetime.

## State flags

`_fsdp_pre_backward_done` deduplicates pre-backward work for fine-grained hooks:

```text
False at init -> True in mfsdp_pre_backward_setup -> False in final callback
```

`post_backward_issued` tracks whether post-backward reshard and gradient
reduction have run for a module in the current microbatch:

```text
False at init -> False in mfsdp_pre_backward_setup -> True in mfsdp_post_backward_hook
```

Both flags are initialized in `FSDPModule._init_fsdp_state()`.

`model_weight_refresh_pending` tracks an optimizer-boundary weight refresh:

```text
False at init
  -> True in the final callback when is_last_backward=True
  -> False when explicit optimizer integration or the next normal root forward
     calls _copy_main_weights_to_model_weights()
```

Intermediate gradient-accumulation backwards do not arm the flag, and
activation recomputation does not consume it.

## Q&A

### Why do pre hooks accept submodules while post hooks require `FSDPModule`?

Fine-grained pre-forward and pre-backward hooks can be attached to inner
submodules to launch FSDP work earlier. Those calls are resolved back to the
owning FSDP module with `_find_fsdp_target()`.

Post-forward and post-backward work is module-level cleanup: resharding,
gradient reduction, and backward-order advancement. Running that cleanup from
arbitrary submodules would make duplicate or partial cleanup easy to trigger, so
the public post hooks currently require an `FSDPModule`.

### Why is `_fsdp_pre_backward_done` needed?

Fine-grained backward hooks may fire multiple times for different submodules
owned by the same FSDP module. `_fsdp_pre_backward_done` ensures the owning FSDP
module performs backward setup once per microbatch.

### When should `skip_final_callback=True` be used?

Use it only when an external schedule owns the backward lifecycle and will call
`mfsdp_post_backward_final_callback(root_module)` itself. The default
registration path should leave it as `False` so autograd queues the final
callback automatically.

### Why does CUDA-graph input recording only happen for direct `FSDPModule` calls?

The recorded graph boundary is the FSDP module call. A fine-grained submodule
hook can be useful for earlier unshard timing, but it does not provide the full
FSDP-module input/output boundary needed by `CudaGraphRunner`.

### Can these APIs be called during CUDA graph replay?

No. The hooks assert that `ctx.cuda_graph_active` is false. CUDA graph replay
uses the captured graph path and must not re-enter Python hook code.

### Where is the hook execution-order summary?

The lifecycle summary belongs in [`design.md`](design.md). This file documents
what each callable API expects and guarantees.
