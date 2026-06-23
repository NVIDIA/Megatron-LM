# Megatron FSDP v2 Hooks API

The public API surface of `hooks.py` — functions callable by external code
(e.g. the 1F1B EP overlap schedule) for manual FSDP lifecycle management.

All functions live in `megatron.core.distributed.fsdp.src.megatron_fsdp.v2.hooks`.

---

## Target resolution

### `_find_fsdp_target(hook_module) → Optional[FSDPModule]`

Resolve the nearest parent FSDPModule for any module.

| Condition | Returns |
|---|---|
| `isinstance(hook_module, FSDPModule)` | `hook_module` itself |
| `hasattr(hook_module, '_fsdp_parent_module')` | `hook_module._fsdp_parent_module` |
| Otherwise | `None` |

`_fsdp_parent_module` is set on every non-FSDPModule sub-module during
`FSDPModule._init_fsdp_state()` (bottom-up pass, nearest ancestor wins).

---

## Forward hooks

### `mfsdp_forward_pre_hook(hook_module, args, kwargs)`

Pre-forward: unshard parameters for the target FSDPModule.

- Resolves target via `_find_fsdp_target`.  No-op if `None`.
- Root phase: sets `ctx.forward_phase = True`, creates CG stream lazily.
- Unshards parameters (forward + optional backward pass).
- Frees stale grad data.
- **CUDA graph capture**: only when `isinstance(hook_module, FSDPModule)` (skipped for sub-modules).

Accepts any module.  Sub-modules delegate to parent via `_fsdp_parent_module`.

**Repeatability**: This function may be called multiple times for the same target
when fine-grained hooks are active (e.g. sub-module forward triggers the hook,
followed by the enclosing FSDPModule's forward also triggering it).  The
implementation MUST be safe to invoke repeatedly without observable overhead —
duplicate ``unshard()`` calls and idempotent bookkeeping must be effectively
free.

### `mfsdp_post_forward_hook(module, *unused)`

Post-forward: reshard parameters.

- **Validates**: `TypeError` if `not isinstance(module, FSDPModule)`.
- Skips if `ctx.backward_phase` and this module is the current backward module
  (activation recomputation: don't reshard before backward needs params).
- `module.reshard()`.

---

## Backward hooks

### `mfsdp_pre_backward_setup(hook_module, grads)`

Pre-backward: root phase transition, unshard, TE gradient-fusion flags.

- Resolves target via `_find_fsdp_target`.  No-op if `None`.
- **Deduplication**: skips if `target._fsdp_pre_backward_done` is `True`.
- Root setup (first call):
  - `ctx.backward_done_modules.clear()`
  - `ctx.forward_phase = False`, `ctx.backward_phase = True`
  - `ctx._advance_backward_module()`
  - Enqueues `_register_post_backward_final_callback` (once).
- `target.unshard(bwd_pass=True)`.
- `target.post_backward_issued = False`.
- TE gradient-fusion: resets `grad_added_to_main_grad`, sets `overwrite_main_grad` for sharding strategies.
- `target._fsdp_pre_backward_done = True`.

Accepts any module.  Compatible with `register_multi_grad_hook` callback signature
`(module, grads)`.

### `mfsdp_post_backward_hook(module)`

Post-backward: mark module done, reshard, reduce gradients.

- **Validates**: `TypeError` if `not isinstance(module, FSDPModule)`.
- `ctx.backward_done_modules.add(id(module))`.
- `ctx._advance_backward_module()`.
- `module.reshard()`.
- If any param group uses `optim_grads` or `optim_grads_params`:
  `module.reduce_grad(async_op=ctx.enable_async_reduce_grad)`.
- `module.post_backward_issued = True`.

### `mfsdp_post_backward_final_callback(root_module)`

Finalise the backward pass for one micro-batch.

- **Validates**: `TypeError` if not FSDPModule; `RuntimeError` if not root.
- Handles modules whose per-module post-backward was skipped (e.g. activation
  recomputation): `reshard()` + `reduce_grad()`.
- Drains pending async reduce-grad events.
- Resets root context: `_post_backward_callback_queued`, `backward_phase`,
  `backward_module`, `backward_done_modules`.
- Clears `_fsdp_pre_backward_done` on all FSDP modules in `forward_order`.
- First micro-batch only: transitions `TracePoolAllocator` from trace to
  optimized plan.

---

## Flag lifecycle

```
_fsdp_pre_backward_done:
    False (init) ─► True (mfsdp_pre_backward_setup) ─► False (mfsdp_post_backward_final_callback)

post_backward_issued:
    False (init) ─► False (mfsdp_pre_backward_setup) ─► True (mfsdp_post_backward_hook)
```

All flags are initialised to `False` in `FSDPModule._init_fsdp_state()`.

---

## Integration with the auto-hook path

The registration functions (`_register_*`) use the public functions internally:

| Registration function | Uses |
|---|---|
| `_register_forward_pre_hook` | `mfsdp_forward_pre_hook` |
| `_register_forward_hook` | `mfsdp_post_forward_hook` |
| `_register_backward_pre_hook` | `mfsdp_pre_backward_setup` (via `_create_custom_backward_hook`) |
| `_register_backward_pre_hook_fine_grained` | `mfsdp_pre_backward_setup` (via `_create_custom_backward_hook`) |
| `_register_backward_hook` | `mfsdp_post_backward_hook` (via `RegisterFSDPBackwardFunction`) |
| `_register_post_backward_final_callback` | `mfsdp_post_backward_final_callback` (via `Variable._execution_engine.queue_callback`) |

This means the auto-hook path and the manual (1F1B overlap schedule) path share
identical logic — they just differ in *when* the functions are called.
