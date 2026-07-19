# Lazy main_grad_buffer management in Megatron FSDP v2

`main_grad_buffer` is the `DataParallelBuffer` owned by each `ParameterGroup`
for optimizer-facing gradients.

During backward, Megatron FSDP v2 stages local parameter gradients into this
buffer, runs the required data-parallel collective, and exposes DTensor views of
the reduced result through `dist_param.grad` or `dist_param.decoupled_grad`.
The optimizer consumes those DTensor gradient views.

Lazy management means the buffer layout is created during FSDP initialization,
but the backing tensor is allocated only when gradients are first produced. When
`zero_grad(set_to_none=True)` clears optimizer-facing gradient references, the
backing tensor can be released. When `zero_grad(set_to_none=False)` is used,
the existing backing tensor is zeroed in place and kept allocated.

## Core idea

`ParameterGroup._init_buffers()` creates the `DataParallelBuffer` metadata for
gradients when the parameter group requires gradients:

```python
if self.requires_grad:
    main_grads_dtype = self.mp_policy.main_grads_dtype_for_param(self.params[0])
    self.main_grad_buffer = self._create_buffer(main_grads_dtype, "main_grad")
```

At this point the buffer has layout metadata (`BufferIndex`, shard sizes,
parameter offsets), but `main_grad_buffer.data` is still `None`. The
corresponding `dist_grads` entries are placeholders.

`ParameterGroup._init_dist_grads()` performs the deferred allocation:

1. return immediately if the group has no grad buffer, does not require grads,
   or the buffer is already allocated;
2. allocate `main_grad_buffer.data` with `torch.empty(...)`;
3. slice the buffer according to the active sharding layout;
4. build DTensor gradient views in `dist_grads`.

The DTensor views are what the optimizer later sees through
`dist_param.grad` or `dist_param.decoupled_grad`.

## Normal lifecycle

| Point in step | Behavior |
| --- | --- |
| FSDP initialization | Create `main_grad_buffer` metadata only. `main_grad_buffer.data` is `None`; `dist_grads` contains placeholders. |
| First post-backward `reduce_grad()` | `_init_dist_grads()` allocates `main_grad_buffer.data` and rebuilds `dist_grads` DTensor views. |
| Gradient reduction | `param.grad` is staged into `main_grad_buffer`; all-reduce or reduce-scatter writes the optimizer-facing result. |
| Optimizer step | Optimizer consumes `dist_param.grad` or `dist_param.decoupled_grad`, which are backed by `main_grad_buffer.data`. |
| `zero_grad(set_to_none=True)` | Clear optimizer-facing gradient references, reset accumulation flags, and release `main_grad_buffer.data` if nothing still references valid gradients. |
| `zero_grad(set_to_none=False)` | Keep `main_grad_buffer.data` allocated and zero it in place. |

`_release_grad_storage_if_unused()` is also called from the forward pre-hook.
That call is idempotent and handles the common case where `zero_grad()` has
already cleared all optimizer-facing gradient references before the next
forward.

The normal root forward additionally calls
`FSDPModule._release_grad_storage_if_unused()` before the first parameter
unshard. This root-wide sweep supports plain PyTorch optimizers: their
`zero_grad(set_to_none=True)` clears `dist_param.grad` but does not call the
FSDP module's zero-grad method, leaving parameter-group accumulation flags from
the previous step. When no optimizer-facing gradient is live anywhere in the
FSDP root, the sweep resets those stale flags through
`ParameterGroup.zero_grad()` and releases every eligible grad buffer. If any
gradient remains live, the sweep is a no-op because the model may be between
gradient-accumulation microbatches.

The root-wide sweep is outside the full-iteration CUDA graph lifecycle. When
`enable_full_iteration_cuda_graph=True`, it returns before gradient-liveness
inspection and never calls `ParameterGroup.zero_grad()`. Full-iteration mode
keeps graph-visible gradient objects and owns its in-place zeroing separately.

Doing this at the root boundary is important: the older per-module release
path ran after each module unshard, so later-layer gradient shards could overlap
the next forward's parameter all-gathers and activations. The per-module call
remains as an idempotent fallback for schedules that invoke child FSDP modules
directly.

## Release guard

`_release_grad_storage_if_unused()` frees `main_grad_buffer.data` only when all
of these are true:

- full-iteration CUDA graph mode is not enabled for the group;
- `main_grad_buffer.data` exists;
- the full staging buffer does not contain accumulated gradient data;
- the reduced output buffer does not contain accumulated gradient data;
- no `dist_param.grad` or `dist_param.decoupled_grad` still references the
  gradient DTensor.

If any of those conditions fail, storage is kept because it may still be needed
by gradient accumulation or the optimizer.

## Accumulation flags

Two flags track where valid gradient data currently lives:

- `_full_grad_buffer_has_accumulated_grad` tracks the full `(0, 0)` staging
  buffer used before the collective.
- `_reduced_grad_buffer_has_accumulated_grad` tracks the collective output
  consumed by the optimizer.

`zero_grad()` resets both flags before trying to release storage. `reduce_grad()`
sets them according to the active sharding strategy and whether the collective
consumes the full staging buffer.

These flags are required because some ranks can have empty local optimizer
shards while the shared buffer still contains valid data for another layout.

## Safe use of `torch.empty`

The lazy allocation uses `torch.empty()` to avoid an unnecessary zero-fill.
This is safe because the first use after allocation is controlled by the
accumulation flags:

- if no previous gradient has accumulated, staging and collective outputs
  overwrite the destination;
- if a previous microbatch has accumulated, later microbatches add into the
  existing buffer.

`zero_grad(set_to_none=False)` is the explicit keep-storage path: it zeros
`main_grad_buffer.data` in place when the buffer exists instead of releasing it.

## CUDA graph exceptions

Full-iteration CUDA graph mode keeps optimizer-facing gradient storage alive so
the captured step can reuse stable gradient objects. In that mode,
both the root-wide and per-parameter-group
`_release_grad_storage_if_unused()` paths return without scanning or freeing
the buffer, and the full-iteration optimizer lifecycle clears the existing
storage in place.

Per-module CUDA graph capture keeps the normal lazy behavior, except compatible
main-grad storage may be initialized before capture so trace and replay use the
same buffer surface.

## Relevant code

| File | Relevant pieces |
| --- | --- |
| `param_group.py` | `_init_buffers()`, `_init_dist_grads()`, `_release_grad_storage_if_unused()`, `zero_grad()` |
| `fsdp_module.py` | `reduce_grad()` installs optimizer-facing grads; root-wide `_release_grad_storage_if_unused()` handles plain optimizer zero-grad |
| `hooks.py` | Root-before-unshard and per-module forward pre-hook release paths, plus CUDA-graph pre-initialization |
