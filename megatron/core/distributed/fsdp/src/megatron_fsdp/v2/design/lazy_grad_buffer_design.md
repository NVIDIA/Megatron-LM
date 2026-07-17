# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# Lazy main_grad_buffer Design for Megatron FSDP v2

## 1. Motivation

In the original Megatron FSDP v2, `main_grad_buffer.data` was allocated **eagerly**
during `_init_buffers()` for every param group with `requires_grad=True`, and was
**never freed** for the lifetime of training. The buffer size is `numel × 4 bytes
(FP32) ÷ dp_size` per param group — collectively a significant fraction of GPU memory.

For a 70B BF16 model with `dp=8`, this is **35 GB of permanently-resident GPU memory**
per rank that is only actually needed during the `reduce_grad()` → `optimizer.step()`
window.

Additionally, for large models with frozen sublayers or expert params that
never receive gradients in a given micro-batch, this memory is wasted
permanently.

**Goal**: defer allocation to the first backward pass, and free the buffer between
steps — matching the dynamic memory behavior of PyTorch FSDP2.

### 1.1 Why PyTorch FSDP2 uses less memory

PyTorch's `fully_shard` uses a **dynamic**
strategy for gradient buffers:

- `_lazy_init()` defers allocation until the first backward pass
- After reduce-scatter, the full (unsharded) gradient buffer is immediately freed
- Only the local shard persists — and can be offloaded to CPU between steps

This gives PyTorch FSDP2 a memory advantage during forward, checkpoint I/O,
and model export.

---

## 2. Design

### 2.1 Key Methods

| Method | Location | Role |
|--------|----------|------|
| `_init_dist_grads()` | `param_group.py` | Lazy-allocate `main_grad_buffer.data` and rebuild `dist_grads` DTensors on first use |
| `_release_grad_storage_if_unused()` | `param_group.py` | Free `main_grad_buffer.data` when it has no live gradients |
| `zero_grad()` | `param_group.py` | Called by `optim.zero_grad()` → triggers `_release_grad_storage_if_unused()` |
| `_rebuild_dist_views()` | `param_group.py` | Rebuild `_local_tensor` views after buffer changes device |
| `_ensure_buffers_on_gpu()` | `param_group.py` | Auto-reload any buffer from CPU back to GPU |

### 2.2 `_init_dist_grads()` — Lazy Allocation

```python
def _init_dist_grads(self) -> None:
    gbuf = self.main_grad_buffer
    if gbuf is None or not self.requires_grad:
        return
    if gbuf.data is not None:
        return  # already initialised

    gbuf.init_data(torch.empty(gbuf.data_size, dtype=gbuf.dtype, device=self.device))

    s = self.sharding_strategy
    is_grad_shard = s in ("optim", "optim_grads", "optim_grads_params")
    placements = [Shard(dim=0)] if is_grad_shard else [Replicate()]

    self.dist_grads = []
    for p, dist_param in zip(self.params, self.dist_params):
        grad_data = gbuf.get_item(
            self.param_idx[p], shard_level="inner" if is_grad_shard else "full"
        )
        if p.requires_grad and grad_data.numel() > 0:
            self.dist_grads.append(
                make_uneven_dtensor(grad_data, p.shape, self.mesh, placements)
            )
        else:
            self.dist_grads.append(None)

```

Key properties:
- **Idempotent**: if `gbuf.data` is already allocated, returns immediately
- **Called from multiple safe points**: backward pre-hook, `reduce_grad()`, post-backward callback
- **Uses `torch.empty`**: the constructor/`zero_grad()` state makes the first
  reduce-scatter overwrite uninitialized storage; only later microbatches
  accumulate into valid gradients — see §5

### 2.3 `_release_grad_storage_if_unused()` — Memory Refresh

```python
def _release_grad_storage_if_unused(self) -> None:
    if self.main_grad_buffer is None or self.main_grad_buffer.data is None:
        return
    if (
        self._full_grad_buffer_has_accumulated_grad
        or self._reduced_grad_buffer_has_accumulated_grad
    ):
        return
    if any(getattr(p, "grad", None) is not None for p in self.dist_params):
        return
    self.main_grad_buffer.data = None
    self.dist_grads = [None for _ in self.params]
```

Guarded: only frees when neither the full staging buffer nor the collective
output contains an accumulated gradient and no `dist_param` still holds a
`.grad` reference.

### 2.4 Call Sites

```
┌──────────────────────────────────────────────────────────────┐
│  _init_buffers()                                             │
│    → main_grad_buffer created but data = None (layout only)  │
│    → dist_grads = [None]                                     │
│                                                              │
│  zero_grad()                          ← optim.zero_grad()    │
│    → dist_param.grad = None                                  │
│    → _full_grad_buffer_has_accumulated_grad = False          │
│    → _reduced_grad_buffer_has_accumulated_grad = False       │
│    → _release_grad_storage_if_unused() ← frees buffer        │
│                                                              │
│  backward_pre_hook                                            │
│    → _init_dist_grads()              ← re-allocates buffer   │
│    → fetch_buffer()                  ← allocates full buf    │
│                                                              │
│  reduce_grad()                                               │
│    → _ensure_buffers_on_gpu()        ← auto-reload safety    │
│    → _init_dist_grads()              ← no-op (already init)  │
│    → main_grad_buffer.reduce_grad()  ← reduce-scatter        │
│                                                              │
│  finish_grad_sync()                                          │
│    → param.main_grad = dist_grad                             │
│                                                              │
│  optim.step()                                                │
│    → copies main_grad → optimizer param                      │
│    → updates weights → copies back to model                  │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. Memory Lifecycle

```
Memory
  ^
  │                         ┌──────────────────────┐
  │                         │ _init_dist_grads()   │  ┌──────────────────────┐
  │  ┌──────────────────────┤ allocate grad buffer ├──┤ _init_dist_grads()   │
  │  │  _maybe_free_grad_   │ (torch.zeros)        │  │ re-allocate          │
  │  │  data() frees buffer │                      │  │                      │
  │  ├──────────────────────┘                      │  ├──────────────────────┘
  │  │  ┌──────────┐         ┌──────────┐          │  │  ┌──────────┐
  │  │  │  model   │         │  model   │          │  │  │  model   │
  │  │  │  weight  │         │  weight  │          │  │  │  weight  │
  │  │  │  buffer  │         │  buffer  │          │  │  │  buffer  │
  └──┴──┴──────────┴─────────┴──────────┴──────────┴──┴──┴──────────┴──────► time
       init    fwd      bwd      optim    zero_grad   fwd    bwd    optim

    │←── step N ──→│                                         │←── step N+1 ──→│

    Gradient buffer: ALLOCATED only during backward → optimizer window
                     FREED during forward and between steps
```

### 3.1 Savings Table

| Phase | Before (eager) | After (lazy) | Delta |
|-------|---------------|--------------|-------|
| Init / construction | gradient shard on GPU | 0 bytes | `−model_params × 4 / dp` |
| Forward | gradient shard on GPU | FREE | `−model_params × 4 / dp` |
| Backward | gradient shard on GPU | gradient shard on GPU | 0 |
| Optimizer step | gradient shard on GPU | gradient shard on GPU | 0 |
| Between steps | gradient shard on GPU | FREE | `−model_params × 4 / dp` |
| Checkpoint I/O | gradient shard on GPU | FREE | freed for I/O buffers |
| Model export | gradient shard on GPU | FREE | freed for export |

**Example** (70B BF16 model, `dp=8`):
- Gradient shard: `70B × 4 bytes ÷ 8 = 35 GB` (FP32)
- **35 GB freed** during forward, between steps, and checkpoint I/O

---

## 4. Edge Cases

### Frozen params (`requires_grad=False`)

`main_grad_buffer` is never created — the existing `if self.requires_grad` guard
in `_init_buffers` already handles this. No change.

### Param groups with no gradient flow

Some param groups (e.g., expert params on ranks that don't own the expert) may
never see a backward pass. With lazy init, the buffer is never allocated —
saves memory permanently.

### Multiple `reduce_grad()` calls (gradient accumulation)

After the first call, `_init_dist_grads()` is a no-op. Two flags track the
independent accumulation locations:

- `_full_grad_buffer_has_accumulated_grad` controls whether staging adds to or
  overwrites the full `(0, 0)` gradient buffer.
- `_reduced_grad_buffer_has_accumulated_grad` controls whether a new collective
  result adds to or overwrites the local reduced output.

A reduce-scatter consumes the full input and clears the first flag. An
all-reduce operates in place, so the full-buffer flag remains set. Any completed
collective sets the reduced-output flag.

| Inner strategy | After a non-final microbatch | After the final microbatch |
| --- | --- | --- |
| `no_shard` | full=`True`, reduced=`False` | full=`True`, reduced=`True` |
| `optim` | full=`True`, reduced=`False` | full=`False`, reduced=`True` |
| `optim_grads` | full=`False`, reduced=`True` | full=`False`, reduced=`True` |
| `optim_grads_params` | full=`False`, reduced=`True` | full=`False`, reduced=`True` |

### Full-iteration CUDA graph

`enable_full_iteration_cuda_graph=True` is an explicit exception to the normal
lazy-freeing policy for optimizer-facing gradients. The full-iteration wrapper
captures forward/backward but runs the optimizer outside the graph, so the local
gradient shard and `decoupled_grad` object must keep stable identities.

- `_pre_backward_setup()` allocates dist grads before capture.
- `_release_grad_storage_if_unused()` keeps optimizer-facing gradient storage alive.
- `zero_grad()` keeps optimizer-facing objects and clears local storage in place.
- Optimizer zero-grad keeps marked `grad`/`decoupled_grad` DTensors and zeroes
  their local storage.
- Full unsharded weight and gradient buffers remain transient. They allocate and
  reshard inside capture, so the CUDA graph private pool owns their stable replay
  addresses and reuses non-overlapping lifetimes.

Both flags default to `False`, so eager and per-module CUDA graph paths retain
their normal lazy allocation/free behavior.

### Activation recomputation

During recomputation, the forward pre-hook fires again. `_release_grad_storage_if_unused()`
is a no-op (data already None or grads are live). The backward pre-hook re-allocates
via `_init_dist_grads()` before backward compute starts — no change in behavior.

### `offload_to_cpu()`

If called before the first `reduce_grad()`, `main_grad_buffer.data` is `None` —
skip. If called after, the buffer exists on GPU and is offloaded normally.
`_ensure_buffers_on_gpu()` auto-reloads on next access.

---

## 5. Safe `torch.empty` Allocation

### 5.1 First-Write Semantics

The gradient buffer uses `torch.empty()` to avoid an unnecessary zero-fill:

```python
gbuf.init_data(torch.empty(gbuf.data_size, dtype=gbuf.dtype, device=self.device))
```

`torch.empty()` returns uninitialized memory, so the first reduced gradient
must overwrite the local shard. Later microbatches may accumulate:

```python
# DataParallelBuffer.reduce_grad()
if accumulate_reduced_grad:
    local_grad_shard += reduced_grad_shard
else:
    local_grad_shard.copy_(reduced_grad_shard)
```

`_reduced_grad_buffer_has_accumulated_grad=False` selects overwrite for the
first collective output. `reduce_grad()` then sets it to `True`, selecting
accumulation for later microbatches. `zero_grad()` resets both accumulation
flags to `False`.

---

## 6. Compatibility with CPU Offload

- `_ensure_buffers_on_gpu()` auto-reloads CPU-offloaded buffers before use
- `_rebuild_dist_views()` re-slices `_local_tensor` views after device move
- `_release_grad_storage_if_unused()` frees GPU allocation regardless of CPU copy state

Combined: the gradient buffer is **fully elastic** — on GPU only when needed,
on CPU (or nowhere) the rest of the time.

---

## 7. Impact on Callers

| Call site | Impact |
|-----------|--------|
| `reduce_grad()` | Lazy init fires — adds one `torch.zeros` + DTensor rebuild on first call |
| `zero_grad()` | Already guarded — no-op if data is None; frees buffer via `_release_grad_storage_if_unused` |
| `_rebuild_dist_views()` | Already guarded — no-op for grads if buffer is None |
| `_copy_main_weights_to_model_weights()` | No impact (uses model_weight + main_weight only) |
| `_compute_per_param_norms()` | Reads `dist_grads` — skips if None |
| `finish_grad_sync()` | Only called after `reduce_grad()` → buffer already created |
| Checkpoint `state_dict()` | Optimizer state dict references `dist_grads` → None for lazy groups (correct) |

---

## 8. Related Files

| File | Relevant Changes |
|------|-----------------|
| `param_group.py` | `_init_dist_grads()`, `_release_grad_storage_if_unused()`, `_rebuild_dist_views()`, `_ensure_buffers_on_gpu()`, lazy `_init_buffers()` |
| `dp_buffer.py` | `_is_on_cpu()`, `_ensure_data_on_gpu()`, `_move_data_to()` |
| `allocator.py` | `release()`, `_auto_resume()`, `resume()` — pool lifecycle |
| `fsdp_module.py` | `zero_grad()` override, `_get_fsdp_modules()`, `offload_to_cpu()`, `reload_to_gpu()` |
| `hooks.py` | `_init_dist_grads()` + `fetch_buffer()` call sites in backward pre-hook |
| `design.md` | Memory-pool release/resume documentation |
| `cpu_offload_design.md` | Full CPU offload architecture |
