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
| `_maybe_free_grad_data()` | `param_group.py` | Free `main_grad_buffer.data` when all params are zero-graded |
| `zero_grad()` | `param_group.py` | Called by `optim.zero_grad()` → triggers `_maybe_free_grad_data()` |
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

    gbuf.init_data(torch.zeros(gbuf.data_size, dtype=gbuf.dtype, device=self.device))

    s = self.sharding_strategy
    is_grad_shard = s in ("optim", "optim_grads", "optim_grads_params")
    placements = [Shard(dim=0)] if is_grad_shard else [Replicate()]

    self.dist_grads = []
    for p, dist_param in zip(self.params, self.dist_params):
        grad_data = gbuf.get_item(self.param_idx[p], as_shard=is_grad_shard)
        if p.requires_grad and grad_data.numel() > 0:
            self.dist_grads.append(
                make_uneven_dtensor(grad_data, p.shape, self.mesh, placements)
            )
        else:
            self.dist_grads.append(None)

    self._grad_buffer_is_fresh = True
```

Key properties:
- **Idempotent**: if `gbuf.data` is already allocated, returns immediately
- **Called from multiple safe points**: backward pre-hook, `reduce_grad()`, post-backward callback
- **Uses `torch.zeros`** (not `torch.empty`): the first reduce-scatter accumulates
  (`overwrite_grad=False`) into the buffer, so it must start from zero — see §5

### 2.3 `_maybe_free_grad_data()` — Memory Refresh

```python
def _maybe_free_grad_data(self) -> None:
    if self.main_grad_buffer is None or self.main_grad_buffer.data is None:
        return
    if any(getattr(p, "grad", None) is not None for p in self.dist_params):
        return
    self.main_grad_buffer.data = None
    self.dist_grads = [None for _ in self.params]
```

Guarded: only frees when no `dist_param` still holds a `.grad` reference.

### 2.4 Call Sites

```
┌──────────────────────────────────────────────────────────────┐
│  _init_buffers()                                             │
│    → main_grad_buffer created but data = None (layout only)  │
│    → dist_grads = [None]                                     │
│                                                              │
│  zero_grad()                          ← optim.zero_grad()    │
│    → dist_param.grad = None                                  │
│    → _maybe_free_grad_data()         ← frees buffer          │
│    → _grad_buffer_is_fresh = True                            │
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

After the first call, `_init_dist_grads()` is a no-op. Subsequent calls just
do the reduce as before. `overwrite_grad=False` ensures proper accumulation
across micro-batches.

### Activation recomputation

During recomputation, the forward pre-hook fires again. `_maybe_free_grad_data()`
is a no-op (data already None or grads are live). The backward pre-hook re-allocates
via `_init_dist_grads()` before backward compute starts — no change in behavior.

### `offload_to_cpu()`

If called before the first `reduce_grad()`, `main_grad_buffer.data` is `None` —
skip. If called after, the buffer exists on GPU and is offloaded normally.
`_ensure_buffers_on_gpu()` auto-reloads on next access.

---

## 5. The `torch.empty` Bug — A Bitter Lesson

### 5.1 What Went Wrong

The initial implementation used `torch.empty()` instead of `torch.zeros()`:

```python
# BROKEN — uninitialised GPU memory
gbuf.init_data(torch.empty(gbuf.data_size, dtype=gbuf.dtype, device=self.device))
```

`torch.empty()` returns a tensor backed by **uninitialised GPU memory** —
arbitrary bit patterns left by prior allocations. `reduce_grad()` then
accumulates into this garbage:

```python
# DataParallelBuffer.reduce_grad()
if overwrite_grad:
    local_grad_shard.copy_(reduced_grad_shard)
elif accumulate_output:
    local_grad_shard += reduced_grad_shard   # ← ADDS to garbage → NaN
```

### 5.2 Why It Hung

1. Micro-batch 1: `+= garbage → NaN`
2. Micro-batches 2–8: `+= NaN → NaN` (NaN propagates)
3. `optimizer.step()` produces NaN weights
4. `_copy_main_params_to_model_params()` copies NaN to model
5. **Hang**: subsequent CUDA operations on NaN tensors stall the GPU or
   trigger NCCL edge-case hangs

### 5.3 The Fix

```python
# CORRECT — zero-initialised
gbuf.init_data(torch.zeros(gbuf.data_size, dtype=gbuf.dtype, device=self.device))
```

`torch.zeros` ensures the first accumulate has a clean slate. Paired with
`_grad_buffer_is_fresh` → `overwrite_grad` for defense-in-depth.

---

## 6. Compatibility with CPU Offload

- `_ensure_buffers_on_gpu()` auto-reloads CPU-offloaded buffers before use
- `_rebuild_dist_views()` re-slices `_local_tensor` views after device move
- `_maybe_free_grad_data()` frees GPU allocation regardless of CPU copy state

Combined: the gradient buffer is **fully elastic** — on GPU only when needed,
on CPU (or nowhere) the rest of the time.

---

## 7. Impact on Callers

| Call site | Impact |
|-----------|--------|
| `reduce_grad()` | Lazy init fires — adds one `torch.zeros` + DTensor rebuild on first call |
| `zero_grad()` | Already guarded — no-op if data is None; frees buffer via `_maybe_free_grad_data` |
| `_rebuild_dist_views()` | Already guarded — no-op for grads if buffer is None |
| `_copy_main_weights_to_model_weights()` | No impact (uses model_weight + main_weight only) |
| `_compute_per_param_norms()` | Reads `dist_grads` — skips if None |
| `finish_grad_sync()` | Only called after `reduce_grad()` → buffer already created |
| Checkpoint `state_dict()` | Optimizer state dict references `dist_grads` → None for lazy groups (correct) |

---

## 8. Related Files

| File | Relevant Changes |
|------|-----------------|
| `param_group.py` | `_init_dist_grads()`, `_maybe_free_grad_data()`, `_rebuild_dist_views()`, `_ensure_buffers_on_gpu()`, lazy `_init_buffers()` |
| `dp_buffer.py` | `_is_on_cpu()`, `_ensure_data_on_gpu()`, `_move_data_to()` |
| `allocator.py` | `release()`, `_auto_resume()`, `resume()` — pool lifecycle |
| `fsdp_module.py` | `zero_grad()` override, `_get_fsdp_modules()`, `offload_to_cpu()`, `reload_to_gpu()` |
| `hooks.py` | `_init_dist_grads()` + `fetch_buffer()` call sites in backward pre-hook |
| `design.md` | Memory-pool release/resume documentation |
| `cpu_offload_design.md` | Full CPU offload architecture |
