# Silent CUDA Graph inside Megatron FSDP v2 — Design

> **Experimental** — CUDA graph support in Megatron FSDP v2 is an experimental
> feature.  The API and behaviour may change in future releases without notice.

## Acknowledgements

Built on [te-graph-runtime](https://github.com/buptzyb/te-graph-runtime) by
[@buptzyb](https://github.com/buptzyb) (Robin Zhang) — a standalone
TE-compatible `make_graphed_callables` with `capture_time_hooks` support.
Vendored under `te_graph_runtime/` with local modifications for FSDP v2.

## 1. Motivation

MCore's CUDA graph system was designed for DDP's memory model — each layer
receives freshly-allocated tensor inputs/outputs.  Megatron FSDP v2 shares
pool-backed buffers across layers, and FSDP hooks (unshard/reshard) are not
part of the graphed region.

This system uses `te-graph-runtime`'s `make_graphed_callables` with
`capture_time_hooks` to run FSDP unshard/reshard **outside** CUDA graph
capture — they are never captured or replayed.  A single batch call captures
all eligible modules in forward execution order with a shared memory pool.

## 2. One knob

```python
fully_shard(module, enable_cuda_graph=True)
```

## 3. Architecture

### 3.0 Why te-graph-runtime instead of torch.cuda.make_graphed_callables?

| Feature | `torch.cuda.make_graphed_callables` | `te-graph-runtime` |
|---------|-------------------------------------|-------------------|
| **Module hooks** | Asserts modules have NO hooks at capture time | `capture_time_hooks` — hooks that run outside graph capture, not registered on modules |
| **Keyword args** | Positional tensor args only | `sample_kwargs` passes keyword tensor args natively |
| **TE/FP8** | No TE-specific support | TE FP8/FP4 recipes, amax reduction, delayed wgrad, RNG tracker state |
| **Parameter grad lifetime** | Standard PyTorch lifetime | TE PR #2937 grad lifetime fix |

FSDP v2 modules require unshard/reshard hooks around every forward/backward —
even during warmup and capture.  PyTorch's version rejects modules with hooks.
`te-graph-runtime`'s `capture_time_hooks` runs these essential operations
outside the CUDA graph region without violating the no-hooks assertion.

### 3.1 te-graph-runtime

`make_graphed_callables` wraps each module with a `Graphed` autograd Function
that replays forward and backward CUDA graphs.  It handles warmup, capture
order (fwds in forward-module order, bwds in reverse), shared memory pool, and
autograd wiring internally.

**Warmup and capture share the same CUDA stream for activation reuse** — intermediate
tensors (cuDNN/cuBLAS workspaces, attention intermediates) allocated during
warmup stay at the same addresses in the same CUDA context, so the capture
phase reuses them instead of freeing and reallocating.  This saves significant
GPU memory vs. a throwaway warmup stream that would fragment the caching
allocator with stale allocations.

Key features we rely on:

- **`capture_time_hooks`** — hook functions that run during warmup + capture
  but are NOT captured into the CUDA graph.  Used for FSDP unshard/reshard.
- **`sample_kwargs`** — keyword tensor arguments passed natively to modules,
  avoiding positional shim complexity.
- **`pool`** — shared `graph_pool_handle` for all modules.
- **`capture_stream`** — optional parameter to serialize captures on the same
  stream (important for shared-pool correctness).

### 3.2 CudaGraphRunner

A lightweight orchestrator stored on `ctx.cuda_graph_runner`:

```python
class CudaGraphRunner:
    def record_module(self, module, args, kwargs):
        """Record sample args during the first optimized forward."""
    def capture_and_install(self, root_module, capture_stream=None):
        """Pop hooks → make_graphed_callables → restore hooks."""
```

### 3.3 capture_time_hooks

Four minimal hooks attached via `capture_time_hooks` to each module:

| Hook | Trigger | Action |
|------|---------|--------|
| `forward_pre_hooks_with_kwargs` | Before warmup/capture forward | `module.unshard()` |
| `forward_hooks_with_kwargs` | After warmup/capture forward | `module.reshard()` |
| `backward_pre_hooks` | Before warmup/capture backward | `module.unshard(bwd_pass=True)` |
| `backward_hooks` | After warmup/capture backward | `module.reshard()` + clear `param.grad` |

These run outside `torch.cuda.graph()` — unshard/reshard are never captured.

## 4. Lifecycle

```
╔═══════════════════════════════════════════════════════════════╗
║ Stage 1 — Microbatch 0 (trace)                                ║
║                                                                ║
║  forward:   _trace_allocate / _trace_free → trace recorded    ║
║  backward:  _trace_allocate / _trace_free → trace continues   ║
║  plan():    pool built, phase="optimized"                      ║
║  snapshot_slots():  freeze slot state for replay              ║
╚═══════════════════════════════════════════════════════════════╝
                          │
╔═══════════════════════════════════════════════════════════════╗
║ Stage 2 — Microbatch 1 (record sample args + eager)           ║
║                                                                ║
║  root_forward_pre_hook:                                        ║
║    Creates shared graph_pool_handle + capture_stream            ║
║    reset_cursor(); restore_slots()                             ║
║                                                                ║
║  forward (per graphed FSDPModule):                             ║
║    forward_pre_hook: unshard params                            ║
║    CudaGraphRunner.record_module() — records sample kwargs     ║
║    module.forward() runs eagerly                               ║
║    post_forward_hook: reshard                                  ║
║                                                                ║
║  backward: eager                                                ║
║    pre_backward: unshard (bwd_pass)                            ║
║    post_backward: reshard + reduce_grad                        ║
║                                                                ║
║  post_backward_final_callback:                                 ║
║    trace → optimized transition (plan)                         ║
║    _maybe_capture_cuda_graphs()                                ║
║      → CudaGraphRunner.capture_and_install()                   ║
║        1. Clone sample kwargs (fresh leaves)                   ║
║        2. Pop all FSDP hooks from module tree                  ║
║        3. make_graphed_callables(tuple(modules), ...)          ║
║           → warmup + forward capture + backward capture        ║
║           → replaces module.forward with graphed version       ║
║        4. Restore FSDP hooks                                   ║
║        5. Mark modules as _fsdp_cg_installed                   ║
╚═══════════════════════════════════════════════════════════════╝
                          │
╔═══════════════════════════════════════════════════════════════╗
║ Stage 3 — Microbatch 2+ (replay)                               ║
║                                                                ║
║  root_forward_pre_hook:                                        ║
║    reset_cursor(); restore_slots()                             ║
║                                                                ║
║  forward (per graphed FSDPModule):                             ║
║    forward_pre_hook: unshard; _fsdp_cg_installed → skip record ║
║    module.forward (graphed) → Graphed.apply → fwd_graph.replay ║
║    post_forward_hook: reshard                                  ║
║                                                                ║
║  backward:                                                      ║
║    pre_backward: unshard (bwd_pass)                            ║
║    Graphed.backward → bwd_graph.replay                         ║
║    post_backward: reshard + reduce_grad                        ║
╚═══════════════════════════════════════════════════════════════╝
```

## 5. Hook strategy

### 5.1 During capture (`make_graphed_callables`)

Real FSDP hooks are **popped** from the entire module tree before calling
`make_graphed_callables` (it asserts modules have no hooks).  `capture_time_hooks`
handle unshard/reshard during warmup, forward capture, and backward capture.

### 5.2 During replay

Real FSDP hooks are **restored** after capture.  They fire normally around
`module.__call__`:

- `forward_pre_hook` → unshard
- `module.forward` (graphed) → fwd replay
- `forward_hook` → reshard
- `backward_pre_hook` → unshard (bwd_pass)
- `Graphed.backward` → bwd replay
- `backward_hook` → reshard + reduce_grad

## 6. Capture stream & shared pool

```python
# hooks.py — root forward pre-hook
ctx.cuda_graph_stream = torch.cuda.Stream()
ctx.cuda_graph_pool = torch.cuda.graph_pool_handle()

# CudaGraphRunner passes them to make_graphed_callables:
make_graphed_callables(
    modules,
    sample_args,
    pool=ctx.cuda_graph_pool,
    capture_stream=ctx.cuda_graph_stream,
)
```

## 7. Parameter gradients

`make_graphed_callables` includes module parameters in the autograd surface
(via `Graphed.apply(*(user_args + module_params))`).  Parameter gradients are
computed during backward capture and returned by `Graphed.backward`.  Autograd
sets `param.grad`; FSDP's post-backward hook (`reduce_grad`) consumes them via
`param.main_grad`.  No manual gradient buffer management needed.

## 8. te-graph-runtime local modifications

Vendored at `te_graph_runtime/`.  Key local changes from upstream:

- **Non-tensor `sample_kwargs`**: `None`-safe guards at 6 access points in
  warmup, backward capture, and `Graphed.forward`.
- **Positional arg replay**: `functionalized` reconstructs capture-time arg
  order from both `user_args` and `user_kwargs`.
- **Memory cleanup**: `gc.collect()` + `torch.cuda.empty_cache()` between
  warmup and capture.
- **Stream management**: optional `capture_stream` parameter; warmup uses
  separate throwaway stream for `torch.compile` compatibility.
- **`param.grad` cleanup**: cleared after each backward hook iteration to
  prevent gradient accumulation memory leaks.

See `te_graph_runtime/README.md` for details and upstream attribution.

## 9. Known issues & fixes

| Issue | Root cause | Fix | Date |
|-------|-----------|-----|------|
| Convergence degradation | Missing `register_generator_state` on graphs — dropout masks baked | `register_generator_state` on both fwd/bwd graphs | 2026-06 |
| `cudaErrorStreamCaptureInvalidated` with `torch.compile` | Shared warmup/capture stream; compile recompilation breaks capture | Separate throwaway warmup stream (workaround; ideal: keep shared stream and fix compile guard mismatch) | 2026-06 |
| `cudaErrorStreamCaptureUnsupported` during compile | `autocast(enabled=False)` triggered guard mismatch and recompilation inside graph | `autocast(cache_enabled=False)` preserves bf16 autocast | 2026-06 |
| OOM during warmup | `warmup_outputs` held tensor refs across `gc.collect` + `empty_cache` | `param.grad = None` prevents gradient accumulation across warmup iterations | 2026-06 |
| Non-tensor kwargs crash `.requires_grad` | `tree_flatten` passes `None` into `static_input_surface` | 6 `is not None` / `isinstance` guards in te-graph-runtime | 2026-06 |
| Positional `hidden_states` missing in replay | `kwargs_keys` validation rejected positional args | `functionalized` checks both `user_args` and `user_kwargs` | 2026-06 |

## 10. Files

| File | Role |
|------|------|
| `cuda_graph_runner.py` | `CudaGraphRunner` — sample arg recording, batch capture orchestration |
| `hooks.py` | Capture trigger, hook save/restore, shared pool+stream creation |
| `fsdp_module.py` | `_FSDPRootContext.cuda_graph_runner` / `cuda_graph_stream` / `cuda_graph_pool` |
| `te_graph_runtime/` | Vendored `make_graphed_callables` with local modifications |
| `design/cuda_graph_design.md` | This document |

## 11. No user-visible knobs

| What happens | User action |
|---|---|
| Trace collection | Automatic (MB0) |
| Pool planning | Automatic (MB0 post-backward) |
| Sample arg recording | Automatic (MB1 forward pre-hook) |
| Graph capture | Automatic (MB1 post-backward, via `make_graphed_callables`) |
| Graph replay | Automatic (MB2+, via `Graphed.apply`) |
| RNG state | Handled by `make_graphed_callables` internally |
| Slot state | Automatic (snapshot/restore) |
