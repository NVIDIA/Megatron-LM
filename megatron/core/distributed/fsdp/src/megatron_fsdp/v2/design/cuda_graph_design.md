# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# Silent CUDA Graph inside Megatron FSDP v2 — Design

> **Experimental** — CUDA graph support in Megatron FSDP v2 is an experimental
> feature.  The API and behaviour may change in future releases without notice.

## Acknowledgements

Built on [te-graph-runtime](https://github.com/buptzyb/te-graph-runtime) by
[@buptzyb](https://github.com/buptzyb) (Robin Zhang) — a standalone
TE-compatible `make_graphed_callables` with `capture_time_hooks` support.
Vendored under `te_graph_runtime/` with local modifications for FSDP v2.

> **Provenance.**  `te_graph_runtime/graph.py` is itself derived from
> TransformerEngine's own graph module
> (`transformer_engine/pytorch/graph.py`) — it is the *same* `make_graphed_callables`
> implementation, extracted into a standalone package and extended with
> `capture_time_hooks`.  Treat it as a fork of the TE source: keep it in sync
> with upstream TE and prefer upstreaming fixes back rather than letting the two
> copies diverge.

## 1. Motivation

**The core conflict: a CUDA graph is static, but FSDP parameters are not.**
A captured CUDA graph bakes in fixed memory addresses and a fixed sequence of
kernels — replay reads and writes exactly the buffers that were live at capture
time.  FSDP, by design, does the opposite to every parameter on every
micro-batch:

- **`unshard`** all-gathers the sharded parameter into a *temporary* unsharded
  buffer immediately before the layer runs, and
- **`reshard`** frees that buffer immediately after, returning to the sharded
  state.

So a parameter's full, usable form only exists transiently, and — with a naive
allocator — at a *different address* each time it is re-materialized.  Capturing
the forward/backward of an FSDP layer with a stock
`torch.cuda.make_graphed_callables` is therefore impossible on two counts:

1. **Moving memory.**  The graph would hard-code the capture-time unshard
   buffer address; the next micro-batch resharded and re-gathered the parameter
   elsewhere, so replay would read stale or freed memory.
2. **Hooks in the captured region.**  The unshard/reshard (and grad-reduce)
   work is driven by module hooks that launch collectives and allocate/free
   buffers — none of which can legally run inside `torch.cuda.graph()` capture,
   and none of which should be *replayed* as static kernels anyway.

**How this design resolves it.**  Two pieces make FSDP layers capturable:

- The [`TracePoolAllocator`](../allocator.py) pins each parameter's unshard
  buffer to a **fixed address** across micro-batches (see README §"Why MFSDP v2
  can support CUDA graphs").  This removes the moving-memory problem — every
  replay finds the weights exactly where capture left them, *provided* the hooks
  re-gather them into that same slot before replay.
- `te-graph-runtime`'s `make_graphed_callables` with **`capture_time_hooks`**
  runs FSDP `unshard`/`reshard` **outside** the graphed region — they execute
  during warmup and capture to place/free the weights, but are never recorded
  into the graph and never replayed.  The graph captures only the layer's math;
  memory movement stays in Python.

A single batched call then captures every eligible module in forward-execution
order into one shared memory pool.  MCore's existing CUDA graph system does not
help here: it was built for DDP's memory model, where each layer receives
freshly-allocated inputs/outputs and parameters are always resident, so it never
had to reconcile capture with per-layer reshard.

## 2. One knob

```python
fully_shard(module, enable_cuda_graph=True)
```

## 3. Workflow at a glance

CUDA graph capture rides on the `TracePoolAllocator`'s existing
trace → plan → optimized progression, adding just a *record* step and a
*capture* step.  End to end it spans three micro-batches — the same three
stages §5 later walks through hook-by-hook:

- **Stage 1 · MB0 — trace.**  Forward/backward run fully eagerly while the
  allocator records every buffer alloc/free; `plan()` then pins each buffer to a
  **fixed address**.  Nothing is captured yet — those fixed addresses are the
  precondition that makes capture legal at all (see §1).
- **Stage 2 · MB1 — record + capture.**  With addresses now stable, the forward
  records sample inputs per eligible layer, and the backward's final callback
  captures forward+backward graphs for all of them in one batched call — then
  swaps each layer's `forward` for a graph-replaying version.
- **Stage 3 · MB2+ — replay.**  Each layer's forward/backward is a graph replay.
  FSDP's real `unshard` / `reshard` / `reduce_grad` hooks still fire *around* the
  replay — landing weights at the fixed addresses before, consuming gradients
  after — but never execute inside the graph.

From here the document zooms in: §4 describes the components that implement
these stages, §5 expands the same three stages with exact hook ordering, and §6
contrasts hook behavior during capture vs. replay.

## 4. Architecture

### 4.0 Why te-graph-runtime instead of torch.cuda.make_graphed_callables?

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

### 4.1 te-graph-runtime

`make_graphed_callables` wraps each module with a `Graphed` autograd Function
that replays forward and backward CUDA graphs.  It handles warmup, capture
order (fwds in forward-module order, bwds in reverse), shared memory pool, and
autograd wiring internally.

> **Shared lineage with TransformerEngine.**  This is the same
> `make_graphed_callables` implementation that TransformerEngine ships and uses
> for its own CUDA graph capture — `te-graph-runtime` is a standalone extract of
> it (plus `capture_time_hooks`).  Behavior, argument surface, and quirks
> therefore track TE's graphed-callable path; fixes here should be kept in sync
> with (and ideally upstreamed to) TE to avoid divergence.

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

### 4.2 CudaGraphRunner

A lightweight orchestrator stored on `ctx.cuda_graph_runner`:

```python
class CudaGraphRunner:
    def record_module(self, module, args, kwargs):
        """Record sample args during the first optimized forward."""
    def record_module_output(self, module, output):
        """Record eager outputs used to link producer and consumer buffers."""
    def capture_and_install(self, root_module, capture_stream=None):
        """Pop hooks → make_graphed_callables → restore hooks."""
```

### 4.3 capture_time_hooks

Four minimal hooks attached via `capture_time_hooks` to each module:

| Hook | Trigger | Action |
|------|---------|--------|
| `forward_pre_hooks_with_kwargs` | Before warmup/capture forward | `module.unshard()` |
| `forward_hooks_with_kwargs` | After warmup/capture forward | `module.reshard()` |
| `backward_pre_hooks` | Before warmup/capture backward | `module.unshard(bwd_pass=True)` |
| `backward_hooks` | After warmup/capture backward | `module.reshard()` + clear `param.grad` |

These run outside `torch.cuda.graph()` — unshard/reshard are never captured.

### 4.4 Inside `make_graphed_callables` (deep dive)

`make_graphed_callables` is the single most important dependency of this
system, and the correctness of the FSDP integration rests on its exact
contract.  This subsection documents what it does, the invariants it assumes,
and where the FSDP integration must cooperate.  Line references are to
`te_graph_runtime/graph.py`.

#### 4.4.1 Signature and return value

```python
graphed = make_graphed_callables(
    modules,                # tuple(nn.Module) — captured in this order
    sample_args,            # tuple(tuple)     — positional sample tensors per module
    num_warmup_iters=3,
    sample_kwargs=...,      # tuple(dict)      — keyword sample tensors per module
    pool=...,               # shared graph_pool_handle
    capture_time_hooks=..., # per-module unshard/reshard hooks (not captured)
    capture_stream=...,     # stream shared by warmup + capture
)
```

For each module it returns a `functionalized` callable and **replaces
`module.forward` in place** with it (`graph.py` builds one per callable in the
final loop).  After installation, calling `module(...)` runs the graphed path
instead of the eager forward.

#### 4.4.2 The static input surface

For every callable the "static input surface" is

```
static_input_surface = flatten(sample_args) + flatten(sample_kwargs) + tuple(module.parameters())
```

(`graph.py:686-688`).  Two consequences drive the whole design:

1. **Parameters are part of the graph's input surface.**  Their gradients are
   produced by the captured backward and returned from `Graphed.backward`, so
   FSDP needs no manual param-grad buffer plumbing (see §8).
2. **Only the *user* args are copied in at replay** — not the parameters.
   `Graphed.forward` copies incoming tensors into the static buffers for
   `i in range(len_user_args)` and *only* when `data_ptr()` differs
   (`graph.py:1351-1357`).  Module parameters are never copied; the graph reads
   them **at the addresses they occupied during capture**.  This is the exact
   reason the FSDP integration requires the `TracePoolAllocator`: `unshard()`
   must place each parameter back at the same fixed address on every
   micro-batch, or replay reads stale/garbage weights.

M-FSDP swaps the registered sharded parameter for its live unsharded compute
leaf inside the capture-time backward pre-hook.  The runtime therefore refreshes
`module.parameters()` immediately after that hook and carries the retained
parameter indices from warmup into capture.  This prevents the graph from
capturing stale `dist_param` objects.  The runner also records eager module
outputs and links matching producer-output/consumer-input storage so adjacent
captured modules share the correct static activation and gradient surfaces.

#### 4.4.3 Phases

| Phase | What happens | Where |
|-------|--------------|-------|
| **Warmup** | Runs each callable `num_warmup_iters` times (default 3) on `capture_stream`, forward + backward, to flush lazy init (cuDNN benchmarking, cuBLAS/attention workspaces, TE module discovery) out of the captured region.  `capture_time_hooks` fire here, so params are unsharded for warmup. | `graph.py:923-1003` |
| **Forward capture** | For each callable **in forward order**, run `func(*args, **kwargs)` under `torch.cuda.graph(pool, stream)`; record `static_outputs`. | `graph.py:1240-1254` |
| **Backward capture** | For each callable **in reverse order**, run `torch.autograd.backward` under graph capture with pre-allocated `static_grad_outputs`; record `static_grad_inputs` (including param grads). | `graph.py:1256-1312` |

Warmup and capture deliberately share one stream so intermediate workspace
tensors keep the same addresses and are reused rather than reallocated (§4.1).

#### 4.4.4 The `Graphed` autograd Function

Replay is wrapped in a `torch.autograd.Function` (`graph.py:1335-1426`):

- **`forward`**: copy changed user inputs into the static surface → replay
  `fwd_graph` (optionally on a side stream with event sync-back) → return
  `static_outputs.detach()`.
- **`backward`** (`@once_differentiable`): copy incoming grads into
  `static_grad_outputs` (skipping when the pointer already matches) → replay
  `bwd_graph` → return `static_grad_inputs`.  Compatible M-FSDP parameter leaves
  bind directly to their `main_grad` storage during capture.  Those slots are
  returned as detached views, while parameter grads backed by internal graph
  storage are cloned before returning so autograd consumers never retain the
  graph's private buffers.

`functionalized` (`graph.py:1428-1480`) is the user-facing wrapper: it strips
the `cuda_graph_stream` / `cuda_graph_event` control kwargs, **reconstructs the
capture-time flattened argument order** from `user_args` (by position) and
`user_kwargs` (by name) — the local fix that lets a recorded kwarg be passed
positionally at replay — appends `module_params`, and calls `Graphed.apply`.

#### 4.4.5 Invariants the caller must uphold

- **No module hooks at capture.**  `make_graphed_callables` asserts the modules
  carry no hooks; FSDP therefore pops all real hooks and passes unshard/reshard
  as `capture_time_hooks` instead (§6.1).
- **Fixed parameter addresses across micro-batches** (see §4.4.2) — provided by
  `TracePoolAllocator` in the `"optimized"` phase.
- **Static shapes, dtypes, and control flow.**  The captured graph encodes one
  concrete shape/dtype path; varying sequence length, batch size, or branch
  taken per micro-batch is unsupported.
- **Parameters unchanged in identity since capture.**  Replay assumes the same
  `module.parameters()` objects/addresses; re-wrapping or reallocating params
  after capture invalidates the graph.
- **Outputs are assumed to require grad.**  Backward capture allocates a
  `static_grad_output` for every output that `requires_grad`; a module that
  returns a mix must keep that structure stable.
- **RNG correctness.**  Dropout and other RNG ops are made replay-correct via
  `register_generator_state` on the fwd/bwd graphs (`graph.py:730-735`); without
  it every replay would reuse the captured mask (§9).

#### 4.4.6 Why this shapes the FSDP integration

The copy-in/replay/copy-out contract means the graphed region is a **pure
function of its static buffers**.  Everything that must vary per micro-batch —
parameter all-gather (`unshard`), buffer release (`reshard`), and gradient
reduction (`reduce_grad`) — is deliberately kept *outside* the graph: unshard
lands the weights at the fixed capture addresses before replay, and reshard /
reduce-scatter consume the param grads autograd installs after replay.  The
graph owns the math; FSDP owns the memory movement around it.

## 5. Lifecycle

The three stages below are the detailed, hook-by-hook expansion of the
trace → capture → replay flow introduced in §3 (Stage 1/2/3 map 1:1).

```
CUDA Graph Lifecycle (trace → capture → replay)

Stage 1 · Microbatch 0 — TRACE
  • forward + backward run eagerly
  • allocator records every alloc / free  →  trace events
  • plan(): conflict-graph coloring → fixed per-key slots
    phase = "optimized" (addresses now deterministic)

        ↓

Stage 2 · Microbatch 1 — RECORD + CAPTURE
  • root_forward_pre_hook: create shared graph_pool + capture_stream,
    set capture_stream as current stream
  • forward (per module): unshard → record_module → eager forward → reshard
  • backward (eager): unshard → reshard + reduce_grad
  • post_backward_final_callback → capture_and_install():
      1. clone sample kwargs (fresh leaves)
      2. pop all FSDP hooks
      3. make_graphed_callables → warmup + fwd capture + bwd capture
         (replaces module.forward with graphed version)
      4. restore FSDP hooks
      5. mark _fsdp_cg_installed

        ↓

Stage 3 · Microbatch 2+ — REPLAY
  • forward (per module): unshard → fwd_graph.replay → reshard
    (_fsdp_cg_installed → skip record)
  • backward: unshard → bwd_graph.replay → reshard + reduce_grad
```

Real FSDP hooks (`unshard` / `reshard` / `reduce_grad`) run **around** every
stage but are never captured into the graph — see §6.

## 6. Hook strategy

### 6.1 During capture (`make_graphed_callables`)

Real FSDP hooks are **popped** from the entire module tree before calling
`make_graphed_callables` (it asserts modules have no hooks).  `capture_time_hooks`
handle unshard/reshard during warmup, forward capture, and backward capture.

### 6.2 During replay

Real FSDP hooks are **restored** after capture.  They fire normally around
`module.__call__`:

- `forward_pre_hook` → unshard
- `module.forward` (graphed) → fwd replay
- `forward_hook` → reshard
- `backward_pre_hook` → unshard (bwd_pass)
- `Graphed.backward` → bwd replay
- `backward_hook` → reshard + reduce_grad

## 7. Capture stream & shared pool

```python
# hooks.py — root forward pre-hook (once, when enable_cuda_graph and
# cuda_graph_stream is None)
ctx.cuda_graph_stream = torch.cuda.Stream()
torch.cuda.set_stream(ctx.cuda_graph_stream)   # becomes the current stream
ctx.cuda_graph_pool = torch.cuda.graph_pool_handle()

# CudaGraphRunner passes them to make_graphed_callables:
make_graphed_callables(
    modules,
    sample_args,
    pool=ctx.cuda_graph_pool,
    capture_stream=ctx.cuda_graph_stream,
)
```

## 8. Parameter gradients

The captured backward binds a parameter leaf directly to its M-FSDP
`main_grad` buffer only when shape, dtype, device, layout, stride, sharding
policy, and TE-fusion ownership are compatible.  This makes trace and replay
write to one stable optimizer-facing allocation and avoids an unnecessary
`param.grad → main_grad` copy.

1. **Capture-time pre-hook:** unshard installs the live compute parameters, then
   the runtime refreshes the parameter input surface.
2. **Backward capture/replay:** compatible leaves use `main_grad` as their
   static gradient buffer; incompatible leaves continue to use graph-owned
   storage.
3. **Autograd hand-off:** graph-owned parameter gradients are cloned before
   being returned.  A compatible `main_grad` result is returned as a detached
   alias, so `param.grad` may already point at `main_grad`.
4. **FSDP post-backward hook:** `reduce_grad` skips the copy when both pointers
   match, clears `param.grad`, and performs the normal reduction.

Gradient reduction and resharding remain outside the graph.  TE fused weight
gradients keep their existing ownership path and are not rebound by the runtime.

## 9. Known issues & fixes

| Issue | Root cause | Fix | Date |
|-------|-----------|-----|------|
| `cudaErrorStreamCaptureInvalidated` with `torch.compile` | Shared warmup/capture stream; compile recompilation breaks capture | Separate throwaway warmup stream (workaround; ideal: keep shared stream and fix compile guard mismatch) | 2026-06 |
| OOM during warmup | `warmup_outputs` held tensor refs across `gc.collect` + `empty_cache` | `param.grad = None` prevents gradient accumulation across warmup iterations | 2026-06 |
| Non-tensor kwargs crash `.requires_grad` | `tree_flatten` passes `None` into `static_input_surface` | 6 `is not None` / `isinstance` guards in te-graph-runtime | 2026-06 |
| Positional `hidden_states` missing in replay | `kwargs_keys` validation rejected positional args | `functionalized` checks both `user_args` and `user_kwargs` | 2026-06 |
| Replayed gradients attached to stale sharded parameters | M-FSDP swaps parameter objects in the backward pre-hook | Refresh live parameter surfaces after the hook and bind compatible `main_grad` buffers | 2026-07 |

## 10. Files

| File | Role |
|------|------|
| `cuda_graph_runner.py` | `CudaGraphRunner` — sample arg recording, hook save/restore, batch capture orchestration |
| `hooks.py` | Capture trigger, shared pool+stream creation |
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
| Slot state | Automatic (fixed per-key slots after `plan()`) |

## 12. Full-iteration optimizer-gradient storage

The default-off `enable_full_iteration_cuda_graph` mode preserves only the
optimizer-facing gradient objects that cross the full-iteration graph boundary.
Transient full weight and gradient buffers allocate and reshard inside capture;
the CUDA graph private pool provides stable replay addresses and lifetime reuse.

- `_pre_backward_setup()` pre-allocates local dist grads before capture.
- `_maybe_free_grad_data()` keeps optimizer-facing local gradient storage resident.
- FSDP and optimizer zero-grad preserve optimizer-facing `grad`/`decoupled_grad`
  identities and clear their local storage in place.
- Full-iteration mode uses `StorageFreeingBucketAllocator`; `TracePoolAllocator`
  remains available for explicit and per-module CUDA graph modes.

All behavior is gated by `enable_full_iteration_cuda_graph=False` by default.
The MCore FSDP adapter enables it when
`TransformerConfig.cuda_graph_impl == "full_iteration"`.

## 13. Forward-only validation and test

When Megatron FSDP v2 is active, `FullCudaGraphWrapper` runs forward-only
validation and test passes eagerly. The bypass happens before static input
staging and does not create or update validation CUDA graph state. Training
iterations continue to use full-iteration capture and replay, while non-v2
models retain their existing validation graph behavior.
