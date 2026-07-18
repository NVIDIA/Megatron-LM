# Megatron FSDP v2 built-in CUDA graph design

> Experimental: this document describes the per-FSDP-module CUDA graph path
> built into Megatron FSDP v2. Full-iteration CUDA graph capture is documented
> separately in [`full_iteration_cuda_graph_design.md`](full_iteration_cuda_graph_design.md).

## Scope

Megatron FSDP v2 can capture selected leaf FSDP modules with CUDA graphs while
keeping FSDP unshard, reshard, and gradient-reduction work outside the graphed
region.

This path is enabled per FSDP module with `enable_cuda_graph=True`. It is useful
when only selected modules are graph-safe or when full-iteration capture is not
desired.

## How to enable it

Direct `fully_shard()` usage:

```python
for layer in model.layers:
    fully_shard(layer, enable_cuda_graph=True)

fully_shard(model)  # root wrapper; do not enable CUDA graph on this parent
```

Megatron training CLI usage for Megatron FSDP v2:

```bash
--use-megatron-fsdp-v2 \
--mfsdp-cuda-graph attn mlp
```

Supported `--mfsdp-cuda-graph` module selectors are:

- `transformer` for `TransformerLayer`
- `mamba` for `MambaLayer`
- `attn` for attention modules
- `mlp` for dense MLP modules
- `moe` for MoE expert MLP modules
- `moe_router` for MoE router modules

Only non-nested leaf FSDP modules are eligible. A parent FSDP module that
contains other FSDP modules cannot also use `enable_cuda_graph=True`; the
runtime raises a `RuntimeError` for that configuration.

## Why FSDP needs special handling

CUDA graph replay uses the same memory addresses that were observed during
capture. FSDP normally materializes full parameters only temporarily:

```text
forward pre-hook  -> unshard parameters into a temporary full buffer
forward compute   -> read full parameters
forward hook      -> reshard and release the temporary full buffer
```

A normal temporary allocator may return a different address on the next
microbatch. That is not compatible with CUDA graph replay.

Megatron FSDP v2 solves this with two components:

1. [`TracePoolAllocator`](../allocator.py) traces one microbatch, builds a
   static key-to-slot plan, and gives each planned FSDP buffer a stable address.
2. `te_graph_runtime.make_graphed_callables()` supports `capture_time_hooks`,
   so FSDP can run unshard/reshard during warmup and capture without recording
   those hooks into the CUDA graph.

The graph captures module math. FSDP memory movement and gradient reduction
remain in normal Python hooks around the graph.

## Lifecycle

```text
Microbatch 0: trace
  eager forward/backward
  TracePoolAllocator records alloc/free events
  post-backward final callback calls plan()
  allocator phase becomes "optimized"

Microbatch 1: record and capture
  forward hooks unshard selected modules into stable trace-pool slots
  CudaGraphRunner records sample inputs and outputs
  backward runs eagerly
  post-backward final callback calls capture_and_install()
  selected module forward methods are replaced with graphed callables

Microbatch 2+: replay
  FSDP hooks still run around each module call
  unshard places parameters back into the same trace-pool slots
  module forward/backward replay CUDA graphs
  post-backward hooks reshard and reduce gradients outside the graph
```

## Runtime pieces

| Component | Role |
| --- | --- |
| `TracePoolAllocator` | Provides stable tensor addresses for FSDP temporary buffers after the trace microbatch. |
| `CudaGraphRunner` | Records sample module inputs/outputs, invokes `make_graphed_callables()`, and installs graphed forwards. |
| `capture_time_hooks` | Run FSDP unshard/reshard during warmup and capture without recording them in the graph. |
| FSDP forward/backward hooks | Continue to run during replay around the graphed module call. |
| `te_graph_runtime` | Vendored TE-compatible graph runtime with local support needed by Megatron FSDP v2. |

## Hook behavior

During capture, real module hooks are temporarily removed because
`make_graphed_callables()` requires hook-free modules. Equivalent FSDP
unshard/reshard work is passed as `capture_time_hooks`.

During replay, real hooks are restored and fire normally:

```text
forward_pre_hook  -> unshard
graphed forward   -> replay forward graph
forward_hook      -> reshard
backward_pre_hook -> unshard for backward
graphed backward  -> replay backward graph
backward_hook     -> reshard and reduce gradients
```

## Parameter gradients

The captured backward may bind compatible parameter-gradient outputs directly
to Megatron FSDP v2 `main_grad` storage. Compatibility requires matching shape,
dtype, device, layout, stride, sharding policy, and gradient-ownership rules.

When direct binding is possible, replay writes into the optimizer-facing
gradient buffer and avoids an extra `param.grad -> main_grad` copy. When it is
not possible, the graph uses graph-owned gradient storage and FSDP copies or
accumulates into `main_grad` in the normal post-backward path.

Gradient reduction remains outside the graph in both cases.

## Requirements and limitations

- Selected modules must have static shapes, dtypes, and control flow across
  replayed microbatches.
- Selected modules must be leaf FSDP modules; nested graph-enabled FSDP modules
  are not supported.
- `TracePoolAllocator` must be used. `fully_shard(..., enable_cuda_graph=True)`
  selects it automatically.
- Releasing the trace pool invalidates captured addresses. Use
  `FSDPModule.release_memory_pool()` so graphs are dropped and recaptured before
  slot tensors are reallocated.
- Full-iteration capture uses a different runtime path; see
  [`full_iteration_cuda_graph_design.md`](full_iteration_cuda_graph_design.md).

## Relevant files

| File | Role |
| --- | --- |
| `fully_shard.py` | Selects `TracePoolAllocator` and records `enable_cuda_graph` in FSDP state. |
| `hooks.py` | Records sample inputs/outputs and triggers batch capture after backward. |
| `cuda_graph_runner.py` | Orchestrates hook save/restore and `make_graphed_callables()` invocation. |
| `te_graph_runtime/` | Vendored graph runtime used for capture and replay. |
| `trace_pool_allocator_design.md` | Details the stable-address allocator used by this path. |
