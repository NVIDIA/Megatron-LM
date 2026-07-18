# Megatron FSDP v2 full-iteration CUDA graph design

> Experimental: this document describes how Megatron FSDP v2 participates in
> MCore full-iteration CUDA graph capture. Per-module built-in capture is
> documented separately in
> [`mfsdp_v2_builtin_cuda_graph_design.md`](mfsdp_v2_builtin_cuda_graph_design.md).

## Scope

Full-iteration CUDA graph capture records the training forward/backward path as
one CUDA graph. The optimizer step runs outside this graph.

Megatron FSDP v2 does not own the full-iteration graph wrapper. Instead, it
keeps FSDP gradient objects and temporary buffers compatible with the wrapper in
`megatron/training/training.py`.

## How to enable it

Use Megatron FSDP v2 with MCore full-iteration CUDA graphs:

```bash
--use-megatron-fsdp-v2 \
--cuda-graph-impl=full_iteration \
--no-check-for-nan-in-loss-and-grad
```

`--cuda-graph-modules` must be empty when `--cuda-graph-impl=full_iteration`.
The full-iteration path captures the whole training iteration rather than a
selected list of module scopes.

The MCore FSDP adapter passes
`enable_full_iteration_cuda_graph=True` to `fully_shard()` when
`TransformerConfig.cuda_graph_impl == "full_iteration"`.

## FSDP responsibilities

Full-iteration graph capture crosses the boundary between backward and the
optimizer step. Therefore the optimizer-facing gradient objects must keep stable
identities even though FSDP normally lazily releases gradient storage.

Megatron FSDP v2 handles this by:

- allocating distributed gradient views before full-iteration capture needs
  them;
- preserving optimizer-facing `grad` or `decoupled_grad` DTensor objects;
- zeroing kept gradient storage in place during `zero_grad()`;
- disabling lazy release of optimizer-facing `main_grad_buffer.data` while
  `enable_full_iteration_cuda_graph=True`;
- keeping transient full weight/gradient buffers outside the optimizer-facing
  object contract.

## Buffer ownership

Full-iteration mode uses `StorageFreeingBucketAllocator`, not the
`TracePoolAllocator` used by the per-module built-in path.

This is intentional. The full-iteration CUDA graph private pool owns replay
addresses for transient buffers captured inside the full iteration. FSDP only
needs to preserve the Python object identities that survive across the graph
boundary and are consumed by the optimizer.

## Validation and test passes

When Megatron FSDP v2 is active, forward-only validation and test passes bypass
full-iteration CUDA graph capture. Training iterations continue to use
full-iteration capture and replay; validation/test forward passes run eagerly.

This avoids creating validation graph state from a different forward-only
execution pattern.

## Limitations

- This path is for training; inference CUDA graph scopes must be disabled.
- `--cuda-graph-modules` is not used with full-iteration capture.
- The optimizer step is outside the captured graph.
- FSDP NaN checking should be disabled with
  `--no-check-for-nan-in-loss-and-grad` for this mode.

## Relevant files

| File | Role |
| --- | --- |
| `mcore_fsdp_adapter.py` | Passes `enable_full_iteration_cuda_graph` into `fully_shard()`. |
| `fully_shard.py` | Stores full-iteration graph mode in FSDP state. |
| `param_group.py` | Preserves optimizer-facing gradient storage and zeroes it in place. |
| `fsdp_module.py` | Releases transient buffers while keeping graph-visible gradient objects stable. |
| `training.py` | Wraps the forward/backward function with `FullCudaGraphWrapper`. |
