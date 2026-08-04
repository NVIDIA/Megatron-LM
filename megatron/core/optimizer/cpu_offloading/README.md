## How to use ?

Add these flags to enable optimizer cpu offload in MCore.

```bash
--optimizer-cpu-offload
--optimizer-offload-fraction 1.0
--use-precision-aware-optimizer
```

## Configuration Recommendations

Gradient copy from GPU to CPU, CPU optimizer step, and subsequent parameter copy from CPU to GPU can be time-consuming operations, and it is recommended to use the flag `--overlap-cpu-optimizer-d2h-h2d` to execute them concurrently.

## Chunked GPU optimizer-state offload

`--chunked-optimizer-state-offload` keeps a configurable fraction of optimizer tensor state and
master weights in pinned CPU memory between updates, while the optimizer update itself still runs
on GPU. Tensor state is restored and updated in bounded parameter chunks. All selected master
weights use one full restore window and remain on CUDA across every state chunk during an update or
distributed-checkpoint state initialization.

```bash
--chunked-optimizer-state-offload
--optimizer-state-offload-chunk-size-mb 256
--optimizer-state-offload-fraction 1.0
--ckpt-format torch_dist
```

A chunk size of `0` restores all selected tensor state for one update and therefore does not bound
the temporary tensor-state peak; argument validation emits a warning for this setting. A positive
size bounds that tensor-state window, but never the full selected master-weight window described
above. The fraction is an approximate byte fraction because a parameter and all of its optimizer
state are an atomic bundle. Partial selection prioritizes bundles that actually include a separate
master before state-only bundles; precision-aware FusedAdam can own a separate master for native
FP32 DistributedOptimizer shards too.
Supported paths are Adam with `DistributedOptimizer`, and BF16 Muon with compact-layout
`LayerWiseDistributedOptimizer`; BF16 and FP8 parameter gather are supported. Full-iteration and
optimizer CUDA graphs are not supported while the fraction is nonzero. When optimizer state is
saved, async distributed-checkpoint save is rejected: its background writer can retain references
to the same pinned CPU buffers that the next optimizer update reuses. Synchronous distributed
optimizer-state save is supported.

Chunked execution also requires optimizer parameter-group metadata updates to be independent of
the parameter subset passed to each chunk. CPU tensor-valued group fields are compared by value;
CUDA tensor-valued fields are only checked for matching shape, dtype, and device to avoid a host
synchronization per chunk. Optimizers whose CUDA group metadata depends on the subset are unsupported.

Steady-state updates use two reusable GPU state windows per active offload manager: H2D for the
next chunk, optimizer compute for the current chunk, and D2H for the previous chunk can overlap.
The normal tensor-state staging bound is therefore about
`2 * optimizer_state_offload_chunk_size_mb` per manager that is actively stepping, not one
chunk; a single atomic parameter bundle can exceed the target. LayerWise also prefetches the first
chunk of child `i+1` while child `i` owns its two-slot pipeline, so its cross-child steady-state
peak is about `3 * optimizer_state_offload_chunk_size_mb` across those two managers. The 256 MiB
example can consequently hold about 768 MiB during that LayerWise overlap window. Temporary state
and master windows are allocated on the current compute stream so their reserved memory returns to
the same allocator pool used by forward/backward; the H2D stream waits for both that compute stream
and the D2H stream before writing them. D2H completion does not block the host at the forward
boundary.

Every `ChainedOptimizer` recursively shares one copy-stream pair across its LayerWise and sibling
`DistributedOptimizer` managers, including ordinary Adam dense/expert chains. Stream sharing
serializes transfer order but does not merge managers' two-slot pools. LayerWise starts only its
first managed child's first state chunk during gradient finalization, then pipelines later children
and chunks with optimizer compute instead of prefetching every child's state at once. A host
synchronization is used only for synchronous distributed-checkpoint access and one-time lazy state
initialization.

Selected master bindings and supported full-size state bindings become CPU tensors as soon as their
asynchronous D2H copies are enqueued. CUDA readers must restore them through the optimizer offload
lifecycle, and host readers must first call the checkpoint synchronization hook. The training loop
enters this state only at the optimizer-to-forward boundary, where no other master reader is
allowed.
MCore's model-to-main, main-to-model, and MXFP8 param-buffer copy entry points restore a
CPU-bound selected master asynchronously, order the current compute stream after that H2D, and then
validate residency. This makes external reload and param-staging entry points self-healing.

Setting `optimizer_state_offload_fraction` to zero is a full disable even when the feature flag is
present: no manager, transfer stream, training hook, or checkpoint-format restriction is installed.
Checkpoint format and async-save restrictions are also skipped when optimizer state is explicitly
excluded from both load and save; excluding only saves permits async model-only checkpoints in
formats that otherwise support async save.

Distributed checkpointing is required whenever optimizer state is loaded or saved because its
sharded-state construction lets the offloader initialize selected state in bounded chunks, and its
optimizer load handoff can preserve CPU canonical tensors instead of device-casting the full state
to CUDA. A loader is not guaranteed to fill every pinned destination in place: fully-parallel
loading may create transport temporaries or replace a destination with a non-pinned CPU tensor.
The offloader adopts and pins selected CPU state afterward, while moving an unselected parameter's
full-size tensor state back to that parameter's CUDA device.
The chunk size therefore bounds the tensor-state portion of optimizer initialization, steady-state
updates, and the offloader's own load conversions. It does not bound the full selected master
window or all temporary memory internal to a distributed checkpoint strategy. The legacy torch
checkpoint path can reconstruct the full state on CUDA and is not supported.

The deprecated `--offload-optimizer-states` argument remains as a parser compatibility spelling.
Configuration validation emits a visible `FutureWarning`, enables the new mode, and consumes the
legacy field so copying the normalized configuration does not warn again. Its default-equivalent
settings are chunk size `0` and fraction `1.0`, with the restrictions above.
