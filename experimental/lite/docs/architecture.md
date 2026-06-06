# Architecture

Megatron Lite source lives under `experimental/lite/megatron/lite` and is split
into three layers:

- `runtime`: lifecycle and training-step orchestration.
- `model`: model registration plus model-specific build/load/export protocols.
- `primitive`: reusable lower-level pieces such as parallel state, tensor-parallel
  layers, checkpoint conversion, MoE utilities, and optimizer wrapping.

The runtime does not know Qwen implementation details. It imports a model
protocol from the model registry, builds the typed implementation config, then
delegates model construction to that protocol.

## Import Boundary

The source root for local use is `experimental/lite`; adding that directory to
`PYTHONPATH` exposes the package as `megatron.lite`. Internal imports also use
`megatron.lite` so user-facing code matches the final package path.

## Runtime Boundary

The runtime API owns:

- Distributed initialization.
- Model protocol loading.
- Model checkpoint save/load dispatch.
- Forward/backward microbatch orchestration.
- Optimizer and learning-rate scheduler stepping.
- Optional model/optimizer offload hooks.

The model protocol owns:

- Architecture config creation.
- Model chunk construction.
- Model-specific recompute/offload wiring.
- Model-specific optimizer construction.
- HF checkpoint load/export mapping.

## Current Deliberate Omissions

This package currently includes only the lite model implementation path. It
intentionally excludes non-lite model/runtime implementations and benchmark
entrypoints. FSDP2 is included as an optimizer primitive and can be selected by
model protocols that support it.
