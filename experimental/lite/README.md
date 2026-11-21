# Megatron Lite

Megatron Lite is an experimental package for building Megatron-native model
implementations behind a small, explicit contract. It is intentionally staged:
this first slice adds the public shape of the package, the runtime/model
registries, primitive interface contracts, validation scaffolding, and a tiny
pure PyTorch model that proves the contracts are importable and executable on a
single CPU process.

## Motivation

Megatron model work often mixes several concerns at once: model composition,
parallel runtime setup, primitive selection, checkpointing, and downstream
training-framework integration. Megatron Lite separates those concerns so that a
model implementation can declare what it needs, a runtime can consume the
declaration through a stable API, and primitive work can be reviewed in small
increments.

This package is meant to make early model bring-up easier to review. Each later
PR can add one capability at a time while keeping the import path and contracts
stable.

## Current Contents

- `megatron.lite.runtime`: runtime creation, backend registration, and runtime
  interface definitions.
- `megatron.lite.model.registry`: model and implementation registry.
- `megatron.lite.primitive`: protocol, config, and bundle dataclasses used by
  model implementations.
- `megatron.lite.model.toy_dense`: a two-layer dense model used only to validate
  the contract.
- `tests/run_primitive_validation.sh`: a local CPU validation entrypoint.
- `skills/`: short operating notes for keeping future Lite changes scoped.

## Non-Goals For This Slice

This PR does not add production model support, Qwen support, distributed
parallelism, FSDP2, LoRA, checkpointing, VERL integration, custom kernels, or
real primitive implementations. Those pieces should land in follow-up slices
after this package boundary is reviewed.

## Local Validation

Run from the repository root:

```bash
experimental/lite/tests/run_primitive_validation.sh
```

The script exports `experimental/lite` on `PYTHONPATH` and runs the local CPU
tests. No GPU is required.
