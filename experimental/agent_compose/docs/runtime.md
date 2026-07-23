# Runtime Interface

The public runtime entrypoint is `megatron.lite.runtime`. It defines a
backend-neutral interface so training applications do not depend on a concrete
Megatron Lite, Megatron Bridge, or integration-specific backend.

## API Tiers

Every backend implements the pretraining tier:

- `build_model`
- `save_checkpoint` and `load_checkpoint`
- `train_mode` and `eval_mode`
- `forward_backward`
- `zero_grad`
- `optimizer_step`
- `lr_scheduler_step`

Implementing `export_weights` adds the `rl_ready` tier. Implementing both
`export_weights` and `to` adds the `rl_best` tier, including model and optimizer
offload between training and rollout phases. `Runtime.tier` reports the highest
implemented tier.

## Shared Contracts

The runtime package exposes:

- `RuntimeConfig`, `ParallelConfig`, and `OptimizerConfig`;
- `Batch`, `PackedBatch`, and legacy `TrainBatch` inputs;
- `ModelOutputs` and `ForwardResult` outputs;
- the opaque `ModelHandle` returned by `build_model`;
- `LossContext` for per-microbatch output and loss policy.

Imports are lazy where Torch-backed data contracts are involved. Importing
`megatron.lite.runtime` alone does not load Torch.

## Backend Registration

A backend module provides `create(hf_path, backend_cfg)` and registers its
dotted module path:

```python
from megatron.lite.runtime import RuntimeConfig, create_runtime, register_runtime

register_runtime("my_backend", "my_package.runtime")
runtime = create_runtime(RuntimeConfig(backend="my_backend"))
```

No built-in backend is registered in the skeleton. Each backend will be added
with its own reference, implementation skill, and lifecycle validation.
