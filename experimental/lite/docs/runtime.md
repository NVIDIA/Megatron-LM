# Runtime

The public runtime entrypoint is `megatron.lite.runtime`.

```python
from megatron.lite.runtime import MegatronLiteConfig, ParallelConfig, RuntimeConfig, create_runtime

cfg = RuntimeConfig(
    backend="mlite",
    hf_path="/path/to/hf-model",
    backend_cfg=MegatronLiteConfig(
        model_name="qwen3",
        impl="lite",
        parallel=ParallelConfig(tp=1, pp=1, cp=1, ep=1),
    ),
)
runtime = create_runtime(cfg)
handle = runtime.build_model()
```

## API Tiers

All runtime backends implement the pretraining tier:

- `build_model`
- `save_checkpoint`
- `load_checkpoint`
- `train_mode`
- `eval_mode`
- `forward_backward`
- `zero_grad`
- `optimizer_step`
- `lr_scheduler_step`

The lite runtime also implements `export_weights` and `to` when the underlying
model and optimizer support those operations.

The `bridge` runtime implements the same runtime contract through
Megatron-Bridge and Megatron-Core optimizer/checkpoint helpers. It imports
`megatron.bridge` lazily from `build_model()`, so config construction and dry-run
examples can execute without Megatron-Bridge installed.

## Config Types

`RuntimeConfig` selects the backend and carries the Hugging Face model path.

`MegatronLiteConfig` carries `mlite` backend settings:

- `model_name`: `qwen3`, `qwen3_moe`, or `qwen3_5`.
- `impl`: currently only `lite`.
- `parallel`: tensor, expert, pipeline, virtual pipeline, and context sizes.
- `optimizer`: Megatron-Core optimizer settings.
- `impl_cfg`: model-specific options consumed by each model protocol.

`BridgeConfig` carries `bridge` backend settings:

- `model_name`: optional model identifier used for benchmark metadata.
- `parallel`: tensor, expert, pipeline, virtual pipeline, and context sizes.
- `optimizer`: Megatron-Core optimizer settings.
- `override_ddp_config`, `override_transformer_config`, and
  `override_optimizer_config`: explicit Megatron-Bridge/Core override maps.
- `param_offload` and `optimizer_offload`: offload model/optimizer state between
  train/eval contexts.

## Backend Registry

The built-in backend keys are `mlite` and `bridge`. Model implementations for
the native runtime remain selected through `MegatronLiteConfig.impl`, which
currently supports `impl="lite"`.

Custom runtime backends can be registered with:

```python
from megatron.lite.runtime import register_runtime

register_runtime("my_backend", "my_package.my_runtime")
```

The target module must expose `create(hf_path, cfg)`.
