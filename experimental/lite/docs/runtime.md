# Runtime

The public runtime entrypoint is `megatron.lite.runtime`.

```python
from megatron.lite.runtime import LiteConfig, ParallelConfig, RuntimeConfig, create_runtime

cfg = RuntimeConfig(
    backend="lite",
    hf_path="/path/to/hf-model",
    backend_cfg=LiteConfig(
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

## Config Types

`RuntimeConfig` selects the backend and carries the Hugging Face model path.

`LiteConfig` carries lite-backend settings:

- `model_name`: `qwen3`, `qwen3_moe`, or `qwen3_5`.
- `impl`: currently only `lite`.
- `parallel`: tensor, expert, pipeline, virtual pipeline, and context sizes.
- `optimizer`: Megatron-Core optimizer settings.
- `impl_cfg`: model-specific options consumed by each model protocol.

## Backend Registry

The only built-in backend key is `lite`.

Custom runtime backends can be registered with:

```python
from megatron.lite.runtime import register_runtime

register_runtime("my_backend", "my_package.my_runtime")
```

The target module must expose `create(hf_path, cfg)`.
