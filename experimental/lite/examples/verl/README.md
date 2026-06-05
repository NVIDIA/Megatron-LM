# VERL Megatron Lite Example

This example shows the import and config mapping needed for a VERL external
engine backed by Megatron Lite.

The full engine shape follows `verl-recipe/verlbb`: keep the VERL
`BaseEngine` lifecycle and replace the runtime imports/config with
`megatron.lite`.

```python
from megatron.lite.runtime import RuntimeConfig, create_runtime

runtime = create_runtime(
    RuntimeConfig(
        backend="mlite",
        hf_path=model_config.local_path,
        backend_cfg=megatron_lite_config,
    )
)
handle = runtime.build_model()
```

Use `backend="mlite"` for the Megatron Lite runtime backend. Use
`impl="lite"` in `MegatronLiteConfig` for the model implementation.
