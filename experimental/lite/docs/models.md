# Models

Megatron Lite model packages register themselves through
`megatron.lite.model.registry.register_model`.

Each registered implementation points at a protocol module. A protocol module is
expected to expose:

- `ImplConfig`: a dataclass for implementation-specific options.
- `build_model_config(hf_path)`: returns a model config object.
- `build_model(model_cfg, impl_cfg)`: returns a `ModelBundle`.

## Toy Dense

`toy_dense` is the only model included in this slice. It is a two-layer PyTorch
MLP used to validate the package boundary:

```python
from megatron.lite.runtime import RuntimeConfig, create_runtime
from megatron.lite.runtime.backends.mlite import MegatronLiteConfig

runtime = create_runtime(
    RuntimeConfig(
        backend="mlite",
        backend_cfg=MegatronLiteConfig(model_name="toy_dense", impl="torch"),
    )
)
handle = runtime.build_model()
```

Toy Dense is not intended as a benchmark, reference transformer, or production
training example.
