# Megatron Lite

Megatron Lite is an experimental training runtime and model implementation layer
for Megatron. The source lives under `experimental/lite/megatron/lite`, and the
public import path is `megatron.lite`.

Do not import `experimental.lite` from user code. Examples and public APIs should
refer to `megatron.lite`.

## Scope

This initial drop contains:

- A lightweight runtime API in `megatron.lite.runtime`.
- Common training primitives in `megatron.lite.primitive`.
- Lite-only model implementations for Qwen3 MoE and Qwen3.5 MoE.
- Hugging Face safetensors load/export helpers for the included models.
- Megatron-Core optimizer wrapping for the lite runtime.

This initial drop intentionally does not include:

- Hybrid model implementations.
- Bridge model/runtime implementations.
- FSDP2 optimizer primitives.
- Benchmark entrypoints or experiment scripts.

## Layout

```text
experimental/lite/
  README.md
  docs/                       Design and usage notes
  megatron/
    lite/
      runtime/                Runtime API, config, backend registry, lite backend
      model/                  Model registry and Qwen model implementations
      primitive/              Parallel, checkpoint, optimizer, module, and op primitives
```

For local source-tree use:

```bash
export PYTHONPATH=/path/to/Megatron-LM/experimental/lite:$PYTHONPATH
```

## Public API

```python
from megatron.lite.runtime import LiteConfig, RuntimeConfig, create_runtime

cfg = RuntimeConfig(
    backend="lite",
    hf_path="/path/to/hf-model",
    backend_cfg=LiteConfig(model_name="qwen3", impl="lite"),
)
runtime = create_runtime(cfg)
handle = runtime.build_model()
```

Model names currently registered by default:

- `qwen3`: Qwen3 MoE lite implementation. HF `model_type` values
  `qwen3_moe` and `qwen2_moe` resolve to this model name.
- `qwen3_moe`: compatibility alias for the same Qwen3 MoE lite implementation.
- `qwen3_5`: Qwen3.5 MoE lite implementation.

## Docs

- [Architecture](docs/architecture.md)
- [Runtime](docs/runtime.md)
- [Models](docs/models.md)
- [Porting Notes](docs/porting.md)
