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
- FSDP2 optimizer primitives for supported lite model protocols.
- Reference runtime backends for comparison runs: `mbridge` for the legacy
  package and `bridge` for real Megatron-Bridge environments.
- A benchmark example that can dry-run or execute `mlite`, `mbridge`, and
  `bridge` backends.

This initial drop intentionally does not include:

- Hybrid model implementations.

## Layout

```text
experimental/lite/
  README.md
  docs/                       Design and usage notes
  examples/                   Optional integration and benchmark examples
  skills/                     Agent-agnostic maintenance skills
  megatron/
    lite/
      runtime/                Runtime API, config, and backend registry
      model/                  Model registry and Qwen model implementations
      primitive/              Parallel, checkpoint, optimizer, module, and op primitives
```

For local source-tree use:

```bash
export PYTHONPATH=/path/to/Megatron-LM/experimental/lite:$PYTHONPATH
```

## Public API

```python
from megatron.lite.runtime import MegatronLiteConfig, RuntimeConfig, create_runtime

cfg = RuntimeConfig(
    backend="mlite",
    hf_path="/path/to/hf-model",
    backend_cfg=MegatronLiteConfig(model_name="qwen3", impl="lite"),
)
runtime = create_runtime(cfg)
handle = runtime.build_model()
```

`backend="mlite"` selects the Megatron Lite runtime backend. `impl="lite"`
selects the model implementation inside the registered model family.
`backend="mbridge"` selects the legacy `mbridge` reference backend used by the
validated benchmark example. `backend="bridge"` selects the Megatron-Bridge
runtime backend and requires an environment where `import megatron.bridge` works
when the model is built.

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
- [Skills](skills/README.md)
- [Bench Example](examples/bench/README.md)

## Acknowledgements

The Qwen3 MoE LoRA adapter support follows Mind-Lab's PEFT/Mint-compatible
adapter work. Thanks to Mind-Lab for the reference implementation and guidance.
