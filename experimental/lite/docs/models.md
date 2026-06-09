# Models

Model code lives under `megatron.lite.model`.

The model registry maps `(model_name, impl)` pairs to model protocol modules.
The runtime loads the protocol and expects the following required symbols:

- `ImplConfig`: dataclass for model-specific knobs.
- `build_model_config(source, **overrides)`: returns a typed model config.
- `build_model(model_cfg, *, impl_cfg)`: returns a `ModelBundle`.

Optional protocol symbols:

- `load_hf_weights(chunk, hf_path, model_cfg, ps)`
- `export_hf_weights(chunks, model_cfg, ps, **kwargs)`
- `vocab_size(model_cfg)`

## Included Models

`qwen3_moe` is the canonical name for the Qwen3 MoE lite implementation:

```text
megatron.lite.model.qwen3_moe.lite.protocol
```

`qwen3` is kept as a legacy compatibility alias for the same implementation. It
does not mean dense Qwen3 support. Hugging Face `model_type` values `qwen3_moe`
and `qwen2_moe` currently resolve through this compatibility path.

`qwen3_5` maps to the Qwen3.5 MoE lite implementation:

```text
megatron.lite.model.qwen3_5.lite.protocol
```

## Acknowledgements

The Qwen3 MoE LoRA adapter support follows Mind-Lab's PEFT/Mint-compatible
adapter work. Thanks to Mind-Lab for the reference implementation and guidance.

## Adding A Model

Add a model package under `model/`, then register it in
`model/registry.py`:

```python
register_model(
    "my_model",
    package="megatron.lite.model.my_model",
    hf_model_types=["my_model"],
    impls={
        "lite": "megatron.lite.model.my_model.lite.protocol",
    },
)
```

New models should keep heavyweight imports inside protocol functions when
possible so importing `megatron.lite` stays cheap.
