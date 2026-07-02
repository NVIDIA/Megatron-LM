# Liger-Kernel Integration

**[Liger-Kernel](https://github.com/linkedin/Liger-Kernel)** is an open-source
collection of Triton kernels for LLM training. Megatron-Core natively supports
two Liger kernels — **RMSNorm** and **vocab-parallel cross-entropy** — opted
in via config flags. No monkey-patching required.

## Installation

Liger-Kernel is an optional runtime dependency:

```bash
pip install liger-kernel
```

## Enabling RMSNorm

The `LigerSpecProvider` swaps Liger's `LigerMegatronRMSNorm` into the layer-norm
slot. Enable it via `use_liger=True` when building the GPT layer spec:

```python
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

layer_spec = get_gpt_layer_local_spec(
    normalization="RMSNorm",
    use_liger=True,
)
```

`use_liger=True` is mutually exclusive with `use_kitchen=True`. When
`normalization` is anything other than `"RMSNorm"`, the layer-norm slot falls
through to the default `LocalSpecProvider` builder.

## Enabling Cross-Entropy

Liger's vocab-parallel cross-entropy plugs into the existing
`cross_entropy_fusion_impl` dispatch in
`LanguageModule.compute_language_model_loss`. Enable it on the
`TransformerConfig`:

```python
config = TransformerConfig(
    ...
    cross_entropy_loss_fusion=True,
    cross_entropy_fusion_impl="liger",
)
```

The kernel handles both TP=1 and TP>1; the tensor-parallel group is passed in
explicitly by Megatron at call time.

## Combined Example

```python
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.transformer.transformer_config import TransformerConfig

config = TransformerConfig(
    num_layers=24,
    hidden_size=2048,
    num_attention_heads=16,
    normalization="RMSNorm",
    cross_entropy_loss_fusion=True,
    cross_entropy_fusion_impl="liger",
)

model = GPTModel(
    config=config,
    transformer_layer_spec=get_gpt_layer_local_spec(
        normalization="RMSNorm",
        use_liger=True,
    ),
    vocab_size=32_000,
    max_sequence_length=4096,
)
```

## Compatibility

| Feature | Supported |
|---|---|
| RMSNorm (`normalization="RMSNorm"`) | yes |
| Other norm types (`LayerNorm`, etc.) | falls through to default |
| Cross-entropy at TP=1 | yes |
| Cross-entropy at TP>1 | yes (kernel performs the in-vocab AllReduce) |
| `deterministic_mode=True` | no — blocks all `cross_entropy_loss_fusion` modes |
| TransformerEngine norm/CE backends | use `LocalSpecProvider`/`TESpecProvider` instead — `LigerSpecProvider` extends `LocalSpecProvider` and is not stackable with TE |

If `liger-kernel` is not installed, instantiating `LigerSpecProvider` or
dispatching to `cross_entropy_fusion_impl="liger"` raises `ImportError` with
an actionable message.
