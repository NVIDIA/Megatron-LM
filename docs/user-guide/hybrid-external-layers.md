# Defining HybridModel layers outside Core

`HybridModel` accepts layers implemented in a downstream package through
`hybrid_layer_config_list`. A layer's configuration constructs its implementation
directly. No layer symbol, registration, or module specification is needed for the
new layer.

The downstream package imports Core's `HybridLayerConfig`, `HybridLayer`, and
`HybridLayerBuildContext`. Core calls these interfaces without importing the
downstream package. The same mechanism can be used by model code in
`megatron.bridge` when that package is introduced.

## Define a config and a layer

This small channel-scaling example demonstrates the interface. Put both classes
in an importable module in your package:

```python
from dataclasses import dataclass

import torch

from megatron.core.models.hybrid import (
    HybridLayer,
    HybridLayerBuildContext,
    HybridLayerConfig,
)


@dataclass
class GainConfig(HybridLayerConfig):
    initial_gain: float = 1.0

    def build_layer(self, context: HybridLayerBuildContext) -> HybridLayer:
        if self.initial_gain <= 0:
            raise ValueError("initial_gain must be positive")
        return GainLayer(self, context)

    def estimate_flops(
        self, *, total_real_tokens_in_batch, seqlen_squared_sum_in_batch
    ):
        return total_real_tokens_in_batch * self.hidden_size


class GainLayer(HybridLayer):
    def __init__(self, config, context) -> None:
        super().__init__(config, context)
        device = "cpu" if config.use_cpu_initialization else torch.cuda.current_device()
        self.gain = torch.nn.Parameter(
            torch.full(
                (config.hidden_size,),
                config.initial_gain,
                dtype=config.params_dtype,
                device=device,
            )
        )

    def forward(
        self,
        hidden_states,
        *,
        attention_mask=None,
        rotary_pos_emb=None,
        packed_seq_params=None,
        padding_mask=None,
        sequence_len_offset=None,
        input_ids=None,
    ):
        return hidden_states * self.gain
```

`build_layer` receives a cloned config for one physical layer. Return a fresh
`HybridLayer` using that config and the supplied context. The base class records
the layer number and explicit process groups. Keep live modules, process groups,
and other runtime resources out of config fields.

The layer owns its entire computation, including any normalization and residual
connection. Its `forward` method returns one tensor with the same shape and
layout as `hidden_states`. Implement the arguments relevant to your operation;
for example, attention must respect attention masks and packed-sequence
boundaries. The stack passes the same arguments during eager execution and full
recomputation. Ordinary module calls preserve PyTorch hooks.

`input_ids` is optional and may be `None`, particularly on downstream pipeline
stages or inside MTP. A layer that needs token IDs must validate their availability.
This interface does not add token transport between pipeline stages or define
token-aware MTP behavior for a new layer.

## Compose the model

Mix the new config with existing layer configs:

```python
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.transformer.attention_layer_config import AttentionLayerConfig
from megatron.core.transformer.transformer_config import TransformerConfig

common = TransformerConfig(num_layers=3, hidden_size=128, num_attention_heads=4)
gain = GainConfig.from_config(common)
gain.initial_gain = 0.5
layers = [AttentionLayerConfig.from_config(common), gain, gain]

# pg_collection is supplied by the caller's distributed initialization.
model = HybridModel(
    config=common,
    vocab_size=32000,
    max_sequence_length=2048,
    hybrid_layer_config_list=layers,
    pg_collection=pg_collection,
    position_embedding_type="rope",
)
```

The two occurrences of `gain` create independent configs and parameters. The
source configs remain unchanged. `from_config` copies normalized common fields
and initializes defaults for additional dataclass fields. Construct a config
directly if its new fields are required and have no defaults. Core's model and
stack check execution compatibility before building layers. Validate
layer-specific fields in `build_layer` or the layer constructor; copying a
normalized config with `from_config` does not rerun `__post_init__`.

The standard training entry point uses the same list in
`megatron.training.models.hybrid.HybridModelConfig`. The existing
`HybridModelBuilder` forwards it to Core. Architecture config objects do not need
to be placed on argparse.

## Supported execution contract

The initial extension supports ordinary hidden-state streams with TP=1 and CP=1,
including eager training and full-layer recomputation (`uniform` and `block`).
Pipeline selection and MTP use the existing `PipelineSplit` and `MTPSplit`
mechanisms. Each selected decoder or MTP occurrence receives its own config and
module; the build context identifies its layer and MTP placement. As with
built-in config lists, repeated MTP sections must refer to the same source config
objects in the same order.

For custom layers, Core's `HybridModel` and `HybridStack` reject sequence
parallelism, TP communication overlap, quantized execution, CUDA graphs,
mHC/wide residual streams, shortcut-MoE,
overlapped expert communication, CPU/activation offloading, and selective recomputation.
Inference contexts are also rejected. Those features require their own layer
integration and validation; merely inheriting this interface does not enable
them. Distributed wrappers still require their normal supported configurations.

A custom attention layer can set the class attribute
`uses_shared_rotary_embeddings = True` to use the model's shared RoPE. Its
`kv_channels` must match the model's shared rotary width. Otherwise, the layer
owns any positional-embedding computation it needs.

## Metrics and checkpoints

The standard training loop requires `estimate_flops` to return this layer's
**forward** FLOPs for the supplied global-batch token statistics. The model
estimator applies its forward/backward factor and includes every decoder and MTP
occurrence. Missing estimates raise an explicit error instead of applying an
unrelated built-in formula. Keep the estimator pure: model setup may call it to
validate availability before training starts, independently of the runtime
layer's construction or forward passes.

A custom MoE implementation that participates in the standard MoE tracker can
set `is_moe_layer = True` and use the inherited MoE logging fields. The
implementation must produce the expected tracker data; the flag does not
implement routing, loss scaling, or expert parallelism. Custom hash routing is
outside this initial contract.

`HybridLayer` supplies the standard `MegatronModule.sharded_state_dict` behavior
with the caller's process groups. Parameters and buffers must still obey that
checkpoint contract. Run-config metadata records the external config's fields
and Python class path. Resume constructs the architecture from the supplied
Python definition, as it does for built-in config lists; metadata is not a
replacement for that definition. Keep the defining package available and the
architecture compatible when resuming.
