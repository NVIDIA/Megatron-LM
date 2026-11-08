# Defining HybridModel layers outside Core

`HybridModel` accepts layers implemented in a downstream package through
`hybrid_layer_config_list`. A layer's configuration references its implementation
class directly. No layer symbol, registration, or module specification is needed
for the new layer.

The downstream package uses the existing `TransformerConfig` and
`MegatronModule` classes. Core constructs the supplied layer class without
importing the downstream package. The same mechanism can be used by model code
in `megatron.bridge` when that package is introduced.

## Define a config and a layer

This small channel-scaling example demonstrates the interface. Put both classes
in an importable module in your package:

```python
from dataclasses import dataclass
from typing import ClassVar

import torch

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig


class GainLayer(MegatronModule):
    def __init__(
        self,
        config,
        *,
        pg_collection,
        layer_number,
        pp_layer_offset=0,
        is_mtp_layer=False,
        mtp_layer_number=None,
        name=None,
    ) -> None:
        super().__init__(config)
        if config.initial_gain <= 0:
            raise ValueError("initial_gain must be positive")
        self.layer_number = layer_number
        self.pg_collection = pg_collection
        self.tp_group = pg_collection.tp
        self.is_mtp_layer = is_mtp_layer
        device = "cpu" if config.use_cpu_initialization else torch.cuda.current_device()
        self.gain = torch.nn.Parameter(
            torch.full(
                (config.hidden_size,),
                config.initial_gain,
                dtype=config.params_dtype,
                device=device,
            )
        )

    @classmethod
    def estimate_flops(
        cls, config, *, total_real_tokens_in_batch, seqlen_squared_sum_in_batch
    ):
        return total_real_tokens_in_batch * config.hidden_size

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


@dataclass
class GainConfig(TransformerConfig):
    layer_type: ClassVar[type[MegatronModule]] = GainLayer
    initial_gain: float = 1.0
```

`layer_type` binds the config to a `MegatronModule` subclass. Model construction
clones the config for each physical layer. `HybridStack` calls that class with the config,
explicit process groups, and ordinary layer-number and pipeline/MTP placement
arguments. Preserve the supplied config and layer number on the module. Keep
live modules, process groups, and other runtime resources out of config fields.

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
layer-specific fields in the layer constructor; copying a normalized config with
`from_config` does not rerun `__post_init__`.

The standard training entry point uses the same list in
`megatron.training.models.hybrid.HybridModelConfig`. The existing
`HybridModelBuilder` forwards it to Core. Architecture config objects do not need
to be placed on argparse.

## Supported execution contract

The initial extension supports ordinary hidden-state streams with TP=1 and CP=1,
including eager training and full-layer recomputation (`uniform` and `block`).
Pipeline selection and MTP use the existing `PipelineSplit` and `MTPSplit`
mechanisms. Each selected decoder or MTP occurrence receives its own config and
module; the constructor arguments identify its layer and MTP placement. As with
built-in config lists, repeated MTP sections must refer to the same source config
objects in the same order.

For custom layers, Core's `HybridModel` and `HybridStack` reject sequence
parallelism, TP communication overlap, quantized execution, CUDA graphs,
mHC/wide residual streams, shortcut-MoE, overlapped expert communication,
CPU/activation offloading, and selective recomputation.
Inference contexts are also rejected. Those features require their own layer
integration and validation; binding a layer class does not enable them.
Distributed wrappers still require their normal supported configurations.

A custom attention module can set the class attribute
`uses_shared_rotary_embeddings = True` to use the model's shared RoPE. Its
`kv_channels` must match the model's shared rotary width. Otherwise, the layer
owns any positional-embedding computation it needs.

## Metrics and checkpoints

Implement the optional `MegatronModule.estimate_flops` class method on a custom
layer to return its **forward** FLOPs for the supplied global-batch token statistics.
The training FLOP counter calls `layer_type.estimate_flops(config, ...)`, applies
its forward/backward factor, and includes every decoder and MTP occurrence from
the constructed model's full config list. Built-in layers use the existing
training formulas.
Missing estimates raise an explicit error instead of applying an
unrelated built-in formula. Keep the estimator pure: model setup may call it to
validate availability before training starts, independently of the runtime
layer's construction or forward passes.

A custom MoE implementation that participates in the standard MoE tracker can
set `is_moe_layer = True` on its module class and use the config's inherited MoE
logging fields. The implementation must produce the expected tracker data; the
flag does not implement routing, loss scaling, or expert parallelism. Custom hash routing is
outside this initial contract.

Use the standard `MegatronModule.sharded_state_dict` behavior and keep the
module's `tp_group` bound to the supplied group. `HybridStack` supplies the
data/context-parallel group in checkpoint metadata. When checkpointing a layer
directly, pass `metadata={"dp_cp_group": pg_collection.dp_cp}` explicitly.
Parameters and buffers must still obey the checkpoint contract. Run-config
metadata records the external config's fields and Python class path. Resume
constructs the architecture from the supplied Python definition, as it does for
built-in config lists; metadata is not a
replacement for that definition. Keep the defining package available and the
architecture compatible when resuming.
