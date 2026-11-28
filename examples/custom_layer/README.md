# Custom layers in HybridModel

[`example.py`](example.py) defines a small `HybridModel` containing a built-in
attention layer followed by a custom `GainLayer`. The custom layer is defined locally, outside of `megatron/core`. The layer learns one
scale per hidden channel and returns `hidden_states * gain`.

## Run Example

Use one CUDA GPU and a prepared Megatron **dev** container with PyTorch and
Transformer Engine. HybridModel uses Transformer Engine for its built-in layers.
See the [container setup guide](../../skills/mcore-build-and-dependency/SKILL.md)
for preparing that environment.

From the repository root inside the container, run:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=1 \
    -m examples.custom_layer.example
```

The script initializes process groups, builds the model, and uses synthetic token
IDs and next-token labels for one SGD step. No dataset or tokenizer is needed.
It checks that the loss is finite, the custom layer receives finite nonzero
gradients, and its parameters change after the update. A successful run prints
the loss and gradient norm, followed by:

```text
Success: HybridModel ran forward/backward and updated the custom layer.
```

## How to implement a custom layer

1. **Extend `MegatronModule`.** Call `super().__init__(config)` and retain the
   supplied `layer_number` and `pg_collection.tp` as `self.tp_group`. The constructor
   also accepts `pp_layer_offset`, `is_mtp_layer`, `mtp_layer_number`, and `name`;
   this pointwise layer does not need them.
2. **Implement `forward`.** Accept `hidden_states` and the optional keyword inputs
   shown in the example: `attention_mask`, `rotary_pos_emb`, `packed_seq_params`,
   `padding_mask`, and `input_ids`. Return a single tensor with the same shape as
   `hidden_states`, rather than an output/context tuple. `GainLayer` operates on
   the last dimension and ignores the optional inputs. Token IDs can be `None`,
   including on downstream pipeline stages and in MTP stacks.
3. **Bind the implementation through its config.** Extend `TransformerConfig`
   with a `layer_type` class variable referencing the actual layer class, plus
   any custom settings. `GainConfig` adds `gain_init`.
4. **Compose the model.** Pass an ordered `hybrid_layer_config_list` to
   `HybridModel`, mixing built-in and custom configs:

   ```python
   hybrid_layer_config_list=[
       AttentionLayerConfig.from_config(config),
       GainConfig.from_config(config),
   ]
   ```

5. **Estimate forward FLOPs.** `GainLayer.estimate_flops` counts one
   multiplication per token and hidden channel:
   `total_real_tokens_in_batch * config.hidden_size`. This pure class method
   performs no computation on tensors and does not include backward FLOPs.
   The example does not print FLOP estimates.

`from_config` copies the normalized root settings and supplies defaults for new
fields. HybridModel creates independent config copies and modules for physical
layers. Set `num_layers` to the number of decoder layers in the list.

No module spec, symbol registration, or import into Core is needed. Custom layers
use this Python config-list API; legacy pattern strings describe built-in layers
only. The custom layer owns its computation, including any normalization or
residual connection it needs.
