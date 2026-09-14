<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Multi-Token Prediction (MTP)

Multi-Token Prediction (MTP) extends the prediction scope to several future tokens at each position. An MTP objective adds extra prediction targets, which can improve data efficiency. It may also encourage representations that anticipate later tokens. This implementation predicts additional tokens in sequence and preserves the causal dependency chain at each depth. The following figure illustrates MTP as used in [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3/).

![Diagram of Multi-Token Prediction depth stack: shared embedding, projection, transformer block, and output head per depth](../../images/multi_token_prediction/MTP_implementation.png)

The *k*-th MTP module includes a shared embedding layer, a projection matrix, a Transformer block, and a shared output head. For the *i*-th input token at depth *k - 1*, the implementation combines the representation of the *i*-th token and the embedding of the *(i + K)*-th token with a linear projection. That combined representation is the input to the Transformer block at depth *k*, which produces the output representation.

For more detail, refer to the [DeepSeek-V3 technical report](https://arxiv.org/pdf/2412.19437.pdf).

## Related Arguments

Train `GPTModel`-style models with MTP by setting `mtp_num_layers` to a positive integer.

The following table summarizes MTP configuration fields:

| Item | Description |
| --- | --- |
| `mtp_num_layers` | Number of MTP layers. MTP extends prediction to multiple future tokens at each position. This stack uses `mtp_num_layers` sequential modules to predict that many additional tokens per position. Default: `None`. |
| `mtp_loss_scaling_factor` | Weight for the MTP loss term. The implementation averages MTP losses across depths, multiplies by this factor, and adds the result to the training objective. Default: `0.1`. |
| `mtp_use_repeated_layer` | Reuse one physical MTP layer for every prediction depth. Parameters are shared, while the hidden state, shifted token input, and query are recomputed at each depth. Default: `False`. |
| `mtp_repeated_layer_shared_components` | Components to reuse from the first invocation of a repeated MTP layer at later prediction depths in the same forward. Supported values are `latent_kv` and `sparse_attention_index`; values must be unique and their order has no effect. Omit the option or use an empty list to disable repeated-layer sharing. Default: `None`. |

## Repeated-Layer Sharing

Set `mtp_use_repeated_layer: true`, `mtp_num_layers > 1`, and
`mtp_repeated_layer_shared_components` to reuse selected results from depth 0
at later prediction depths within the same forward and microbatch. Currently,
this requires `experimental_attention_variant: dsa` and the repeated-layer GPT MTP path.

| Shared components | Later prediction depths |
| --- | --- |
| omitted or `[]` | Recompute KV and follow the ordinary top-k schedule. |
| `["latent_kv"]` | Reuse depth-0 KV; follow the ordinary top-k schedule. |
| `["sparse_attention_index"]` | Recompute KV; reuse depth-0 top-k indices. |
| `["latent_kv", "sparse_attention_index"]` | Reuse both depth-0 KV and top-k indices. |

Queries and sparse attention are computed at every depth. Shared KV remains
attached to autograd: consumer gradients accumulate into the depth-0 KV projection.
This is training-time activation reuse, not inference KV caching.

Ordinary IndexShare (`dsa_indexer_topk_freq` and `dsa_indexer_skip_topk_offset`)
still determines how depth 0 obtains its indices, including reuse from an earlier
decoder layer. Repeated-layer index sharing then reuses those indices at later
depths. An ordinary IndexShare source must run earlier in the same PP/VPP
execution segment; cross-PP index sharing remains unsupported.

Index sharing avoids additional indexer evaluations and auxiliary-loss contributions
at later depths; review `dsa_indexer_loss_coeff` when changing this setting.
It disables the combined DSA kernel, but separate fused top-k and sparse-attention
kernels remain eligible. KV-only sharing does not itself disable the combined kernel.

Compatibility:

- Full uniform recompute supports sharing with `recompute_num_layers: 1`.
  With block recompute, MTP retains its existing non-checkpointed fallback.
- Selective recompute supports `mlp`, `moe`, `moe_act`, `shared_experts`,
  `layernorm`, `mhc`, and `mla_up_proj`, subject to their existing requirements.
  With KV sharing, `mla_up_proj` checkpoints Q only and retains the shared KV graph.
- Selective `core_attn` supports index-only sharing, but not KV sharing.
  Set `recompute_modules` explicitly for KV sharing: selective recompute defaults
  to `core_attn`.
- Attention CUDA graph capture is unsupported; MoE-only scopes remain compatible.

## Pipeline Parallel Layout for MTP

MTP supports user-defined placement of MTP layers across pipeline stages through `pipeline_model_parallel_layout`. By default, all MTP layers sit on the last pipeline stage; you can override placement in the layout string.

### MTP Standalone Mode

When MTP layers are placed in a separate virtual pipeline (VPP) stage that is not on the last pipeline rank, the `mtp_standalone` flag is automatically set to `True`. MTP then runs in its own pipeline stage.

### Layout Format

Use `m` for MTP layers in the pipeline layout string. For example:
- `"E|t*3|(t|)*5mL"` - MTP in the last stage
- `"E|t*3|(t|)*4tm|L"` - MTP in the second-to-last stage with a decoder layer
- `"E|t*3|(t|)*3tt|m|L"` - MTP in a standalone stage (second-to-last) with no other layers

### Constraints

- Place all MTP layers in the same virtual pipeline stage.
- Do not place MTP layers on the first pipeline rank.

## Implementation Notes

- For models with MTP layers, the final LayerNorm sits in the stage that contains the last decoder layer, not in the post-process stage. That can change gradient norm reduction slightly in deterministic mode when LayerNorm would otherwise live in another stage. For bitwise alignment, disable gradient norm clipping.
- MTP loss is computed in the post-processing stage.

## Unsupported Combinations

Arbitrary `AttnMaskType` and learned absolute position embeddings are not supported with MTP. Context Parallel support is specific to the attention implementation. For repeated DSA sharing, the supported CP path is DSA with `cp_comm_type=allgather`; it reuses the selected depth-0 global latent KV and/or global sparse-attention index within the same microbatch CP group. Speculative decoding is not yet supported when `mtp_repeated_layer_shared_components` is non-empty because its serial calls do not establish one ordered repeated forward; disable repeated-layer sharing or set `num_speculative_tokens=0`.
