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

Context Parallel (CP), arbitrary `AttnMaskType`, and learned absolute position embeddings are not supported with MTP.

## K3-style Hybrid Attention Residuals and MTP

Hybrid MTP can combine Kimi Delta Attention (`K`), gated MLA (`+`), dense MLP
(`-`), and MoE (`E`) with `enable_attention_residuals=True`. This integrates the
K3 residual structure into MCore's existing MTP embedding shift, projection,
prediction head, and auxiliary loss. It does not claim to reproduce K3's
unpublished training-time MTP topology. The source reference is
[Kimi-K3 at f831ab6](https://huggingface.co/moonshotai/Kimi-K3/tree/f831ab66814297da540d832a5235f8e904f29d06),
`modeling_kimi_linear.py`, specifically `_apply_attn_res` and
`KimiDecoderLayer._forward_attn_residual`.

### Residual state and layer counting

Hybrid patterns count attention and FFN as separate entries: K3's block size of
12 physical decoder layers maps to `attn_res_block_layers=24`, not 12. For
example, a 16-pair training proxy can use
`K-KEKE+EKEKEKE+EKEKEKE+EKEKEKE+E/+E` with `num_layers=32` and
`mtp_num_layers=1`. Append another `/+E` for two MTP depths. The trunk may use
`|` for pipeline boundaries; MTP remains on the last stage. FLA is the default
AttnRes backend; `attn_res_impl=compile` explicitly selects the compiled path.
For K3's MLA NoPE, set `no_rope_freq=1` and keep `qk_pos_emb_head_dim=64`:
these shared key/query channels are still projected, but are not rotated.
MLA's `no_rope_freq` path currently supports training, not cached inference.
Use `qk_layernorm=True` for K3's query/key-value latent norms. SiTU-GLU can use
the existing native activation path (`use_te_activation_func=False`) on images
without TE's SiTUGLU operation; selecting `situ_glu` preserves that choice.
KDA's short convolutions use SiLU independently of the FFN activation, including
when the FFN uses SiTU-GLU. Both native and fused convolution paths follow this
rule from the reference implementation.

The trunk exports an immutable tuple of the embedding, completed residual
blocks, and final partial **before** its output aggregation and normalization.
Every MTP depth reads that same tuple. Its projected embedding/previous-hidden
combination initializes a fresh local partial, and each nested entry attends
over the trunk tuple plus that partial. The entry's bias/dropout contribution
is added directly to the partial, without reconstructing a delta by subtracting
BF16 tensors. Normal module hooks and selective recomputation are preserved.

At the end of each MTP depth, aggregate the trunk tuple plus its final partial
once, then apply the existing MTP output norm. The result feeds the next depth's
projection, but is never appended to the trunk tuple. `mtp_detach_heads` detaches
the hidden state, all trunk sources, and shifted embeddings while retaining
gradients for MTP parameters. Repeated MTP layers share parameters, not mutable
residual history. There is no module-level cache of sources across microbatches.

Start training validation without activation offloading or recomputation, then
enable existing selective recomputation independently. This support does not
extend full-layer recomputation, CUDA graphs, standalone MTP pipeline placement,
or the existing activation-offload module allowlist. The broader offloading
follow-up is developed on a separate branch.

The K3-specific GPU tests include independent native-Torch output/input/parameter
gradient comparisons, full-dimension NoPE MLA, and ten optimizer updates with
real KDA/MLA/MoE and two MTP depths. Reduced end-to-end training has been verified
for single-GPU MTP1/MTP2 and TP2 with sequence parallelism. PP2 validation was
interrupted by cluster SSH loss; K3 end-to-end selective-recompute, compile, and
checkpoint-resume runs remain pending. This is not a complete distributed
training support matrix.
