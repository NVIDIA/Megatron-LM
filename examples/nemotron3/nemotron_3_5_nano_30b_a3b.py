# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Architecture-only Nemotron 3.5 Nano 30B-A3B HybridModel example.

The direct layer definition intentionally reuses one configuration per layer
family. Nemotron 3.5 Nano is therefore a compatibility example for the direct
API, not an example of occurrence-specific layer heterogeneity.

Model dimensions are based on:
https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/main/config.json
"""

from copy import deepcopy

import torch

from megatron.core.activations import squared_relu
from megatron.core.models.hybrid.hybrid_architecture import HybridLayerSpec, PipelineSplit
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.transformer import TransformerConfig
from megatron.training.models.hybrid import HybridModelConfig


def _transformer_config() -> TransformerConfig:
    """Return the model-wide Nano transformer configuration."""

    return TransformerConfig(
        num_layers=52,
        hidden_size=2688,
        num_attention_heads=32,
        num_query_groups=2,
        kv_channels=128,
        ffn_hidden_size=1856,
        moe_ffn_hidden_size=1856,
        num_moe_experts=128,
        moe_router_topk=6,
        moe_router_num_groups=1,
        moe_router_group_topk=1,
        moe_router_topk_scaling_factor=2.5,
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_load_balancing_type="seq_aux_loss",
        moe_shared_expert_intermediate_size=3712,
        moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=True,
        mamba_state_dim=128,
        mamba_head_dim=64,
        mamba_num_heads=64,
        mamba_num_groups=8,
        mtp_num_layers=2,
        mtp_use_repeated_layer=True,
        activation_func=squared_relu,
        gated_linear_unit=False,
        add_bias_linear=False,
        add_qkv_bias=False,
        normalization="RMSNorm",
        layernorm_epsilon=1.0e-5,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        init_method_std=0.02,
        is_hybrid_model=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        pipeline_model_parallel_size=2,
    )


def make_model_config() -> HybridModelConfig:
    """Define Nano with direct specs split into PP2/VPP2 logical chunks."""

    transformer = _transformer_config()
    submodules = hybrid_stack_spec.submodules

    # Aliases make the family-uniform nature of Nano explicit. The architecture
    # resolver isolates the configuration of every resolved occurrence.
    mamba = HybridLayerSpec(module_spec=submodules.mamba_layer, config=deepcopy(transformer))
    attention = HybridLayerSpec(
        module_spec=submodules.attention_layer, config=deepcopy(transformer)
    )
    moe = HybridLayerSpec(module_spec=submodules.moe_layer, config=deepcopy(transformer))

    # Chunks are VPP-major then PP-rank: (vp0, pp0), (vp0, pp1),
    # (vp1, pp0), (vp1, pp1). Each chunk contains exactly 13 layers.
    chunk_0 = [mamba, moe] * 2 + [mamba, attention, moe] + [mamba, moe] * 2 + [mamba, attention]
    chunk_1 = [moe, mamba] * 3 + [attention] + [moe, mamba] * 3
    chunk_2 = [attention] + [moe, mamba] * 3 + [attention] + [moe, mamba] * 2 + [moe]
    chunk_3 = [mamba, moe, mamba, attention, moe] + [mamba, moe] * 4

    layer_specs = [
        chunk_0,
        PipelineSplit(),
        chunk_1,
        PipelineSplit(),
        chunk_2,
        PipelineSplit(),
        chunk_3,
    ]

    mtp_attention = HybridLayerSpec(
        module_spec=submodules.attention_layer, config=deepcopy(transformer)
    )
    mtp_moe = HybridLayerSpec(module_spec=submodules.moe_layer, config=deepcopy(transformer))

    return HybridModelConfig(
        transformer=transformer,
        layer_specs=layer_specs,
        mtp_layer_specs=[mtp_attention, mtp_moe],
        vocab_size=131072,
        seq_length=262144,
        position_embedding_type="none",
        parallel_output=True,
        share_embeddings_and_output_weights=False,
    )


__all__ = ["make_model_config"]
