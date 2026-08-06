# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Architecture-only NVIDIA Nemotron Labs 3 Puzzle 75B-A9B example.

Unlike Nano, Puzzle is heterogeneous within the MoE family: every MoE
occurrence carries the expert width and router top-k published for that block.

Model dimensions and ordered block configurations are based on:
https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-BF16/blob/main/config.json
"""

from copy import deepcopy

import torch

from megatron.core.activations import squared_relu
from megatron.core.models.hybrid.hybrid_architecture import HybridLayerSpec, PipelineSplit
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.transformer import TransformerConfig
from megatron.training.models.hybrid import HybridModelConfig


def _transformer_config() -> TransformerConfig:
    """Return the model-wide Puzzle transformer configuration."""

    return TransformerConfig(
        num_layers=88,
        hidden_size=4096,
        num_attention_heads=32,
        num_query_groups=2,
        kv_channels=128,
        ffn_hidden_size=21504,
        moe_ffn_hidden_size=1280,
        num_moe_experts=512,
        moe_router_topk=4,
        moe_router_num_groups=1,
        moe_router_group_topk=1,
        moe_router_topk_scaling_factor=5.0,
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_load_balancing_type="seq_aux_loss",
        moe_shared_expert_intermediate_size=5376,
        moe_shared_expert_overlap=True,
        moe_latent_size=1024,
        moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=True,
        mamba_state_dim=96,
        mamba_head_dim=64,
        mamba_num_heads=128,
        mamba_num_groups=8,
        mtp_num_layers=1,
        mtp_use_repeated_layer=False,
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


def _moe_layer(base_config: TransformerConfig, width: int, topk: int) -> HybridLayerSpec:
    """Create one occurrence-specific stock MoE layer."""

    config = deepcopy(base_config)
    config.moe_ffn_hidden_size = width
    config.moe_router_topk = topk
    return HybridLayerSpec(module_spec=hybrid_stack_spec.submodules.moe_layer, config=config)


def make_model_config() -> HybridModelConfig:
    """Define Puzzle with direct specs split into PP2/VPP2 logical chunks."""

    transformer = _transformer_config()
    submodules = hybrid_stack_spec.submodules
    mamba = HybridLayerSpec(module_spec=submodules.mamba_layer, config=deepcopy(transformer))
    attention = HybridLayerSpec(
        module_spec=submodules.attention_layer, config=deepcopy(transformer)
    )

    def moe(width: int, topk: int) -> HybridLayerSpec:
        return _moe_layer(transformer, width, topk)

    # Chunks are VPP-major then PP-rank. The inline MoE arguments are the
    # published blockwise (expert width, active experts) values.
    chunk_0 = [
        mamba,
        moe(1280, 4),
        mamba,
        moe(1280, 8),
        mamba,
        moe(1280, 10),
        mamba,
        attention,
        moe(1280, 8),
        mamba,
        moe(1280, 8),
        mamba,
        moe(1280, 8),
        mamba,
        moe(1280, 12),
        mamba,
        attention,
        moe(1280, 8),
        mamba,
        moe(1280, 10),
        mamba,
        moe(1280, 8),
    ]
    chunk_1 = [
        mamba,
        moe(2688, 12),
        mamba,
        attention,
        moe(1536, 14),
        mamba,
        moe(2688, 12),
        mamba,
        moe(1536, 12),
        mamba,
        moe(1536, 12),
        mamba,
        moe(2688, 12),
        mamba,
        attention,
        moe(2688, 12),
        mamba,
        moe(2688, 12),
        mamba,
        moe(1536, 10),
        mamba,
        moe(2688, 12),
    ]
    chunk_2 = [
        mamba,
        moe(2688, 12),
        mamba,
        attention,
        moe(1792, 12),
        mamba,
        moe(1792, 14),
        mamba,
        moe(1280, 10),
        mamba,
        moe(1280, 10),
        mamba,
        moe(1280, 12),
        mamba,
        attention,
        moe(1280, 8),
        mamba,
        moe(1280, 12),
        mamba,
        moe(1280, 10),
        mamba,
        moe(1280, 8),
    ]
    chunk_3 = [
        mamba,
        moe(1280, 8),
        mamba,
        attention,
        moe(1280, 10),
        mamba,
        moe(1280, 10),
        mamba,
        moe(1280, 10),
        mamba,
        moe(1280, 12),
        mamba,
        attention,
        moe(1280, 12),
        mamba,
        moe(1280, 14),
        mamba,
        moe(1280, 16),
        mamba,
        moe(1792, 18),
        mamba,
        moe(2048, 18),
    ]

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
    mtp_moe = _moe_layer(transformer, width=2688, topk=22)

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
