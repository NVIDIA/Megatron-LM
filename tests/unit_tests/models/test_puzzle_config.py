# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Basic forward/backward test for a tiny Puzzle-like config-list model."""

import torch

from megatron.core.activations import squared_relu
from megatron.core.models.hybrid import MTPSplit
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_layer_config import MambaLayerConfig
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention_layer_config import AttentionLayerConfig
from megatron.core.transformer.moe.moe_layer_config import MoELayerConfig
from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def test_small_puzzle_forward_backward():
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    MTPLossLoggingHelper.tracker.clear()
    try:
        config = TransformerConfig(
            num_layers=4,
            hidden_size=128,
            num_attention_heads=4,
            num_query_groups=2,
            mamba_num_heads=4,
            mamba_head_dim=64,
            mamba_state_dim=16,
            mamba_num_groups=1,
            num_moe_experts=4,
            moe_ffn_hidden_size=64,
            moe_router_topk=1,
            moe_router_score_function="sigmoid",
            moe_router_load_balancing_type="none",
            moe_latent_size=32,
            moe_shared_expert_intermediate_size=64,
            moe_shared_expert_overlap=False,
            moe_grouped_gemm=True,
            activation_func=squared_relu,
            normalization="RMSNorm",
            add_bias_linear=False,
            bf16=True,
            params_dtype=torch.bfloat16,
            gradient_accumulation_fusion=False,
            is_hybrid_model=True,
        )
        attention = AttentionLayerConfig.from_config(config)
        small_moe = MoELayerConfig.from_config(config)
        large_moe = MoELayerConfig.from_config(config)
        large_moe.moe_ffn_hidden_size = 128
        large_moe.moe_router_topk = 2
        model = HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
            vocab_size=128,
            max_sequence_length=8,
            hybrid_layer_config_list=[
                MambaLayerConfig.from_config(config),
                small_moe,
                attention,
                large_moe,
                MTPSplit,
                attention,
                large_moe,
            ],
        ).cuda()
        assert model.config.mtp_num_layers == 1

        tokens = torch.arange(8, device="cuda").unsqueeze(0)
        losses = model(
            input_ids=tokens, position_ids=tokens, attention_mask=None, labels=tokens + 1
        )
        assert losses.shape == tokens.shape
        assert torch.isfinite(losses).all()
        losses.mean().backward()
        assert model.embedding.word_embeddings.weight.grad is not None
        assert any(parameter.grad is not None for parameter in model.mtp.parameters())
    finally:
        MTPLossLoggingHelper.tracker.clear()
        Utils.destroy_model_parallel()
