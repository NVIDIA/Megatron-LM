# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exercise Shortcut MoE and wide residuals through the inference graph manager."""

import torch

from megatron.core import parallel_state
from megatron.core.activations import squared_relu
from megatron.core.models.hybrid.hybrid_layer_specs import (
    wide_residual_gated_delta_product_inference_stack_spec,
)
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.models.hybrid.shortcut_block import ShortcutMoEBlock
from megatron.core.transformer.attention import Attention
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.wide_residual_config import WideResidualConfig
from tests.unit_tests.inference.engines.test_gdp_cuda_graph_e2e import (
    FLASH_ATTENTION_VERSION,
    MAX_SEQ_LEN,
    VOCAB_SIZE,
)
from tests.unit_tests.inference.engines.test_gdp_cuda_graph_e2e import (
    TestGDPCudaGraphE2E as _GDPGraphTest,
)
from tests.unit_tests.test_utilities import Utils


class TestShortcutMoECudaGraphE2E(_GDPGraphTest):
    """Greedy tokens must match eager inference for a GDP/Shortcut-MoE pair."""

    @classmethod
    def setup_class(cls):
        Utils.initialize_distributed()
        Utils.initialize_model_parallel(
            expert_model_parallel_size=torch.distributed.get_world_size()
        )

    def _create_model(self):
        torch.manual_seed(1234)
        config = TransformerConfig(
            params_dtype=torch.bfloat16,
            num_layers=3,
            hidden_size=256,
            num_attention_heads=16,
            ffn_hidden_size=512,
            activation_func=squared_relu,
            mamba_num_heads=8,
            mamba_head_dim=32,
            mamba_num_groups=8,
            mamba_state_dim=64,
            gdp_num_householder=2,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_pre_softmax=True,
            moe_router_dtype="fp32",
            moe_token_dispatcher_type="alltoall",
            moe_shortcut_connection=True,
            moe_shared_expert_intermediate_size=512,
            moe_shortcut_post_norm=True,
            wide_residual=WideResidualConfig(num_streams=3),
            use_cpu_initialization=True,
            transformer_impl="inference_optimized",
            normalization="RMSNorm",
            cuda_graph_impl="local",
            inference_cuda_graph_scope="block",
            inference_rng_tracker=True,
            tensor_model_parallel_size=1,
            expert_model_parallel_size=parallel_state.get_expert_model_parallel_world_size(),
            pipeline_model_parallel_size=1,
            pipeline_dtype=torch.bfloat16,
            add_bias_linear=False,
            is_hybrid_model=True,
            attention_dropout=0.0,
            attention_backend=AttnBackend.flash,
            flash_attention_version=FLASH_ATTENTION_VERSION,
        )
        model = HybridModel(
            config=config,
            hybrid_stack_spec=wide_residual_gated_delta_product_inference_stack_spec,
            vocab_size=VOCAB_SIZE,
            max_sequence_length=MAX_SEQ_LEN,
            parallel_output=True,
            hybrid_layer_pattern="ME*",
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
        ).cuda()
        assert any(isinstance(layer, ShortcutMoEBlock) for layer in model.decoder.layers)
        for param in model.parameters():
            param.data = param.data.to(config.params_dtype)
        model.eval()
        for module in model.modules():
            if isinstance(module, Attention):
                module.batch_invariant_mode = True
        return model
