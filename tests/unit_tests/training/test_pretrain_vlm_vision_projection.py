# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""pretrain_vlm.py builds its vision projection as a dense MLP, also with MoE language layers."""

import pytest
import torch

from megatron.core.models.multimodal.llava_spec import (
    decoder_model_with_local_default_spec,
    decoder_model_with_transformer_engine_default_spec,
)
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class TestPretrainVlmVisionProjection:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("use_te", [False, True])
    @pytest.mark.parametrize("num_experts", [None, 2])
    def test_projection_is_dense_mlp(self, use_te, num_experts):
        import pretrain_vlm

        spec_fn = (
            decoder_model_with_transformer_engine_default_spec
            if use_te
            else decoder_model_with_local_default_spec
        )
        language_spec = spec_fn(num_experts=num_experts)

        submodules = pretrain_vlm.get_vision_projection_submodules(language_spec, use_te=use_te)

        assert isinstance(submodules, MLPSubmodules)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            num_moe_experts=num_experts,
            add_bias_linear=False,
            use_cpu_initialization=True,
        )
        projector = MultimodalProjector(
            config=config, submodules=submodules, projector_type="mlp", input_size=128
        ).cuda()
        output = projector(torch.zeros((3, 2, 128), device="cuda"))
        assert output.shape == torch.Size([3, 2, 64])
