# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestDotProductAttention:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_forward_with_distinct_key_and_value_channels(self):
        config = TransformerConfig(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            attention_dropout=0.0,
            masked_softmax_fusion=False,
        )
        attention = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            k_channels=24,
            v_channels=16,
        ).cuda()

        sequence_length = 8
        micro_batch_size = 2
        query = torch.ones(
            sequence_length,
            micro_batch_size,
            config.num_attention_heads,
            24,
            device="cuda",
        )
        key = torch.ones_like(query)
        value = torch.ones(
            sequence_length,
            micro_batch_size,
            config.num_attention_heads,
            16,
            device="cuda",
        )
        attention_mask = torch.zeros(
            1,
            1,
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device="cuda",
        )

        output = attention(query, key, value, attention_mask)

        assert output.shape == (sequence_length, micro_batch_size, 64)
