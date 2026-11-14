# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import os

import pytest
import torch

from megatron.core.extensions.transformer_engine import TERowParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.custom_layers.batch_invariant_kernels import set_batch_invariant_mode
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def test_dense_tp_batch_invariant_replay():
    if int(os.environ.get("WORLD_SIZE", "1")) < 4:
        pytest.skip("Requires four tensor-parallel ranks")
    Utils.initialize_model_parallel(4, 1)
    try:
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=256,
            num_attention_heads=4,
            tensor_model_parallel_size=4,
            params_dtype=torch.bfloat16,
            bf16=True,
            batch_invariant_mode=True,
            attention_backend=AttnBackend.flash,
            flash_attention_version=4,
            attention_dropout=0.0,
            hidden_dropout=0.0,
        )
        with set_batch_invariant_mode(True, backend="te_native"):
            layer = TERowParallelLinear(
                256,
                256,
                config=config,
                init_method=torch.nn.init.normal_,
                bias=False,
                input_is_parallel=True,
                skip_bias_add=True,
                is_expert=False,
            ).eval()
            assert layer.symmetric_ar_type == "two_shot"
            assert config.symmetric_ar_type is None
            torch.manual_seed(100 + torch.distributed.get_rank())
            inputs = torch.randn(256, 1, 64, device="cuda", dtype=torch.bfloat16)
            with torch.no_grad():
                expected = layer(inputs[:1])[0].clone()
                for _ in range(3):
                    for rows in (1, 64, 256):
                        actual = layer(inputs[:rows])[0][:1]
                        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    finally:
        Utils.destroy_model_parallel()
