# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.experts import SequentialMLP
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class TestSequentialExpertPadding:
    def setup_method(self):
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("enabled,grouped", [(False, False), (True, False), (True, True)])
    @pytest.mark.parametrize("first_count", [0, 3, 16, 17])
    def test_expert_gemm_rows_preserve_storage_contract(self, enabled, grouped, first_count):
        config = TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            ffn_hidden_size=64,
            moe_ffn_hidden_size=64,
            num_moe_experts=2,
            moe_router_topk=1,
            moe_router_pre_softmax=True,
            add_bias_linear=False,
            gated_linear_unit=True,
            activation_func=torch.nn.functional.silu,
            bias_activation_fusion=False,
            params_dtype=torch.bfloat16,
            use_accuracy_compatible=enabled,
            dsa_accuracy_compatible=True,
            moe_grouped_gemm=grouped,
        )
        experts = SequentialMLP(
            2,
            config,
            MLPSubmodules(linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear),
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        ).cuda()
        observed = []

        def record_rows(module, inputs):
            tokens, probs = inputs
            observed.append((tokens.shape[0], probs.shape[0]))

        handles = [
            expert.register_forward_pre_hook(record_rows) for expert in experts.local_experts
        ]
        counts = torch.tensor([first_count, 19], dtype=torch.int64)
        tokens = torch.randn(
            first_count + 19, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        probs = torch.ones(first_count + 19, device="cuda", dtype=torch.float32, requires_grad=True)
        try:
            output, bias = experts(tokens, counts, probs)
            expected_rows = 32 if enabled and not grouped and 0 < first_count < 17 else first_count
            assert observed == [(expected_rows, expected_rows), (19, 19)]
            assert output.shape == tokens.shape
            assert bias is None
            output.float().sum().backward()
            assert tokens.grad.shape == tokens.shape
            assert probs.grad.shape == probs.shape
            assert torch.isfinite(tokens.grad).all()
            assert torch.isfinite(probs.grad).all()
        finally:
            for handle in handles:
                handle.remove()
