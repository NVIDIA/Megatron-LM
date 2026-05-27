# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_submodules,
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from megatron.training.arguments import parse_args
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


@pytest.mark.skipif(
    not is_te_min_version("1.9.0.dev0"),
    reason="TE Grouped MLP is only supported in TE 1.9.0.dev0 and later.",
)
class TestTEGroupedMLP:

    def setup_method(self, method, use_cpu_initialization=False, swiglu=True):
        Utils.initialize_model_parallel(1, 1)
        num_layers = 1
        self.hidden_size = 16
        self.num_experts = 2
        self.gated_linear_unit = swiglu
        self.activation_func = F.silu if swiglu else F.gelu
        self.use_cpu_initialization = use_cpu_initialization

        tf_config = TransformerConfig(
            num_layers=num_layers,
            hidden_size=self.hidden_size,
            num_attention_heads=4,
            num_moe_experts=self.num_experts,
            use_cpu_initialization=self.use_cpu_initialization,
            add_bias_linear=False,
            gated_linear_unit=self.gated_linear_unit,
            activation_func=self.activation_func,
            bias_activation_fusion=False,
            bf16=True,
            params_dtype=torch.bfloat16,
            moe_router_load_balancing_type="sinkhorn",
            moe_router_topk=1,
        )

        self.fc1_ffn_hidden_size = tf_config.ffn_hidden_size
        self.fc2_ffn_hidden_size = tf_config.ffn_hidden_size
        # If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        if self.gated_linear_unit:
            self.fc1_ffn_hidden_size *= 2

        ## Vanilla sequential GEMM
        # Set random seed for reproducability
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        sequential_submodules = get_submodules(
            get_gpt_layer_local_submodules(self.num_experts, moe_grouped_gemm=False).mlp
        )
        assert isinstance(sequential_submodules, MoESubmodules)
        self.sequential_mlp = MoELayer(tf_config, sequential_submodules)

        self.args = parse_args(ignore_unknown_args=True)
        self.args.bf16 = True
        # Bias is not supported in grouped gemm currently, thus we disable the
        # bias in the linear layer.
        self.args.add_bias_linear = False
        self.sequential_mlp = Float16Module(self.sequential_mlp.config, self.sequential_mlp).module

        ## Grouped GEMM
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        tf_config.moe_grouped_gemm = True
        grouped_submodules = get_submodules(
            get_gpt_layer_with_transformer_engine_submodules(
                self.num_experts, moe_grouped_gemm=True
            ).mlp
        )
        assert isinstance(grouped_submodules, MoESubmodules)
        self.grouped_mlp = MoELayer(tf_config, grouped_submodules)
        assert isinstance(self.grouped_mlp.experts, TEGroupedMLP)
        self.grouped_mlp = Float16Module(self.grouped_mlp.config, self.grouped_mlp).module

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_constructor(self):
        assert isinstance(self.sequential_mlp, MoELayer)
        assert isinstance(self.grouped_mlp, MoELayer)

        num_weights_smm = sum([p.numel() for p in self.sequential_mlp.parameters()])
        num_weights_gmm = sum([p.numel() for p in self.grouped_mlp.parameters()])

        # For the same hyper-parm model configs except the `moe_grouped_gemm`,
        # GroupedGEMM and sequential GEMMs should hold the same number of parms.
        assert num_weights_smm == num_weights_gmm
        # expected num weights: router linear weights+bias + MLP weights(no bias) of all experts
        expected_num_weights = (
            self.hidden_size * self.num_experts
            + self.hidden_size
            * (self.fc1_ffn_hidden_size + self.fc2_ffn_hidden_size)
            * self.num_experts
        )
        assert num_weights_smm == expected_num_weights

        assert torch.equal(self.sequential_mlp.router.weight, self.grouped_mlp.router.weight)

        # weights of linear_fc1: [fc1_ffn_hidden_size, hidden_size]
        # weights of linear_fc2: [hidden_size, fc2_ffn_hidden_size]
        for i in range(self.num_experts):
            assert getattr(self.grouped_mlp.experts.linear_fc1, f"weight{i}").shape == (
                self.fc1_ffn_hidden_size,
                self.hidden_size,
            )
            assert getattr(self.grouped_mlp.experts.linear_fc2, f"weight{i}").shape == (
                self.hidden_size,
                self.fc2_ffn_hidden_size,
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    def test_gpu_forward_backward(self):
        self.sequential_mlp.cuda()
        self.grouped_mlp.cuda()
        # Copy the weights to ensure the same init value
        with torch.no_grad():
            for i in range(self.num_experts):
                self.sequential_mlp.experts.local_experts[i].linear_fc1.weight.copy_(
                    getattr(self.grouped_mlp.experts.linear_fc1, f"weight{i}")
                )
                self.sequential_mlp.experts.local_experts[i].linear_fc2.weight.copy_(
                    getattr(self.grouped_mlp.experts.linear_fc2, f"weight{i}")
                )
        # [sequence length, batch size, hidden size]
        seq_len = 32
        batch_size = 2
        hidden_states = torch.rand(
            (seq_len, batch_size, self.hidden_size),
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=True,
        )
        hidden_states.retain_grad()

        output_smm, _ = self.sequential_mlp(hidden_states)
        output_smm.mean().backward()
        smm_results = [output_smm, hidden_states.grad]
        for i in range(self.num_experts):
            smm_results.append(self.sequential_mlp.experts.local_experts[i].linear_fc1.weight.grad)
            smm_results.append(self.sequential_mlp.experts.local_experts[i].linear_fc2.weight.grad)

        hidden_states.grad = None
        output_gmm, _ = self.grouped_mlp(hidden_states)
        output_gmm.mean().backward()
        gmm_results = [output_gmm, hidden_states.grad]
        for i in range(self.num_experts):
            gmm_results.append(getattr(self.grouped_mlp.experts.linear_fc1, f"weight{i}").grad)
            gmm_results.append(getattr(self.grouped_mlp.experts.linear_fc2, f"weight{i}").grad)

        for smm_result, gmm_result in zip(smm_results, gmm_results):
            torch.testing.assert_close(smm_result, gmm_result)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    def test_gpu_forward_backward_with_no_tokens_allocated(self):
        """Test the case when no token is allocated for groupedGEMM kernels."""
        self.grouped_mlp.cuda()
        num_allocated_tokens = 0
        tokens_per_expert = torch.zeros(self.num_experts, dtype=torch.int32)
        hidden_states = torch.rand((num_allocated_tokens, self.hidden_size), dtype=torch.bfloat16)
        hidden_states = hidden_states.cuda()
        probs = torch.rand((num_allocated_tokens,), dtype=torch.float32)
        probs = probs.cuda()
        output, _ = self.grouped_mlp.experts(
            hidden_states, tokens_per_expert=tokens_per_expert, permuted_probs=probs
        )
        assert torch.equal(output, torch.zeros_like(output))
        assert output.shape == (num_allocated_tokens, self.hidden_size)

        output.mean().backward()
        for i in range(self.num_experts):
            assert getattr(self.grouped_mlp.experts.linear_fc1, f"weight{i}").grad is not None
            assert getattr(self.grouped_mlp.experts.linear_fc2, f"weight{i}").grad is not None
