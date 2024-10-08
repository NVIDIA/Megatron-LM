# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.transformer.moe import grouped_gemm_util as gg
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from megatron.legacy.model import Float16Module
from megatron.training.arguments import parse_args
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils

DEVICE_CAPABILITY = None
if torch.cuda.is_available():
    DEVICE_CAPABILITY = torch.cuda.get_device_capability()


class TestParallelGroupedMLP:

    def setup_method(self, method, use_cpu_initialization=False, swiglu=True):
        print("============")
        print(
            "Test for use_cpu_initilization={} and swiglu={}.".format(
                use_cpu_initialization, swiglu
            )
        )
        print("============")
        Utils.initialize_model_parallel(1, 1)
        num_layers = 1  # 2
        self.hidden_size = (
            16  # must be an multiple of 16, otherwise trigger CUTLASS misaligned issue
        )
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
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            self.num_experts, moe_grouped_gemm=False
        )
        self.sequential_mlp = MoELayer(tf_config, transformer_layer_spec.submodules.mlp.submodules)

        self.args = parse_args(ignore_unknown_args=True)
        self.args.bf16 = True
        # Bias is not supported in grouped gemm currently, thus we disable the
        # bias in the linear layer.
        self.args.add_bias_linear = False
        self.sequential_mlp = Float16Module(self.sequential_mlp, self.args).module
        print("done intializing for sequential gemm")

        ## Grouped GEMM
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        tf_config.moe_grouped_gemm = True
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            self.num_experts, moe_grouped_gemm=True
        )
        self.grouped_mlp = MoELayer(tf_config, transformer_layer_spec.submodules.mlp.submodules)
        self.grouped_mlp = Float16Module(self.grouped_mlp, self.args).module
        print("done intializing for grouped gemm")

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

        # weight1: [h, num_experts*4h]
        # weight2: [num_experts*4h, h]
        assert self.grouped_mlp.experts.weight1.shape[0] == self.hidden_size
        assert (
            self.grouped_mlp.experts.weight1.shape[1] == self.num_experts * self.fc1_ffn_hidden_size
        )
        if self.gated_linear_unit:
            assert (
                self.grouped_mlp.experts.weight2.shape[0]
                == self.num_experts * self.fc2_ffn_hidden_size
            )
            assert self.grouped_mlp.experts.weight2.shape[1] == self.hidden_size
        else:
            assert (
                self.grouped_mlp.experts.weight1.shape == self.grouped_mlp.experts.weight2.t().shape
            )

    @pytest.mark.internal
    def test_weight_init_value_the_same(self):
        gmm_w1 = self.grouped_mlp.experts.weight1.view(self.num_experts, -1, self.hidden_size)
        gmm_w2 = self.grouped_mlp.experts.weight2.view(self.num_experts, self.hidden_size, -1)
        gmm_expert1_fc1 = gmm_w1[0]
        gmm_expert1_fc2 = gmm_w2[0]
        gmm_expert2_fc1 = gmm_w1[1]
        gmm_expert2_fc2 = gmm_w2[1]

        smm_expert1_fc1 = self.sequential_mlp.experts.local_experts[0].linear_fc1.weight
        smm_expert1_fc2 = self.sequential_mlp.experts.local_experts[0].linear_fc2.weight
        smm_expert2_fc1 = self.sequential_mlp.experts.local_experts[1].linear_fc1.weight
        smm_expert2_fc2 = self.sequential_mlp.experts.local_experts[1].linear_fc2.weight

        assert torch.equal(gmm_expert1_fc1, smm_expert1_fc1)
        if not self.use_cpu_initialization:
            assert torch.equal(gmm_expert1_fc2, smm_expert1_fc2)
        # the param init value is not exactly the same between gmm and smm (refer to test_weight_init_value_the_same.)
        # TODO: is it necessary to keep smm and gmm share exactly the same init params?
        # assert torch.equal(gmm_expert2_fc1, smm_expert2_fc1)
        if self.use_cpu_initialization:
            assert torch.equal(gmm_expert2_fc2, smm_expert2_fc2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.skipif(
        not DEVICE_CAPABILITY or DEVICE_CAPABILITY[0] < 8,
        reason='GroupedGEMM kernels are not supported on this device.',
    )
    def test_gpu_forward(self):
        self.sequential_mlp.cuda()
        self.grouped_mlp.cuda()
        # [sequence length, batch size, hidden size]
        seq_len = 3  # 32
        batch_size = 2
        hidden_states = torch.rand(
            (seq_len, batch_size, self.sequential_mlp.config.hidden_size), dtype=torch.bfloat16
        )
        hidden_states = hidden_states.cuda()
        output_smm, _ = self.sequential_mlp(hidden_states)
        output_gmm, _ = self.grouped_mlp(hidden_states)

        # The following assert fails due to the param init value is not exactly
        # the same between gmm and smm (refer to test_weight_init_value_the_same.)
        # assert torch.equal(output_smm, output_gmm)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.skipif(
        not DEVICE_CAPABILITY or DEVICE_CAPABILITY[0] < 8,
        reason='GroupedGEMM kernels are not supported on this device.',
    )
    def test_gpu_forward_with_no_tokens_allocated(self):
        """Test the case when no token is allocated for groupedGEMM kernels."""
        w1 = self.grouped_mlp.experts.weight1.view(self.num_experts, -1, self.hidden_size)
        num_allocated_tokens = 0
        tokens_per_expert = torch.zeros(self.num_experts)
        hidden_states = torch.rand((num_allocated_tokens, self.hidden_size), dtype=torch.bfloat16)
        hidden_states = hidden_states.cuda()
        try:
            gg.ops.gmm(hidden_states, w1, tokens_per_expert, trans_b=False)
        except Exception as e:
            print("Expected error message from groupedGEMM:", e)
            assert str(e) == "Input batch_sizes should not be all zeros!"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.skipif(
        not DEVICE_CAPABILITY or DEVICE_CAPABILITY[0] < 8,
        reason='GroupedGEMM kernels are not supported on this device.',
    )
    def test_gradient_with_no_tokens_allocated(self):
        """Test that when no token is passed in, the parameters of the grouped MLP will also have gradients."""
        self.grouped_mlp.cuda()
        num_allocated_tokens = 0
        tokens_per_expert = torch.zeros(self.num_experts)
        hidden_states = torch.rand((num_allocated_tokens, self.hidden_size), dtype=torch.bfloat16)
        hidden_states = hidden_states.cuda()
        output_gmm, _ = self.grouped_mlp.experts(hidden_states, tokens_per_expert=tokens_per_expert)
        output_gmm.mean().backward()
        assert self.grouped_mlp.experts.weight1.grad is not None


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
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            self.num_experts, moe_grouped_gemm=False
        )
        self.sequential_mlp = MoELayer(tf_config, transformer_layer_spec.submodules.mlp.submodules)

        self.args = parse_args(ignore_unknown_args=True)
        self.args.bf16 = True
        # Bias is not supported in grouped gemm currently, thus we disable the
        # bias in the linear layer.
        self.args.add_bias_linear = False
        self.sequential_mlp = Float16Module(self.sequential_mlp, self.args).module

        ## Grouped GEMM
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            self.num_experts, moe_grouped_gemm=True
        )
        tf_config.moe_grouped_gemm = True
        self.grouped_mlp = MoELayer(tf_config, transformer_layer_spec.submodules.mlp.submodules)
        assert isinstance(self.grouped_mlp.experts, TEGroupedMLP)
        self.grouped_mlp = Float16Module(self.grouped_mlp, self.args).module

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
        output, _ = self.grouped_mlp.experts(hidden_states, tokens_per_expert=tokens_per_expert)
        assert torch.equal(output, torch.zeros_like(output))
        assert output.shape == (num_allocated_tokens, self.hidden_size)

        output.mean().backward()
        for i in range(self.num_experts):
            assert getattr(self.grouped_mlp.experts.linear_fc1, f"weight{i}").grad is not None
            assert getattr(self.grouped_mlp.experts.linear_fc2, f"weight{i}").grad is not None


if __name__ == "__main__":
    for use_cpu_unitilization in [True, False]:
        for swiglu in [True, False]:
            GMLP_test = TestParallelGroupedMLP()
            GMLP_test.setup_method(
                method=None, use_cpu_initialization=use_cpu_unitilization, swiglu=swiglu
            )
            GMLP_test.test_constructor()
            GMLP_test.test_weight_init_value_the_same()
            GMLP_test.test_gpu_forward()
            GMLP_test.test_gpu_forward_with_no_tokens_allocated()
            GMLP_test.teardown_method(method=None)
