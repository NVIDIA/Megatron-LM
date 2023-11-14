# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest

import torch

from megatron.arguments import parse_args
from megatron.core.models.gpt.gpt_layer_specs import gpt_layer_with_transformer_engine_spec_moe
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.switch_mlp import SwitchMLP
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.model import Float16Module
from tests.unit_tests.test_utilities import Utils

class TestParallelSwitchMLP:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1,1)
        num_layers=1 # 2
        self.hidden_size=2 # 12
        self.num_experts = 2

        # Vanilla sequential GEMM
        model_parallel_cuda_manual_seed(123)
        tf_config_smm = TransformerConfig(
            num_layers=num_layers, hidden_size=self.hidden_size, num_attention_heads=4,
            num_moe_experts=self.num_experts, use_cpu_initialization=False, add_bias_linear=False,
            bf16=True, params_dtype=torch.bfloat16,
            moe_grouped_gemm=False)
        self.switch_mlp_smm = SwitchMLP(tf_config_smm,
            gpt_layer_with_transformer_engine_spec_moe.submodules.mlp.submodules)

        self.args = parse_args(extra_args_provider=None, ignore_unknown_args=False)
        self.args.bf16=True
        # Bias is not supported in grouped gemm currently, thus we disable the
        # bias in the linear layer.
        self.args.add_bias_linear=False
        self.switch_mlp_smm = Float16Module(self.switch_mlp_smm, self.args).module
        print("done intializing for sequential gemm")

        # Grouped GEMM
        model_parallel_cuda_manual_seed(123)
        tf_config_gmm = TransformerConfig(
            num_layers=num_layers, hidden_size=self.hidden_size, num_attention_heads=4,
            num_moe_experts=self.num_experts, use_cpu_initialization=False, add_bias_linear=False,
            bf16=True, # Currently GroupedGEMM only supports bf16.
            params_dtype=torch.bfloat16,
            moe_grouped_gemm=True)
        self.switch_mlp_gmm = SwitchMLP(tf_config_gmm,
            gpt_layer_with_transformer_engine_spec_moe.submodules.mlp.submodules)
        self.switch_mlp_gmm = Float16Module(self.switch_mlp_gmm, self.args).module
        print("done intializing for grouped gemm")

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        assert isinstance(self.switch_mlp_smm, SwitchMLP)
        assert isinstance(self.switch_mlp_gmm, SwitchMLP)

        num_weights_smm = sum([p.numel() for p in self.switch_mlp_smm.parameters()])
        num_weights_gmm = sum([p.numel() for p in self.switch_mlp_gmm.parameters()])

        # For the same hyper-parm model configs except the `moe_grouped_gemm`,
        # GroupedGEMM and sequential GEMMs should hold the same number of parms.
        assert num_weights_smm == num_weights_gmm

        # TODO: The param init value is not exactly the same between gmm and smm
        # assert torch.equal(self.switch_mlp_smm.router.weight, self.switch_mlp_gmm.router.weight)
        # assert num_weights_smm == 2330, 'num_weights_sm=', num_weights_smm

        # weight1: [num_experts*4h, h]
        # weight2: [num_experts, h, 4h]
        assert self.switch_mlp_gmm.weight1.shape[0] == self.num_experts * 4 * self.hidden_size
        assert self.switch_mlp_gmm.weight1.shape[1] == self.hidden_size
        assert self.switch_mlp_gmm.weight1.shape == \
            self.switch_mlp_gmm.weight2.t().shape

    def test_weight_init_value_the_same(self):
        gmm_w1 = self.switch_mlp_gmm.weight1.view(self.num_experts, -1, self.hidden_size)
        gmm_w2 = self.switch_mlp_gmm.weight2.view(self.num_experts, self.hidden_size, -1)
        gmm_expert0_fc1 = gmm_w1[0]
        gmm_expert0_fc2 = gmm_w2[0]
        gmm_expert1_fc1 = gmm_w1[1]
        gmm_expert1_fc2 = gmm_w2[1]

        smm_expert0_fc1 = self.switch_mlp_smm.local_experts[0].linear_fc1.weight
        smm_expert0_fc2 = self.switch_mlp_smm.local_experts[0].linear_fc2.weight
        smm_expert1_fc1 = self.switch_mlp_smm.local_experts[1].linear_fc1.weight
        smm_expert1_fc2 = self.switch_mlp_smm.local_experts[1].linear_fc2.weight

        assert torch.equal(gmm_expert0_fc1, smm_expert0_fc1)
        assert torch.equal(gmm_expert0_fc2, smm_expert0_fc2)
        # the param init value is not exactly the same between gmm and smm (refer to test_weight_init_value_the_same.)
        # TODO: is it necessary to keep smm and gmm share exactly the same init params?
        # assert torch.equal(gmm_expert1_fc1, smm_expert1_fc1)
        # assert torch.equal(gmm_expert1_fc2, smm_expert1_fc2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_forward(self):
        self.switch_mlp_smm.cuda()
        self.switch_mlp_gmm.cuda()
        # [sequence length, batch size, hidden size]
        seq_len = 3 #32
        batch_size = 2
        hidden_states = torch.ones(
            (seq_len, batch_size, self.switch_mlp_smm.config.hidden_size),
            dtype=torch.bfloat16)
        hidden_states = hidden_states.cuda()
        output_smm, _ = self.switch_mlp_smm(hidden_states)
        output_gmm, _ = self.switch_mlp_gmm(hidden_states)

        # The following assert fails due to two reasons:
        #   (i) the param init value is not exactly the same between gmm and smm (refer to test_weight_init_value_the_same.)
        #   (ii) the router weight init value is not fixed in this UT.
        # assert torch.equal(output_smm, output_gmm),print(output_smm, output_gmm)

if __name__ == "__main__":
    SMLP_test = TestParallelSwitchMLP()
    SMLP_test.setup_method(method=None)
    SMLP_test.test_constructor()
    SMLP_test.test_weight_init_value_the_same()
    SMLP_test.test_gpu_forward()
    SMLP_test.teardown_method(method=None)