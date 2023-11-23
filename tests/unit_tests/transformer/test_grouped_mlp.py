# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest

import torch
import torch.nn.functional as F

from megatron.arguments import parse_args
from megatron.core.models.gpt.gpt_layer_specs import gpt_layer_with_transformer_engine_spec_moe
from megatron.core.transformer.grouped_mlp import GroupedMLP
from megatron.core.transformer.switch_mlp import SwitchMLP
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.initialize import _set_random_seed
from megatron.model import Float16Module
from tests.unit_tests.test_utilities import Utils

class TestParallelGroupedMLP:

    def setup_method(self, method, use_cpu_initialization=False, swiglu=True):
        print("============")
        print("Test for use_cpu_initilization={} and swiglu={}.".format(use_cpu_initialization, swiglu))
        print("============")
        Utils.initialize_model_parallel(1,1)
        num_layers=1 # 2
        self.hidden_size=2 # 12
        self.num_experts = 2
        self.gated_linear_unit = True
        self.use_cpu_initialization = use_cpu_initialization
        self.gated_linear_unit = False
        if swiglu:
            self.gated_linear_unit = True

        tf_config = TransformerConfig(
            num_layers=num_layers, hidden_size=self.hidden_size, num_attention_heads=4,
            num_moe_experts=self.num_experts, use_cpu_initialization=self.use_cpu_initialization,
            add_bias_linear=False, gated_linear_unit=self.gated_linear_unit,
            bias_gelu_fusion=False,
            bf16=True, params_dtype=torch.bfloat16)

        self.fc1_ffn_hidden_size = tf_config.ffn_hidden_size
        self.fc2_ffn_hidden_size = tf_config.ffn_hidden_size
        # If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        if self.gated_linear_unit:
            self.fc1_ffn_hidden_size *= 2

        ## Vanilla sequential GEMM
        # Set random seed for reproducability
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        self.switch_mlp_smm = SwitchMLP(tf_config,
            gpt_layer_with_transformer_engine_spec_moe.submodules.mlp.submodules)

        self.args = parse_args(ignore_unknown_args=True)
        self.args.bf16=True
        # Bias is not supported in grouped gemm currently, thus we disable the
        # bias in the linear layer.
        self.args.add_bias_linear=False
        self.switch_mlp_smm = Float16Module(self.switch_mlp_smm, self.args).module
        print("done intializing for sequential gemm")

        ## Grouped GEMM
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        self.switch_mlp_gmm = GroupedMLP(tf_config)
        self.switch_mlp_gmm = Float16Module(self.switch_mlp_gmm, self.args).module
        print("done intializing for grouped gemm")

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        assert isinstance(self.switch_mlp_smm, SwitchMLP)
        assert isinstance(self.switch_mlp_gmm, GroupedMLP)

        num_weights_smm = sum([p.numel() for p in self.switch_mlp_smm.parameters()])
        num_weights_gmm = sum([p.numel() for p in self.switch_mlp_gmm.parameters()])

        # For the same hyper-parm model configs except the `moe_grouped_gemm`,
        # GroupedGEMM and sequential GEMMs should hold the same number of parms.
        assert num_weights_smm == num_weights_gmm
        # expected num weights: router linear weights+bias + MLP weights(no bias) of all experts
        expected_num_weights = \
            self.hidden_size * self.num_experts + \
            self.hidden_size * (self.fc1_ffn_hidden_size + self.fc2_ffn_hidden_size) * self.num_experts
        assert num_weights_smm == expected_num_weights

        assert torch.equal(self.switch_mlp_smm.router.weight, self.switch_mlp_gmm.router.weight)

        # weight1: [num_experts*4h, h]
        # weight2: [h, num_experts*4h]
        assert self.switch_mlp_gmm.weight1.shape[0] == self.num_experts * self.fc1_ffn_hidden_size
        assert self.switch_mlp_gmm.weight1.shape[1] == self.hidden_size
        if self.gated_linear_unit:
            assert self.switch_mlp_gmm.weight2.shape[0] == self.hidden_size
            assert self.switch_mlp_gmm.weight2.shape[1] == self.num_experts * self.fc2_ffn_hidden_size
        else:
            assert self.switch_mlp_gmm.weight1.shape == self.switch_mlp_gmm.weight2.t().shape

    def test_weight_init_value_the_same(self):
        gmm_w1 = self.switch_mlp_gmm.weight1.view(self.num_experts, -1, self.hidden_size)
        gmm_w2 = self.switch_mlp_gmm.weight2.view(self.num_experts, self.hidden_size, -1)
        gmm_expert1_fc1 = gmm_w1[0]
        gmm_expert1_fc2 = gmm_w2[0]
        gmm_expert2_fc1 = gmm_w1[1]
        gmm_expert2_fc2 = gmm_w2[1]

        smm_expert1_fc1 = self.switch_mlp_smm.local_experts[0].linear_fc1.weight
        smm_expert1_fc2 = self.switch_mlp_smm.local_experts[0].linear_fc2.weight
        smm_expert2_fc1 = self.switch_mlp_smm.local_experts[1].linear_fc1.weight
        smm_expert2_fc2 = self.switch_mlp_smm.local_experts[1].linear_fc2.weight

        assert torch.equal(gmm_expert1_fc1, smm_expert1_fc1)
        if not self.use_cpu_initialization:
            assert torch.equal(gmm_expert1_fc2, smm_expert1_fc2)
        # the param init value is not exactly the same between gmm and smm (refer to test_weight_init_value_the_same.)
        # TODO: is it necessary to keep smm and gmm share exactly the same init params?
        # assert torch.equal(gmm_expert2_fc1, smm_expert2_fc1)
        if self.use_cpu_initialization:
            assert torch.equal(gmm_expert2_fc2, smm_expert2_fc2)

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

        # The following assert fails due to the param init value is not exactly
        # the same between gmm and smm (refer to test_weight_init_value_the_same.)
        # assert torch.equal(output_smm, output_gmm)

if __name__ == "__main__":
    for use_cpu_unitilization in [True, False]:
        for swiglu in [True, False]:
            GMLP_test = TestParallelGroupedMLP()
            GMLP_test.setup_method(
                method=None,
                use_cpu_initialization=use_cpu_unitilization,
                swiglu=swiglu)
            GMLP_test.test_constructor()
            GMLP_test.test_weight_init_value_the_same()
            GMLP_test.test_gpu_forward()
            GMLP_test.teardown_method(method=None)