# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
from importlib.metadata import version

import pytest
import torch

from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TERowParallelLinear
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.experts import SequentialMLP
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.moe_utils import get_default_model_comm_pgs
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils


class TestParallelSequentialMLP:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        print("done intializing")
        num_moe_experts = 2
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=True,
            activation_func=torch.nn.functional.silu,
            gated_linear_unit=True,
            bias_activation_fusion=True,
            moe_router_load_balancing_type="sinkhorn",
            moe_router_topk=1,
            add_bias_linear=False,
        )
        transformer_layer_spec = get_gpt_layer_local_spec(
            num_experts=num_moe_experts, moe_grouped_gemm=False
        )
        self.sequential_mlp = MoELayer(
            transformer_config, transformer_layer_spec.submodules.mlp.submodules
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_constructor(self):
        assert isinstance(self.sequential_mlp, MoELayer)

        num_weights = sum([p.numel() for p in self.sequential_mlp.parameters()])
        assert num_weights == 3480

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_forward(self):
        sequential_mlp = self.sequential_mlp
        sequential_mlp.cuda()
        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones((32, 2, sequential_mlp.config.hidden_size))
        hidden_states = hidden_states.cuda()
        output, output_bias = sequential_mlp(hidden_states)
        assert output.shape[0] == 32
        assert output.shape[1] == 2
        assert output.shape[2] == sequential_mlp.config.hidden_size
        assert output.dtype == torch.float32
        assert output.device.type == 'cuda'


class TestTEParallelSequentialMLP:
    def setup_method(self, method):
        Utils.initialize_model_parallel(tensor_model_parallel_size=2, expert_model_parallel_size=2)
        model_parallel_cuda_manual_seed(123)
        num_moe_experts = 4
        model_comm_pgs = get_default_model_comm_pgs()
        self.transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=False,
            activation_func=torch.nn.functional.silu,
            gated_linear_unit=True,
            bias_activation_fusion=False,
            moe_router_load_balancing_type="sinkhorn",
            moe_router_topk=1,
            params_dtype=torch.bfloat16,
            expert_model_parallel_size=2,
            tensor_model_parallel_size=2,
            sequence_parallel=True,
            add_bias_linear=False,
        )

        self.local_mlp_spec = MLPSubmodules(
            linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear
        )
        self.te_mlp_spec = MLPSubmodules(
            linear_fc1=TEColumnParallelLinear, linear_fc2=TERowParallelLinear
        )
        print("Done intializing")

        self.num_local_experts = 2
        model_parallel_cuda_manual_seed(123)
        self.local_sequential_mlp = SequentialMLP(
            self.num_local_experts,
            self.transformer_config,
            self.local_mlp_spec,
            model_comm_pgs=model_comm_pgs,
        )

        model_parallel_cuda_manual_seed(123)
        self.te_sequential_mlp = SequentialMLP(
            self.num_local_experts,
            self.transformer_config,
            self.te_mlp_spec,
            model_comm_pgs=model_comm_pgs,
        )

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_te_min_version("1.7.0"),
        reason="Transformer Engine under v1.7.0 doesn't support MoE training.",
    )
    @pytest.mark.internal
    def test_constructor(self):
        for i in range(self.num_local_experts):
            assert torch.equal(
                self.local_sequential_mlp.local_experts[i].linear_fc1.weight,
                self.te_sequential_mlp.local_experts[i].linear_fc1.weight,
            )
            assert torch.equal(
                self.local_sequential_mlp.local_experts[i].linear_fc2.weight,
                self.te_sequential_mlp.local_experts[i].linear_fc2.weight,
            )

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_te_min_version("1.7.0"),
        reason="Transformer Engine under v1.7.0 doesn't support MoE training.",
    )
    @pytest.mark.internal
    def test_gpu_forward(self):
        self.local_sequential_mlp.cuda()
        self.te_sequential_mlp.cuda()
        seq_len = 4
        batch_size = 2

        tokens_per_expert = torch.tensor([2, 2], device="cuda")
        hidden_states = torch.rand(
            (seq_len, batch_size, self.local_sequential_mlp.config.hidden_size),
            dtype=torch.bfloat16,
            device="cuda",
        )
        probs = torch.rand((seq_len, batch_size), dtype=torch.float32, device="cuda")

        output_local, _ = self.local_sequential_mlp(hidden_states, tokens_per_expert, probs)
        output_te, _ = self.te_sequential_mlp(hidden_states, tokens_per_expert, probs)
        assert torch.equal(output_local, output_te)

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_te_min_version("1.7.0"),
        reason="Transformer Engine under v1.7.0 doesn't support MoE training.",
    )
    @pytest.mark.internal
    def test_gpu_forward_with_one_local_expert(self):
        model_parallel_cuda_manual_seed(123)
        model_comm_pgs = get_default_model_comm_pgs()
        local_sequential_mlp = SequentialMLP(
            1, self.transformer_config, self.local_mlp_spec, model_comm_pgs=model_comm_pgs
        )
        model_parallel_cuda_manual_seed(123)
        te_sequential_mlp = SequentialMLP(
            1, self.transformer_config, self.te_mlp_spec, model_comm_pgs=model_comm_pgs
        )
        seq_len = 4
        batch_size = 2

        tokens_per_expert = torch.tensor([4], device="cuda")
        hidden_states = torch.rand(
            (seq_len, batch_size, self.local_sequential_mlp.config.hidden_size),
            dtype=torch.bfloat16,
            device="cuda",
        )
        probs = torch.rand((seq_len, batch_size), dtype=torch.float32, device="cuda")

        output_local, _ = local_sequential_mlp(hidden_states, tokens_per_expert, probs)
        output_te, _ = te_sequential_mlp(hidden_states, tokens_per_expert, probs)
        assert torch.equal(output_local, output_te)

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_te_min_version("1.7.0"),
        reason="Transformer Engine under v1.7.0 doesn't support MoE training.",
    )
    @pytest.mark.internal
    def test_gpu_forward_with_no_tokens_allocated(self):
        self.local_sequential_mlp.cuda()
        self.te_sequential_mlp.cuda()
        seq_len = 4
        batch_size = 2

        tokens_per_expert = torch.tensor([0, 4], device="cuda")
        hidden_states = torch.rand(
            (seq_len, batch_size, self.local_sequential_mlp.config.hidden_size),
            dtype=torch.bfloat16,
            device="cuda",
        )
        probs = torch.rand((seq_len, batch_size), dtype=torch.float32, device="cuda")

        output_local, _ = self.local_sequential_mlp(hidden_states, tokens_per_expert, probs)
        output_te, _ = self.te_sequential_mlp(hidden_states, tokens_per_expert, probs)
        assert torch.equal(output_local, output_te)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()


if __name__ == "__main__":
    MLP_test = TestTEParallelSequentialMLP()
    MLP_test.setup_method(method=None)
    MLP_test.test_constructor()
    MLP_test.test_gpu_forward()
    MLP_test.test_gpu_forward_with_one_local_expert()
    MLP_test.test_gpu_forward_with_no_tokens_allocated()
    MLP_test.teardown_method(method=None)
