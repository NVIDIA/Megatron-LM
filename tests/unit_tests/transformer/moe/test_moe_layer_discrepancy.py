# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import time

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


class TestMoELayerDispatcherDiscrepancy:
    def setup_method(self, method):
        pass

    @pytest.mark.parametrize("num_moe_experts", [8])
    @pytest.mark.parametrize("grouped_gemm", [False])
    @pytest.mark.parametrize(
        "tp_size,ep_size", [(1, 1), (1, 2), (1, 8), (2, 1), (8, 1), (2, 2), (2, 4), (4, 2)]
    )
    @pytest.mark.internal
    def test_moe_layer_dispatcher_discrepancy(
        self, num_moe_experts, grouped_gemm, tp_size, ep_size
    ):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, expert_model_parallel_size=ep_size
        )
        self.transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=4096,
            num_attention_heads=32,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=False,
            moe_token_dispatcher_type="allgather",
            moe_router_topk=2,
            moe_aux_loss_coeff=0.01,
            moe_grouped_gemm=grouped_gemm,
            moe_router_dtype="fp64",
            add_bias_linear=False,
            tensor_model_parallel_size=tp_size,
            expert_model_parallel_size=ep_size,
            sequence_parallel=True if (tp_size > 1) else False,
        )
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=num_moe_experts, moe_grouped_gemm=grouped_gemm
        )
        # Init input and layer
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        input = torch.randn(1, 4096, 4096).cuda().float()

        # Init allgather moe layer
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        layer = (
            TransformerLayer(self.transformer_config, transformer_layer_spec.submodules)
            .cuda()
            .float()
        )
        ag_moe_layer = layer.mlp
        ag_moe_layer.eval()
        # Init a2a moe layer
        self.transformer_config.moe_token_dispatcher_type = "alltoall"
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        layer = (
            TransformerLayer(self.transformer_config, transformer_layer_spec.submodules)
            .cuda()
            .float()
        )
        a2a_moe_layer = layer.mlp
        a2a_moe_layer.eval()

        # Check if parameters are the same
        for ag_param, a2a_param in zip(ag_moe_layer.parameters(), a2a_moe_layer.parameters()):
            assert torch.equal(ag_param, a2a_param)
        torch.distributed.barrier()

        # Allgather the input to check if the input is the same in all the ranks
        # Check if input is the same across all ranks
        input_ag_shape = (torch.distributed.get_world_size(), *(input.shape))
        input_ag = torch.zeros(input_ag_shape, device=input.device, dtype=input.dtype)
        torch.distributed.all_gather_into_tensor(
            input_ag, input, group=torch.distributed.group.WORLD
        )
        if torch.distributed.get_rank() == 0:
            for i in range(1, torch.distributed.get_world_size()):
                assert torch.equal(input_ag[0], input_ag[i]), f"Input differs at rank {i}"
            # print(f"Input is the same across all ranks")

        # Test allgather dispatcher
        with torch.no_grad():
            ag_output = ag_moe_layer(input)[0]
            a2a_output = a2a_moe_layer(input)[0]

        assert torch.allclose(
            ag_output, a2a_output, atol=1e-6
        ), f"Ag output: {ag_output.min()}, {ag_output.max()}, {ag_output.sum()}, a2a output: {a2a_output.min()}, {a2a_output.max()}, {a2a_output.sum()}, diff: {torch.abs(ag_output - a2a_output).max()}"
        # print(f"Allgather and A2A output is the same", flush=True)

        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("num_moe_experts", [8])
    @pytest.mark.parametrize("grouped_gemm", [False, True])
    @pytest.mark.parametrize(
        "tp_size,ep_size", [(1, 1), (1, 2), (1, 8), (2, 1), (8, 1), (2, 2), (2, 4), (4, 2)]
    )
    @pytest.mark.internal
    def test_moe_layer_ag_dispatcher_discrepancy(
        self, num_moe_experts, grouped_gemm, tp_size, ep_size
    ):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, expert_model_parallel_size=ep_size
        )
        self.transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=4096,
            num_attention_heads=32,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=False,
            moe_token_dispatcher_type="allgather",
            moe_router_topk=2,
            moe_aux_loss_coeff=0.01,
            moe_grouped_gemm=grouped_gemm,
            moe_router_dtype="fp64",
            add_bias_linear=False,
            tensor_model_parallel_size=tp_size,
            expert_model_parallel_size=ep_size,
            sequence_parallel=True if (tp_size > 1 and ep_size > 1) else False,
            bf16=True,
        )
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=num_moe_experts, moe_grouped_gemm=grouped_gemm
        )
        # Init input and layer
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        input = torch.randn(1, 4096, 4096).cuda().bfloat16()

        # Init allgather moe layer
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        layer = (
            TransformerLayer(self.transformer_config, transformer_layer_spec.submodules)
            .cuda()
            .bfloat16()
        )
        ag_moe_layer = layer.mlp
        ag_moe_layer.eval()

        # Test allgather dispatcher
        ag_output = ag_moe_layer(input)[0]
        # Allgather the output to check if it's the same in all ranks
        ag_output_ag_shape = (torch.distributed.get_world_size(), *(ag_output.shape))
        ag_output_ag = torch.zeros(
            ag_output_ag_shape, device=ag_output.device, dtype=ag_output.dtype
        )
        torch.distributed.all_gather_into_tensor(
            ag_output_ag, ag_output, group=torch.distributed.group.WORLD
        )
        # Check if output is the same across all ranks
        if parallel_state.get_data_parallel_rank() == 0:
            for i in range(1, parallel_state.get_tensor_model_parallel_world_size()):
                if not torch.allclose(ag_output_ag[0], ag_output_ag[i]):
                    print(f"Allgather output differs at rank {torch.distributed.get_rank()}")
                    print(
                        f"ag_output_ag[0]: min {ag_output_ag[0].double().min()}, max {ag_output_ag[0].double().max()}, std {ag_output_ag[0].double().std()}"
                    )
                    print(
                        f"ag_output_ag[{i}]: min {ag_output_ag[i].double().min()}, max {ag_output_ag[i].double().max()}, std {ag_output_ag[i].double().std()}"
                    )
                    raise ValueError("Allgather output differs at rank {i}")
        torch.cuda.synchronize()
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("num_moe_experts", [8])
    @pytest.mark.parametrize("grouped_gemm", [False])
    @pytest.mark.parametrize(
        "tp_size,ep_size", [(1, 1), (1, 2), (1, 8), (2, 1), (4, 1), (8, 1), (2, 4), (4, 2)]
    )
    @pytest.mark.internal
    def test_moe_layer_a2a_dispatcher_discrepancy(
        self, num_moe_experts, grouped_gemm, tp_size, ep_size
    ):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, expert_model_parallel_size=ep_size
        )
        self.transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=4096,
            num_attention_heads=32,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=False,
            moe_token_dispatcher_type="alltoall",
            moe_router_topk=2,
            moe_aux_loss_coeff=0.01,
            moe_grouped_gemm=grouped_gemm,
            moe_router_dtype="fp64",
            add_bias_linear=False,
            tensor_model_parallel_size=tp_size,
            expert_model_parallel_size=ep_size,
            sequence_parallel=True if (tp_size > 1 and ep_size > 1) else False,
            bf16=True,
        )
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=num_moe_experts, moe_grouped_gemm=grouped_gemm
        )
        # Init input and layer
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        input = torch.randn(1, 4096, 4096).cuda().bfloat16()

        # Init a2a moe layer
        layer = (
            TransformerLayer(self.transformer_config, transformer_layer_spec.submodules)
            .cuda()
            .bfloat16()
        )
        a2a_moe_layer = layer.mlp
        a2a_moe_layer.eval()

        # Test alltoall dispatcher
        a2a_output = a2a_moe_layer(input)[0]
        # Allgather the output to check if it's the same in all ranks
        at_output_ag_shape = (torch.distributed.get_world_size(), *(a2a_output.shape))
        at_output_ag = torch.zeros(
            at_output_ag_shape, device=a2a_output.device, dtype=a2a_output.dtype
        )
        torch.distributed.all_gather_into_tensor(
            at_output_ag, a2a_output, group=torch.distributed.group.WORLD
        )
        # Check if output is the same across all ranks
        if parallel_state.get_data_parallel_rank() == 0:
            for i in range(1, parallel_state.get_tensor_model_parallel_world_size()):
                if not torch.equal(at_output_ag[0], at_output_ag[i]):
                    print(
                        f"at_output_ag[0]: min {at_output_ag[0].double().min()}, max {at_output_ag[0].double().max()}, sum {at_output_ag[0].double().sum()}"
                    )
                    print(
                        f"at_output_ag[{i}]: min {at_output_ag[i].double().min()}, max {at_output_ag[i].double().max()}, sum {at_output_ag[i].double().sum()}"
                    )
                    print(f"diff {torch.abs(at_output_ag[0] - at_output_ag[i]).max()}")
                    print(f"A2A output differs at rank {torch.distributed.get_rank()}")
                    raise ValueError(f"A2A output differs at rank {i}")
        torch.cuda.synchronize()

        Utils.destroy_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
