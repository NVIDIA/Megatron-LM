# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import time

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


class TestMoELayerDispatcherDiscrepancy:
    def setup_method(self, method):
        pass

    @pytest.mark.parametrize("num_moe_experts", [None, 8])
    @pytest.mark.parametrize("grouped_gemm", [False])
    @pytest.mark.parametrize("tp_size,ep_size", [(4, 1)])
    @pytest.mark.internal
    def test_moe_layer_dispatcher_discrepancy(
        self, num_moe_experts, grouped_gemm, tp_size, ep_size
    ):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, expert_model_parallel_size=ep_size
        )
        # Init input and layer
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        input = torch.randn(1, 4096, 4096).cuda().bfloat16()

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
            sequence_parallel=False,
            bf16=True,
        )
        transformer_layer_spec = get_gpt_layer_local_spec(
            num_experts=num_moe_experts, moe_grouped_gemm=grouped_gemm
        )
        layer = (
            TransformerLayer(self.transformer_config, transformer_layer_spec.submodules)
            .cuda()
            .bfloat16()
        )
        moe_layer = layer.mlp
        layer.eval()

        # # Print weight statistics for each rank
        # rank = torch.distributed.get_rank()

        # for name, param in moe_layer.named_parameters():
        #     # Convert to fp64 for accurate statistics
        #     param_fp64 = param.to(torch.float64)
        #     param_min = param_fp64.min().item()
        #     param_max = param_fp64.max().item()
        #     param_abs_sum = param_fp64.abs().sum().item()
        #     param_mean = param_fp64.mean().item()

        #     torch.distributed.barrier()
        #     time.sleep(torch.distributed.get_rank() / 100)
        #     print(f"Rank {rank} | {name} | min: {param_min:.6e}, max: {param_max:.6e}, abs_sum: {param_abs_sum:.6e}, mean: {param_mean:.6e}")

        # torch.distributed.barrier()

        # Allgather the input to check if the input is the same in all the ranks
        # input_ag_shape = (torch.distributed.get_world_size(), *(input.shape))
        # input_ag = torch.zeros(input_ag_shape, device=input.device, dtype=input.dtype)
        # torch.distributed.all_gather_into_tensor(input_ag, input, group=torch.distributed.group.WORLD)

        # Check if input is the same across all ranks
        # if torch.distributed.get_rank() == 0:
        #     for i in range(1, torch.distributed.get_world_size()):
        #         assert torch.equal(input_ag[0], input_ag[i]), f"Input differs at rank {i}"
        #     print(f"Input is the same across all ranks")

        # Test allgather dispatcher
        moe_layer.config.moe_token_dispatcher_type = "allgather"
        with torch.no_grad():
            ag_output = moe_layer(input)[0]
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
                if not torch.equal(ag_output_ag[0], ag_output_ag[i]):
                    print(f"Allgather output differs at rank {torch.distributed.get_rank()}")
                    raise ValueError("Allgather output differs at rank {i}")
            print(f"Allgather output is the same across all ranks", flush=True)
        torch.cuda.synchronize()

        # Test alltoall dispatcher
        moe_layer.config.moe_token_dispatcher_type = "alltoall"
        with torch.no_grad():
            at_output = moe_layer(input)[0]
        # Allgather the output to check if it's the same in all ranks
        at_output_ag_shape = (torch.distributed.get_world_size(), *(at_output.shape))
        at_output_ag = torch.zeros(
            at_output_ag_shape, device=at_output.device, dtype=at_output.dtype
        )
        torch.distributed.all_gather_into_tensor(
            at_output_ag, at_output, group=torch.distributed.group.WORLD
        )
        # Check if output is the same across all ranks
        if parallel_state.get_data_parallel_rank() == 0:
            for i in range(1, parallel_state.get_tensor_model_parallel_world_size()):
                if not torch.equal(at_output_ag[0], at_output_ag[i]):
                    print(f"A2A output differs at rank {torch.distributed.get_rank()}")
                    raise ValueError("A2A output differs at rank {i}")
            print(f"A2A output is the same across all ranks", flush=True)
        torch.cuda.synchronize()

        assert torch.equal(ag_output, at_output)
        print(f"Allgather and A2A output is the same", flush=True)

        Utils.destroy_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
