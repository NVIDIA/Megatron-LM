# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.transformer.moe.moe_utils import clear_aux_losses_tracker
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.moe.test_token_dispatcher import MoEModelTestContainer


class AuxlossTestContainer(MoEModelTestContainer):
    def partition_input(self, input):
        partitioned_input = input.chunk(
            parallel_state.get_tensor_and_context_parallel_world_size(), dim=0
        )[parallel_state.get_tensor_and_context_parallel_rank()]
        output = partitioned_input.clone().detach()
        output.requires_grad = True
        return output

    @pytest.mark.internal
    def aux_loss_test(self, input, baseline_grad):
        partitioned_input = self.partition_input(input)
        moe_layer = self.moe_layer
        probs, indices = moe_layer.router(partitioned_input)
        probs.sum().mul_(0).backward()
        aux_loss_grad = partitioned_input.grad
        torch.distributed.barrier()
        ans = self.partition_input(baseline_grad)
        assert torch.allclose(aux_loss_grad, ans), f"Diff: {(aux_loss_grad/ans).mean()}"
        loss = parallel_state.get_moe_layer_wise_logging_tracker()['load_balancing_loss']
        clear_aux_losses_tracker()


class TestAuxLoss:
    def setup_method(self, method):
        baseline_container = AuxlossTestContainer(
            tp_size=1,
            ep_size=1,
            pp_size=1,
            cp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="alltoall",
            moe_aux_loss_coeff=0.1,
        )
        moe_layer = baseline_container.moe_layer
        self.input = torch.randn((32, 8, moe_layer.config.hidden_size)).cuda()
        self.input.requires_grad = True
        probs, indices = moe_layer.router(self.input)
        probs.sum().mul_(0).backward()  # zero out the main gradients
        self.baseline_grad = self.input.grad
        self.input.grad = None
        clear_aux_losses_tracker()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.parametrize(
        "tp_size,ep_size,cp_size", [(8, 1, 1), (4, 2, 1), (1, 1, 8), (2, 1, 4), (2, 2, 2)]
    )
    def test_allgather_dispatcher(self, tp_size, ep_size, cp_size):
        container = AuxlossTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            cp_size=cp_size,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="allgather",
            moe_aux_loss_coeff=0.1,
        )
        container.aux_loss_test(self.input, self.baseline_grad)

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.parametrize(
        "tp_size,ep_size,cp_size", [(8, 1, 1), (4, 2, 1), (1, 1, 8), (2, 1, 4), (2, 2, 2)]
    )
    def test_a2a_dispatcher(self, tp_size, ep_size, cp_size):
        container = AuxlossTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            cp_size=cp_size,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="alltoall",
            moe_aux_loss_coeff=0.1,
        )
        container.aux_loss_test(self.input, self.baseline_grad)


class TestSeqAuxLoss:
    def setup_method(self, method):
        baseline_container = AuxlossTestContainer(
            tp_size=1,
            ep_size=1,
            pp_size=1,
            cp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="seq_aux_loss",
            moe_token_dispatcher_type="alltoall",
            moe_aux_loss_coeff=0.1,
        )
        moe_layer = baseline_container.moe_layer
        self.input = torch.randn((32, 8, moe_layer.config.hidden_size)).cuda()
        self.input.requires_grad = True
        probs, indices = moe_layer.router(self.input)
        probs.sum().mul_(0).backward()  # zero out the main gradients
        self.baseline_grad = self.input.grad
        self.input.grad = None
        clear_aux_losses_tracker()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.parametrize(
        "tp_size,ep_size,cp_size", [(8, 1, 1), (4, 2, 1), (1, 1, 8), (2, 1, 4), (2, 2, 2)]
    )
    def test_a2a_dispatcher(self, tp_size, ep_size, cp_size):
        container = AuxlossTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            cp_size=cp_size,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="seq_aux_loss",
            moe_token_dispatcher_type="alltoall",
            moe_aux_loss_coeff=0.1,
        )
        container.aux_loss_test(self.input, self.baseline_grad)
