# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from types import SimpleNamespace

import pytest
import torch

from megatron.core.tensor_parallel.layers import (
    _expert_grads_need_own_dp_domain,
    linear_with_frozen_weight,
)
from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region
from tests.unit_tests.test_utilities import Utils


def test_expert_grads_need_own_dp_domain_etp_lt_tp():
    """EP=1 / ETP=1 / TP=2 expert wgrad must leave the dense dp_cp bucket."""
    frozen = SimpleNamespace(
        expert_model_parallel_size=1,
        tensor_model_parallel_size=2,
        expert_tensor_parallel_size=1,
        dsa_accuracy_compatible=True,
    )
    assert _expert_grads_need_own_dp_domain(frozen) is True
    frozen.dsa_accuracy_compatible = False
    assert _expert_grads_need_own_dp_domain(frozen) is False
    eq = SimpleNamespace(
        expert_model_parallel_size=1, tensor_model_parallel_size=2, expert_tensor_parallel_size=2
    )
    assert _expert_grads_need_own_dp_domain(eq) is False
    ep2 = SimpleNamespace(
        expert_model_parallel_size=2, tensor_model_parallel_size=2, expert_tensor_parallel_size=1
    )
    assert _expert_grads_need_own_dp_domain(ep2) is True
    missing = SimpleNamespace(
        expert_model_parallel_size=1, tensor_model_parallel_size=2, expert_tensor_parallel_size=None
    )
    assert _expert_grads_need_own_dp_domain(missing) is False


@pytest.mark.parametrize("tensor_parallel,allreduce_dgrad", [(1, False), (8, True)])
def test_LinearWithFrozenWeight(tensor_parallel, allreduce_dgrad):
    Utils.initialize_model_parallel(tensor_parallel, 1)

    size_per_partition = int(8 / tensor_parallel)

    # Input is an 8x8 identity matrix.
    input_data = torch.eye(8).cuda()
    input_data.requires_grad = True

    # Weight is an 8x8 matrix of all ones. If tensor parallelism > 1, the weight is partitioned evenly across GPUs.
    weight = torch.ones((size_per_partition, 8)).cuda()

    # Bias is a vector of length 8 of all zeros. If tensor parallelism > 1, the bias is partitioned evenly across GPUs
    bias = torch.zeros((size_per_partition)).cuda()

    gradient_accumulation_fusion = False
    sequence_parallel = False
    grad_output_buffer = None
    wgrad_deferral_limit = None

    output_parallel = linear_with_frozen_weight(
        input_data,
        weight,
        bias,
        gradient_accumulation_fusion,
        allreduce_dgrad,
        sequence_parallel,
        grad_output_buffer,
        wgrad_deferral_limit,
    )
    output = gather_from_tensor_model_parallel_region(
        output_parallel
    )  # no-op if tensor_parallel == 1.
    output.sum().backward()

    expected_output = torch.ones(8).cuda()
    expected_grad = 8 * torch.ones(8).cuda()

    assert torch.allclose(output, expected_output)
    assert torch.allclose(input_data.grad, expected_grad)

    Utils.destroy_model_parallel()


def test_LinearWithFrozenWeight_3d_input_matches_torch_linear():
    Utils.initialize_model_parallel(1, 1)

    input_data = torch.randn(4, 3, 8, device="cuda", requires_grad=True)
    weight = torch.randn(6, 8, device="cuda")
    bias = torch.randn(6, device="cuda")

    expected_input = input_data.detach().clone().requires_grad_(True)
    expected = torch.nn.functional.linear(expected_input, weight, bias)
    expected.sum().backward()

    actual = linear_with_frozen_weight(input_data, weight, bias, False, False, False, None, None)
    actual.sum().backward()

    assert torch.allclose(actual, expected)
    assert torch.allclose(input_data.grad, expected_input.grad)

    Utils.destroy_model_parallel()
